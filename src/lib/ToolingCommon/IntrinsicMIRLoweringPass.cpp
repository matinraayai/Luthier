//===-- IntrinsicMIRLoweringPass.cpp --------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the Intrinsic MIR Lowering Pass.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IntrinsicMIRLoweringPass.h"
#include "luthier/Tooling/IntrinsicProcessorsAnalysis.h"
#include "luthier/Tooling/MIRConvenience.h"
#include "luthier/Tooling/PhysicalRegAccessVirtualizationPass.h"
#include "luthier/Tooling/StateValueArraySpecs.h"
#include "luthier/Tooling/WrapperAnalysisPasses.h"
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>

namespace luthier {

char IntrinsicMIRLoweringPass::ID = 0;

LUTHIER_INITIALIZE_LEGACY_PASS_BODY(IntrinsicMIRLoweringPass, "mir-lowering",
                                    "Intrinsic MIR lowering pass",
                                    true /* Only looks at CFG */,
                                    false /* Analysis Pass */)

bool IntrinsicMIRLoweringPass::runOnMachineFunction(llvm::MachineFunction &MF) {
  bool Changed{false};

  auto &IModule = const_cast<llvm::Module &>(
      *getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI().getModule());

  auto &IMAM = getAnalysis<IModuleMAMWrapperPass>().getMAM();

  const auto &ToBeLoweredIntrinsics =
      IMAM.getCachedResult<IntrinsicIRLoweringInfoMapAnalysis>(IModule)
          ->getLoweringInfo();

  const auto &IntrinsicsProcessors =
      IMAM.getResult<IntrinsicsProcessorsAnalysis>(IModule);

  auto &TargetMAMAndModule =
      IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);

  auto &TargetMAM = TargetMAMAndModule.getTargetAppMAM();

  auto &TargetModule = TargetMAMAndModule.getTargetAppModule();

  auto &PreambleDescriptor =
      TargetMAM.getResult<FunctionPreambleDescriptorAnalysis>(TargetModule);

  auto &IPIP = IMAM.getResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  const auto &TargetMI = *IPIP.at(MF.getFunction());

  llvm::MCRegister SVAVGPR =
      TargetMAM
          .getResult<LRStateValueStorageAndLoadLocationsAnalysis>(TargetModule)
          .getStateValueArrayLoadPlanForInstPoint(TargetMI)
          ->StateValueArrayLoadVGPR;

  for (auto &MBB : MF) {
    const auto &RegAccessVirtualizationPass =
        getAnalysis<PhysicalRegAccessVirtualizationPass>();
    // Keeps track of the physical registers that have been written to inside
    // the current basic block
    llvm::DenseMap<llvm::MCRegister, llvm::Register> MBBOverwrittenPhysRegs;

    for (auto &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.isInlineAsm()) {
        auto IntrinsicIdx = getIntrinsicInlineAsmPlaceHolderIdx(MI);
        if (auto Err = IntrinsicIdx.takeError()) {
          MF.getContext().reportError({}, llvm::toString(std::move(Err)));
          return false;
        };
        if (*IntrinsicIdx == -1)
          continue;
        llvm::SmallVector<std::pair<llvm::InlineAsm::Flag, llvm::Register>, 4>
            ArgVec;
        for (unsigned I = llvm::InlineAsm::MIOp_FirstOperand,
                      NumOps = MI.getNumOperands();
             I < NumOps; ++I) {
          const llvm::MachineOperand &MO = MI.getOperand(I);
          if (!MO.isImm())
            continue;
          const llvm::InlineAsm::Flag F(MO.getImm());
          const llvm::Register Reg(MI.getOperand(I + 1).getReg());
          ArgVec.emplace_back(F, Reg);
          // Skip to one before the next operand descriptor, if it exists.
          I += F.getNumOperandRegisters();
        }
        auto *TII = MF.getSubtarget().getInstrInfo();
        auto &MRI = MF.getRegInfo();

        auto MIBuilder = [&](int Opcode) {
          auto Builder =
              llvm::BuildMI(MBB, MI, llvm::MIMetadata(MI), TII->get(Opcode));
          return Builder;
        };

        auto SVAAccessorBuilder = [&](KernelArgumentType KA) {
          auto LaneId =
              stateValueArray::getKernelArgumentLaneIdStoreSlotBeginForWave64(
                  KA);
          LUTHIER_REPORT_FATAL_ON_ERROR(LaneId.takeError());
          auto ArgSize =
              stateValueArray::getKernelArgumentStoreSlotSizeForWave64(KA);
          LUTHIER_REPORT_FATAL_ON_ERROR(ArgSize.takeError());

          llvm::SmallVector<llvm::Register, 2> Out;

          for (unsigned short i = 0; i < *ArgSize; i++) {
            Out.push_back(
                MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass));
            llvm::BuildMI(MBB, MI, llvm::MIMetadata(MI),
                          TII->get(llvm::AMDGPU::V_READLANE_B32), Out.back())
                .addReg(SVAVGPR, 0)
                .addImm(*LaneId + i);
          }
          // Add the requested kernarg
          auto TargetMF = TargetMI.getParent()->getParent();
          if (TargetMF->getFunction().getCallingConv() ==
              llvm::CallingConv::AMDGPU_KERNEL) {
            PreambleDescriptor.Kernels[TargetMF]
                .RequestedKernelArguments.insert(KA);
          } else {
            PreambleDescriptor.DeviceFunctions[TargetMF]
                .RequestedKernelArguments.insert(KA);
          }

          // Emit a reg sequence if the arg size was greater than 1
          if (*ArgSize > 1) {
            // First create a reg sequence MI
            auto Builder = MIBuilder(llvm::AMDGPU::REG_SEQUENCE);

            auto MergedReg = MRI.createVirtualRegister(
                llvm::SIRegisterInfo::getSGPRClassForBitWidth(*ArgSize * 32));
            Builder.addReg(MergedReg, llvm::RegState::Define);

            // Split the src reg into 32-bit regs, and merge them in the
            for (const auto &[SubIdx, Reg] : llvm::enumerate(Out)) {
              Builder.addReg(Reg).addImm(
                  llvm::SIRegisterInfo::getSubRegFromChannel(SubIdx));
            }
            return MergedReg;
          } else {
            return Out[0];
          }
        };

        auto VirtRegBuilder = [&](const llvm::TargetRegisterClass *RC) {
          return MRI.createVirtualRegister(RC);
        };

        auto PhysRegAccessor = [&](llvm::MCRegister Reg) {
          // If the physical reg has been written into in the past instructions
          // then return its current virtual reg from the map
          if (MBBOverwrittenPhysRegs.contains(Reg))
            return MBBOverwrittenPhysRegs[Reg];
          else {
            // otherwise just get the virtual register from the reg
            // virtualization pass
            auto Out =
                RegAccessVirtualizationPass.getMCRegLocationInMBB(Reg, MBB);
            if (!Out) {
              LUTHIER_REPORT_FATAL_ON_ERROR(
                  llvm::make_error<GenericLuthierError>(llvm::formatv(
                      "Failed to find the virtual register associated with "
                      "register {0} in MBB {1}.",
                      llvm::printReg(Reg, MF.getSubtarget().getRegisterInfo()),
                      MBB.getName())));
            }
            return Out;
          }
        };

        // Set of physical reg that are written to by the current intrinsic
        // being processed
        llvm::DenseMap<llvm::MCRegister, llvm::Register> ToBeOverwrittenRegs;

        auto &IRLoweringInfo = ToBeLoweredIntrinsics[*IntrinsicIdx];

        std::optional<IntrinsicProcessor> Processor =
            IntrinsicsProcessors.getProcessorIfRegistered(
                IRLoweringInfo.getIntrinsicName());
        if (!Processor.has_value())
          MF.getFunction().getContext().emitError(
              "Intrinsic processor was not found in the intrinsic processor "
              "map.");
        if (auto Err = Processor->MIRProcessor(
                IRLoweringInfo, ArgVec, MIBuilder, VirtRegBuilder,
                SVAAccessorBuilder, MF, PhysRegAccessor, ToBeOverwrittenRegs)) {
          MF.getFunction().getContext().emitError(
              "Failed to lower the intrinsic; Error message: " +
              llvm::toString(std::move(Err)));
        }
        // Remove the dummy inline assembly placeholder of the processed
        // intrinsic
        MI.eraseFromParent();
        Changed = true;
        // For physical registers that need to be written to, overwrite the
        // uses of their original virtual value with the new one and
        // keep track of it
        if (!ToBeOverwrittenRegs.empty()) {
          for (const auto &[PhysReg, VirtReg] : ToBeOverwrittenRegs) {
            for (auto &Use : MRI.use_operands(PhysRegAccessor(PhysReg))) {
              Use.setReg(VirtReg);
            }
            MBBOverwrittenPhysRegs.insert_or_assign(PhysReg, VirtReg);
          }
        }
      }
    }
  }
  return Changed;
}

void IntrinsicMIRLoweringPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  /// Needs the physical register virtualization pass to provide access to
  /// the holding virtual registers
  AU.addRequiredTransitive<PhysicalRegAccessVirtualizationPass>();
  AU.addRequired<IModuleMAMWrapperPass>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
};

} // namespace luthier