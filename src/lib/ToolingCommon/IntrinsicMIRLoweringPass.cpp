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
#include "AMDGPUGenRegisterInfoHeader.inc"
#include "luthier/Tooling/FunctionAnnotations.h"
#include "luthier/Tooling/InitialEntryPointAnalysis.h"
#include "luthier/Tooling/IntrinsicProcessorsAnalysis.h"
#include "luthier/Tooling/MIRConvenience.h"
#include "luthier/Tooling/StateValueArraySpecs.h"
#include "luthier/Tooling/WrapperAnalysisPasses.h"
#include <AMDGPU.h>
#include <SIInstrInfo.h>
#include <llvm/ADT/BreadthFirstIterator.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineSSAUpdater.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>

namespace luthier {

char IntrinsicMIRLoweringPass::ID = 0;

LUTHIER_INITIALIZE_LEGACY_PASS_BODY(IntrinsicMIRLoweringPass, "mir-lowering",
                                    "Intrinsic MIR lowering pass",
                                    true /* Only looks at CFG */,
                                    false /* Analysis Pass */)

static llvm::SmallVector<std::pair<llvm::InlineAsm::Flag, llvm::Register>>
getInlineAsmArgs(const llvm::MachineInstr &MI) {
  llvm::SmallVector<std::pair<llvm::InlineAsm::Flag, llvm::Register>> Out;
  for (unsigned I = llvm::InlineAsm::MIOp_FirstOperand,
                NumOps = MI.getNumOperands();
       I < NumOps; ++I) {
    const llvm::MachineOperand &MO = MI.getOperand(I);
    if (!MO.isImm())
      continue;
    const llvm::InlineAsm::Flag F(MO.getImm());
    const llvm::Register Reg(MI.getOperand(I + 1).getReg());
    Out.emplace_back(F, Reg);
    // Skip to one before the next operand descriptor, if it exists.
    I += F.getNumOperandRegisters();
  }
  return Out;
}

static llvm::Expected<llvm::SmallVector<llvm::Constant *>>
getPCSectionsIntrinsicExtraInfo(const llvm::MDNode &PCSections) {
  for (const auto [Idx, MDOp] : llvm::enumerate(PCSections.operands())) {
    if (auto *MDS = llvm::dyn_cast<llvm::MDString>(&MDOp);
        MDS && MDS->getString() == IntrinsicExtraInfoHeader) {
      if (Idx < PCSections.getNumOperands() - 1) {
        auto *ExtraInfoArrayMD =
            llvm::dyn_cast<llvm::MDNode>(PCSections.getOperand(Idx).get());
        if (!ExtraInfoArrayMD)
          return LUTHIER_MAKE_GENERIC_ERROR(
              llvm::formatv("The entry list of the {0} in the PC Sections is "
                            "not an LLVM Metadata",
                            IntrinsicExtraInfoHeader));
        llvm::SmallVector<llvm::Constant *> Out;
        Out.reserve(ExtraInfoArrayMD->getNumOperands());
        for (const llvm::MDOperand &ConstantOp : ExtraInfoArrayMD->operands()) {
          if (auto *CV = llvm::mdconst::extract<llvm::Constant>(ConstantOp)) {
            Out.push_back(CV);
          } else {
            return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                "Failed to extract a constant value from MD {0}", ConstantOp));
          }
        }
      } else {
        return llvm::SmallVector<llvm::Constant *>{};
      }
    }
  }
  return llvm::SmallVector<llvm::Constant *>{};
}

bool IntrinsicMIRLoweringPass::lowerIntrinsics(
    llvm::Module &IModule,
    llvm::DenseMap<llvm::Register, ScalarValueArgument>
        &ScalarSVAArgumentVirtualPlaceHolders,
    std::unique_ptr<StateValueArraySpecs> &SVASpecs) {
  bool Changed{false};

  llvm::LLVMContext &Ctx = IModule.getContext();

  llvm::MachineModuleInfo &MMI =
      getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI();

  llvm::ModuleAnalysisManager &IMAM =
      getAnalysis<IModuleMAMWrapperPass>().getMAM();

  const auto &IntrinsicsProcessors =
      IMAM.getResult<IntrinsicsProcessorsAnalysis>(IModule);

  auto &TargetModuleAndMAM =
      IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule);
  llvm::Module &TargetModule = TargetModuleAndMAM.getTargetAppModule();
  llvm::ModuleAnalysisManager &TargetMAM = TargetModuleAndMAM.getTargetAppMAM();
  bool IsInitialEntryPointKernel =
      TargetMAM.getResult<InitialEntryPointAnalysis>(TargetModule)
          .getInitialEntryPoint()
          .isKernel();

  /// Set of all scalar arguments used across the entire module
  llvm::SmallDenseSet<ScalarValueArgument> ScalarArgumentsUsed{};

  /// If the initial entry point is not a kernel, then we are instrumenting
  /// newly discovered code from an already running instrumented kernel. In
  /// this situation, we have already initialized the SVA to save all possible
  /// scalar arguments just to be safe
  if (!IsInitialEntryPointKernel) {
    for (std::underlying_type_t<ScalarValueArgument> I =
             SCALAR_VALUE_ARGUMENT_FIRST;
         I <= SCALAR_VALUE_ARGUMENT_LAST; I++) {
      ScalarArgumentsUsed.insert(static_cast<ScalarValueArgument>(I));
    }
  }

  /// Loop for lowering intrinisics starts here
  for (llvm::Function &F : IModule) {
    bool IsInjectedPayload = F.hasFnAttribute(InjectedPayloadAttribute);
    if (llvm::MachineFunction *MF = MMI.getMachineFunction(F)) {
      const llvm::TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
      const llvm::TargetRegisterInfo *TRI =
          MF->getSubtarget().getRegisterInfo();
      llvm::MachineRegisterInfo &MRI = MF->getRegInfo();

      /// To ensure access to physical registers required by intrinsics, we
      /// create an SSAUpdater per non-overlapping physical registers inside
      /// each machine function the first time it is read/written by an
      /// intrinsic. The SSAUpdaters keep track of each physical register's
      /// virtual value in each basic block and materilizes new vregs as
      /// requested by each intrinsic. To ensure PHI nodes are correctly emitted
      /// by the SSAUpdaters, we traverse the MF in dominance order so that each
      /// block has all its predecessors processed before we emit its PHIs.
      /// When a phys reg is read by an intrinsic for the very first time, a
      /// read from the physical register to a virtual register is emitted in
      /// the entry block of the function to be used by the intrinsic. After we
      /// process a return block of the machine function, all physical registers
      /// that have an SSA updater associated with them will have a read from
      /// their virt reg back to their physical register to indicate the value
      /// should be restored. The last instruction of the return block will also
      /// implicitly use the physical registers to indicate to the register
      /// allocator they are indeed used.
      /// If an intrinsic requests access to a physical regunit larger than
      /// what we keep track of, we emit a reg sequenece instruction to
      /// merge the register units together
      llvm::DenseMap<llvm::MCRegister, std::unique_ptr<llvm::MachineSSAUpdater>>
          PhysRegValueSSAUpdaters;

      llvm::MachineDominatorTree &DOMTree =
          getAnalysis<llvm::MachineDominatorTreeWrapperPass>(F).getDomTree();

      for (auto *MBBNode : llvm::breadth_first(DOMTree.getRootNode())) {
        llvm::MachineBasicBlock *MBB = MBBNode->getBlock();
        for (llvm::MachineInstr &MI :
             llvm::make_early_inc_range(*MBB)) {
          if (MI.isInlineAsm()) {
            llvm::StringRef IntrinsicName =
                MI.getOperand(llvm::InlineAsm::MIOp_AsmString).getSymbolName();
            /// Empty inline assembly instructions are left as is
            if (IntrinsicName.empty())
              continue;

            auto ArgVec = getInlineAsmArgs(MI);

            auto MIBuilder = [&](int Opcode) {
              llvm::MachineInstrBuilder Builder = llvm::BuildMI(
                  *MBB, MI, llvm::MIMetadata(MI), TII->get(Opcode));
              /// Don't propogate the PC sections since it's only used for
              /// intrinsic lowering
              Builder->setPCSections(*MF, nullptr);
              return Builder;
            };

            auto SVAScalarArgumentAccessor = [&](ScalarValueArgument SA) {
              /// Emit an implicit DEF as a place holder for the read-lanes
              /// we're going to emit later on
              llvm::MachineInstrBuilder Builder =
                  llvm::BuildMI(*MBB, MI, llvm::MIMetadata(MI),
                                TII->get(llvm::AMDGPU::IMPLICIT_DEF));
              /// Don't propogate the PC sections since it's only used for
              /// intrinsic lowering
              Builder->setPCSections(*MF, nullptr);
              llvm::Register SAVirtReg =
                  MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass);
              (void)Builder.addReg(SAVirtReg, llvm::RegState::Define);
              ScalarSVAArgumentVirtualPlaceHolders.insert({SAVirtReg, SA});
              ScalarArgumentsUsed.insert(SA);
              return SAVirtReg;
            };

            auto VirtRegBuilder = [&](const llvm::TargetRegisterClass *RC) {
              return MRI.createVirtualRegister(RC);
            };

            auto PhysRegAccessor = [&](llvm::MCRegister Reg) {
              if (!IsInjectedPayload)
                LUTHIER_CTX_EMIT_ON_ERROR(
                    Ctx,
                    LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                        "Function {0} is not an injected payload. Physical "
                        "registers can only be accessed inside "
                        "injected payloads",
                        MF->getName())));
              llvm::SmallVector<llvm::Register> VirtSubRegs;
              /// Get the available virt reg for each sub regunit
              for (llvm::MCRegUnit RegUnit : TRI->regunits(Reg)) {
                for (llvm::MCRegUnitRootIterator Root(RegUnit, TRI);
                     Root.isValid(); ++Root) {
                  auto RootPhysRegIt = PhysRegValueSSAUpdaters.find(*Root);
                  if (RootPhysRegIt == PhysRegValueSSAUpdaters.end()) {
                    RootPhysRegIt =
                        PhysRegValueSSAUpdaters
                            .insert({*Root,
                                     std::make_unique<llvm::MachineSSAUpdater>(
                                         *MF)})
                            .first;
                    llvm::MachineBasicBlock &EntryBlock = MF->front();
                    EntryBlock.addLiveIn(*Root);
                    const llvm::TargetRegisterClass *RootRegClass =
                        TRI->getPhysRegBaseClass(*Root);

                    if (!RootRegClass)
                      LUTHIER_CTX_EMIT_ON_ERROR(
                          Ctx,
                          LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                              "Physical register {0} doesn't have a reg class",
                              llvm::printReg(*Root, TRI))));
                    const llvm::TargetRegisterClass *RootCrossCopyRegClass =
                        TRI->getCrossCopyRegClass(RootRegClass);
                    if (!RootCrossCopyRegClass)
                      LUTHIER_CTX_EMIT_ON_ERROR(
                          Ctx, LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                                   "Physical register {0} doesn't have a copy "
                                   "reg class",
                                   llvm::printReg(*Root, TRI))));
                    llvm::Register NewRootVirtReg =
                        MRI.createVirtualRegister(RootCrossCopyRegClass);
                    (void)llvm::BuildMI(EntryBlock, EntryBlock.begin(),
                                        llvm::MIMetadata(),
                                        TII->get(llvm::AMDGPU::COPY))
                        .addReg(NewRootVirtReg, llvm::RegState::Define)
                        .addReg(*Root);
                    RootPhysRegIt->getSecond()->Initialize(NewRootVirtReg);
                  }
                  VirtSubRegs.push_back(
                      RootPhysRegIt->getSecond()->GetValueInMiddleOfBlock(MBB));
                }
              }

              /// If there are only a single virt reg, return it; Otherwise,
              /// make a reg sequence to merge the registers together before
              /// returning it
              if (VirtSubRegs.size() == 1)
                return VirtSubRegs[0];

              // First create a reg sequence MI
              auto Builder = MIBuilder(llvm::AMDGPU::REG_SEQUENCE);

              auto MergedReg = VirtRegBuilder(TRI->getPhysRegBaseClass(Reg));
              (void)Builder.addReg(MergedReg, llvm::RegState::Define);

              for (auto [Idx, VirtSubReg] : llvm::enumerate(VirtSubRegs)) {
                (void)Builder.addReg(VirtSubReg).addImm(Idx);
              }
              return MergedReg;
            };

            // Set of physical reg that are written to by the current intrinsic
            // being processed
            llvm::DenseMap<llvm::MCRegister, llvm::Register>
                ToBeOverwrittenRegs;

            // List of constants passed to the MIR processor by the IR processor
            llvm::MDNode *PCSections = MI.getPCSections();
            if (!PCSections) {
              LUTHIER_CTX_EMIT_ON_ERROR(
                  Ctx, LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                           "Failed to get the PC sections MD from intrinsic "
                           "place holder instruction {0}",
                           MI)));
            }
            llvm::Expected<llvm::SmallVector<llvm::Constant *>>
                ExtraInfoConstsOrErr =
                    getPCSectionsIntrinsicExtraInfo(*PCSections);

            std::optional<IntrinsicProcessor> Processor =
                IntrinsicsProcessors.getProcessorIfRegistered(IntrinsicName);
            if (!Processor.has_value())
              LUTHIER_CTX_EMIT_ON_ERROR(
                  Ctx, LUTHIER_MAKE_GENERIC_ERROR(
                           llvm::formatv("Intrinsic processor for {0} was not "
                                         "found in the intrinsic processors.",
                                         IntrinsicName)));
            if (auto Err = Processor->MIRProcessor(
                    *MF, ArgVec, *ExtraInfoConstsOrErr, MIBuilder,
                    VirtRegBuilder, SVAScalarArgumentAccessor, PhysRegAccessor,
                    ToBeOverwrittenRegs)) {
              LUTHIER_CTX_EMIT_ON_ERROR(Ctx, Err);
            }
            // Record overwritten physical registers
            for (const auto &[PhysReg, VirtReg] : ToBeOverwrittenRegs) {
              /// Create an SSA Updater if we already don't have one for the
              /// physical register's reg units
              for (llvm::MCRegUnit RegUnit : TRI->regunits(PhysReg)) {
                for (llvm::MCRegUnitRootIterator Root(RegUnit, TRI);
                     Root.isValid(); ++Root) {
                  const llvm::TargetRegisterClass *RootRegClass =
                      TRI->getPhysRegBaseClass(*Root);

                  if (!RootRegClass)
                    LUTHIER_CTX_EMIT_ON_ERROR(
                        Ctx,
                        LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                            "Physical register {0} doesn't have a reg class",
                            llvm::printReg(*Root, TRI))));
                  const llvm::TargetRegisterClass *RootCrossCopyRegClass =
                      TRI->getCrossCopyRegClass(RootRegClass);
                  if (!RootCrossCopyRegClass)
                    LUTHIER_CTX_EMIT_ON_ERROR(
                        Ctx, LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                                 "Physical register {0} doesn't have a copy "
                                 "reg class",
                                 llvm::printReg(*Root, TRI))));

                  auto SubIdx = TRI->getSubRegIndex(PhysReg, *Root);

                  llvm::Register SubVirtReg =
                      MRI.createVirtualRegister(RootCrossCopyRegClass);
                  (void)MIBuilder(llvm::AMDGPU::COPY)
                      .addReg(SubVirtReg, llvm::RegState::Define)
                      .addReg(VirtReg, llvm::RegState::NoFlags, SubIdx);
                  auto RootPhysRegIt = PhysRegValueSSAUpdaters.find(*Root);
                  if (RootPhysRegIt == PhysRegValueSSAUpdaters.end()) {
                    RootPhysRegIt =
                        PhysRegValueSSAUpdaters
                            .insert({*Root,
                                     std::make_unique<llvm::MachineSSAUpdater>(
                                         *MF)})
                            .first;
                    RootPhysRegIt->getSecond()->Initialize(SubVirtReg);
                  }
                  RootPhysRegIt->getSecond()->AddAvailableValue(MBB,
                                                                SubVirtReg);
                }
              }
            }
            // Remove the dummy inline assembly placeholder of the processed
            // intrinsic
            MI.eraseFromParent();
            Changed |= true;
            /// If we have reached a return block, emit a restore to physical
            /// register instruction for all registers that have an SSAUpdater
            if (MBB->isReturnBlock()) {
              for (auto &[PhysReg, SSAUpdater] : PhysRegValueSSAUpdaters) {
                llvm::Register FinalVReg =
                    SSAUpdater->GetValueAtEndOfBlock(MBB);
                (void)llvm::BuildMI(*MBB, MBB->end(), llvm::MIMetadata(),
                                    TII->get(llvm::AMDGPU::COPY))
                    .addReg(PhysReg, llvm::RegState::Define)
                    .addReg(FinalVReg);
                MBB->back().addOperand(
                    llvm::MachineOperand::CreateReg(PhysReg, false, true));
              }
            }
          }
        }
      }
    }
  }
  /// Now that we have a head count of what scalar argument values all the
  /// intrinsics use across the module we can finalize the layout of the
  /// state-value array
  SVASpecs = StateValueArraySpecs::setModuleSVASpec(IModule, MMI.getTarget(),
                                                    ScalarArgumentsUsed);
  return Changed;
}

bool IntrinsicMIRLoweringPass::runOnModule(llvm::Module &IModule) {

  llvm::DenseMap<llvm::Register, ScalarValueArgument>
      ScalarSVAArgumentVirtualPlaceHolders{};
  llvm::DenseMap<llvm::Register, llvm::MCRegister>
      PhysicalRegisterPlaceHolders{};

  std::unique_ptr<StateValueArraySpecs> SVASpecs{nullptr};

  /// Lower the intrinsics, and get a head count of all scalar args used and the
  /// physical registers used throughout the module
  bool Changed =
      lowerIntrinsics(IModule, ScalarSVAArgumentVirtualPlaceHolders, SVASpecs);
  /// Now that we have gotten a head count of all accessed physical registers
  /// and the scalar arguments, we start populating their virtual registers with
  /// their expected values

  return Changed;
}

void IntrinsicMIRLoweringPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  /// Needs the physical register virtualization pass to provide access to
  /// the holding virtual registers
  AU.addRequired<IModuleMAMWrapperPass>();
  AU.addRequired<llvm::MachineModuleInfoWrapperPass>();
  ModulePass::getAnalysisUsage(AU);
};

} // namespace luthier