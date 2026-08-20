//===-- InjectedPayloadPreserveLiveRegsPass.cpp ---------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file InjectedPayloadPreserveLiveRegsPass.cpp
/// Implements \c InjectedPayloadPreserveLiveRegsPass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InjectedPayloadPreserveLiveRegsPass.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include <AMDGPU.h>
#include <SIInstrInfo.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetOpcodes.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Debug.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-payload-preserve-live-regs"

namespace luthier {

llvm::PreservedAnalyses InjectedPayloadPreserveLiveRegsPass::run(
    Prototype &IP, PrototypeAnalysisManager &IPAM) {
  LLVM_DEBUG(luthier::dbgs()
             << "=== Luthier Injected Payload Preserve Live Regs Pass ===\n");

  llvm::Module &IModule = IP.getInstrumentationModule();
  llvm::Module &TargetModule = IP.getTargetModule();

  // This pass only reads the instrumentation module, so it goes through that
  // module's own managers.
  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<IModuleAnalysisManagerPrototypeProxy>(IP).getManager();

  llvm::MachineModuleInfo &MMI =
      MAM.getResult<llvm::MachineModuleAnalysis>(IModule).getMMI();
  llvm::FunctionAnalysisManager &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(IModule)
          .getManager();

  // Target-side FAM is where the target module's MachineFunctionAnalysis
  // results live (populated by CodeDiscoveryPass); we walk it to iterate
  // every PATCHPOINT MI in the target module.
  llvm::FunctionAnalysisManager &TargetFAM =
      IPAM.getResult<TargetFunctionAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  const IPPredicatedLiveness &Liveness =
      IPAM.getResult<IPPredicatedLivenessAnalysis>(IP);
  IPPredicatedCFG &CFG =
      IPAM.getResult<IPPredCFGAnalysis>(IP).getVecCFG();

  bool Changed = false;

  // Iterate PATCHPOINT markers in the target module directly (no
  // precomputed per-payload map). For each, compute the live-out at the
  // PATCHPOINT — that is the set the payload must preserve — by unioning
  // successor PMBBs' live-ins and stepping backward through the parent
  // MBB up to (but not including) the PATCHPOINT itself.
  for (llvm::Function &TF : TargetModule) {
    llvm::MachineFunctionAnalysis::Result *MFRes =
        TargetFAM.getCachedResult<llvm::MachineFunctionAnalysis>(TF);
    if (!MFRes)
      continue;
    llvm::MachineFunction &TMF = MFRes->getMF();
    const llvm::TargetRegisterInfo &TargetTRI =
        *TMF.getSubtarget().getRegisterInfo();

    for (llvm::MachineBasicBlock &MBB : TMF) {
      if (!CFG.contains(MBB))
        continue;
      const PredicatedMachineBasicBlock &PMBB = CFG.at(MBB);

      // MBB live-out = union of successor PMBBs' live-ins.
      llvm::LivePhysRegs Live(TargetTRI);
      for (const PredicatedMachineBasicBlock &Succ : PMBB.successors())
        if (const llvm::LivePhysRegs *SuccLive =
                Liveness.getPMBBLiveIns(Succ))
          for (llvm::MCPhysReg R : *SuccLive)
            Live.addReg(R);

      for (auto MIt = MBB.rbegin(), MEnd = MBB.rend(); MIt != MEnd; ++MIt) {
        llvm::MachineInstr &MI = *MIt;
        if (MI.getOpcode() != llvm::TargetOpcode::PATCHPOINT) {
          Live.stepBackward(MI);
          continue;
        }

        // PatchpointOpers layout: ID, NBytes, Target, NArgs, CC. Operand
        // 2 is the payload extern handle (a \c GlobalAddress \c Function *
        // in the target module). Its name matches the IModule payload
        // definition's name.
        const llvm::MachineOperand &TargetOp = MI.getOperand(2);
        assert(TargetOp.isGlobal() &&
               "PATCHPOINT target operand must be a GlobalAddress");
        const auto *ExternHandle =
            llvm::cast<llvm::Function>(TargetOp.getGlobal());
        llvm::Function *PayloadDef = IModule.getFunction(ExternHandle->getName());
        if (!PayloadDef || !PayloadDef->hasFnAttribute(InjectedPayloadAttribute)) {
          Live.stepBackward(MI);
          continue;
        }
        llvm::MachineFunction *MF = MMI.getMachineFunction(*PayloadDef);
        if (!MF) {
          Live.stepBackward(MI);
          continue;
        }

        LLVM_DEBUG({
          luthier::dbgs() << "  payload " << PayloadDef->getName()
                          << " live-out={";
          for (llvm::MCPhysReg R : Live)
            luthier::dbgs() << " " << llvm::printReg(R, &TargetTRI);
          luthier::dbgs() << " }\n";
        });

        // Compute Preserve = LiveOut(PATCHPOINT) \ (Reads U Writes).
        const InjectedPayloadSideEffects &Acc =
            FAM.getResult<InjectedPayloadSideEffectsAnalysis>(*PayloadDef);
        llvm::SmallVector<llvm::MCPhysReg, 16> Preserve;
        for (llvm::MCPhysReg R : Live) {
          if (Acc.reads_contains(R) || Acc.writes_contains(R))
            continue;
          Preserve.push_back(R);
        }

        // Advance the walk past the PATCHPOINT itself so that a later
        // PATCHPOINT MI upstream in the same MBB sees the correct
        // (post-payload-effects) live state.
        Live.stepBackward(MI);

        if (Preserve.empty())
          continue;

        const llvm::TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
        const llvm::TargetRegisterInfo *TRI =
            MF->getSubtarget().getRegisterInfo();
        llvm::MachineRegisterInfo &MRI = MF->getRegInfo();

    llvm::MachineBasicBlock &EntryMBB = MF->front();
    auto EntryInsertPt = EntryMBB.SkipPHIsAndLabels(EntryMBB.begin());

    // For each preserved phys-reg, emit an entry-block COPY into a fresh
    // virtual register, and at every return block, emit a COPY back to
    // the phys-reg before the terminator + add an implicit-use of the
    // phys-reg on the terminator so the register allocator treats it as
    // a live-out of the function.
    for (llvm::MCPhysReg PhysReg : Preserve) {
      const llvm::TargetRegisterClass *RC = TRI->getPhysRegBaseClass(PhysReg);
      if (!RC) {
        LLVM_DEBUG(luthier::dbgs()
                   << "  skipping " << llvm::printReg(PhysReg, TRI)
                   << ": no reg class\n");
        continue;
      }
      // TODO: Fix this
      // Skip architectural registers and their sub-halves (VCC/EXEC/FLAT_SCR/
      // XNACK_MASK). The full 64-bit registers are caught by the
      // non-allocatable cross-copy check below, but their 32-bit halves
      // ($vcc_hi, $exec_lo, ...) report an allocatable sreg_32 cross-copy class
      // and would otherwise slip through and get an ill-formed save/restore:
      // the payload uses these as scratch (e.g. the bank-conflict hook clobbers
      // $vcc for its compares), so the half's regunit live range is not jointly
      // dominated and regalloc aborts with "Use not jointly dominated by defs".
      // Per this pass's design these are application-level architectural state
      // the lifted code doesn't expect to survive the instrumentation boundary.
      bool IsArchReg = false;
      for (llvm::MCPhysReg ArchReg :
           {llvm::AMDGPU::VCC, llvm::AMDGPU::EXEC, llvm::AMDGPU::FLAT_SCR,
            llvm::AMDGPU::XNACK_MASK}) {
        if (TRI->regsOverlap(PhysReg, ArchReg)) {
          IsArchReg = true;
          break;
        }
      }
      if (IsArchReg) {
        LLVM_DEBUG(luthier::dbgs()
                   << "  skipping " << llvm::printReg(PhysReg, TRI)
                   << ": architectural register (not preserved)\n");
        continue;
      }
      const llvm::TargetRegisterClass *CrossCopyRC =
          TRI->getCrossCopyRegClass(RC);
      if (!CrossCopyRC) {
        LLVM_DEBUG(luthier::dbgs()
                   << "  skipping " << llvm::printReg(PhysReg, TRI)
                   << ": no cross-copy class\n");
        continue;
      }
      // Architectural regs like $exec, $vcc, $scc, $flat_scr report a
      // cross-copy class that's not allocatable. createVirtualRegister
      // asserts on those. The inlined payload's effects on them are
      // either handled separately (frame regs by InjectedPayloadPEIPass)
      // or are application-level architectural state that the lifted
      // code doesn't expect to survive an instrumentation boundary.
      if (!CrossCopyRC->isAllocatable()) {
        LLVM_DEBUG(luthier::dbgs()
                   << "  skipping " << llvm::printReg(PhysReg, TRI)
                   << ": cross-copy class not allocatable\n");
        continue;
      }
      llvm::Register SaveVReg = MRI.createVirtualRegister(CrossCopyRC);
      // Entry: %savevreg = COPY $physreg ; mark $physreg live-in.
      llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::COPY))
          .addReg(SaveVReg, llvm::RegState::Define)
          .addReg(PhysReg);
      if (!EntryMBB.isLiveIn(PhysReg))
        EntryMBB.addLiveIn(PhysReg);

      // Return blocks: emit restore COPY before the first terminator and
      // tag the terminator with an implicit-use of the physreg.
      for (llvm::MachineBasicBlock &MBB : *MF) {
        if (!MBB.isReturnBlock())
          continue;
        auto FirstTerm = MBB.getFirstTerminator();
        llvm::BuildMI(MBB, FirstTerm, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::COPY))
            .addReg(PhysReg, llvm::RegState::Define)
            .addReg(SaveVReg);
        // Add implicit use of $physreg on the terminator so the live-out
        // is visible to RA.
        if (FirstTerm != MBB.end()) {
          FirstTerm->addOperand(llvm::MachineOperand::CreateReg(
              PhysReg, /*isDef=*/false, /*isImp=*/true));
        }
      }
      Changed = true;
    }
    EntryMBB.sortUniqueLiveIns();
      } // end for each PATCHPOINT MI in reverse
    } // end for each MBB in TMF
  } // end for each target function

  if (!Changed)
    return llvm::PreservedAnalyses::all();

  // Preserve the outer MAM proxy so the Prototype adaptor doesn't
  // wipe every cached module-level analysis for both modules on the way out —
  // downstream passes still need the cached MachineFunctionAnalysis results
  // for the instrumentation module we just mutated.
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  // Preservation copies are emitted into existing injected-payload MFs in the
  // instrumentation module. No MachineFunction is created or destroyed and the
  // target module is untouched, so the inner managers remain accurate; only
  // Prototype-level analyses over payload liveness are dropped.
  PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  return PA;
}

} // namespace luthier
