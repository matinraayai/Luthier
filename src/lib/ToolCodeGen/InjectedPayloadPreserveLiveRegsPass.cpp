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
        auto *PayloadMFRes =
            FAM.getCachedResult<llvm::MachineFunctionAnalysis>(*PayloadDef);
        if (!PayloadMFRes) {
          Live.stepBackward(MI);
          continue;
        }
        llvm::MachineFunction *MF = &PayloadMFRes->getMF();

        LLVM_DEBUG({
          luthier::dbgs() << "  payload " << PayloadDef->getName()
                          << " live-out={";
          for (llvm::MCPhysReg R : Live)
            luthier::dbgs() << " " << llvm::printReg(R, &TargetTRI);
          luthier::dbgs() << " }\n";
        });

        // Compute Preserve = LiveOut(PATCHPOINT) \ (Reads U Writes),
        // restricted to the widest live reg at each unit. \c LivePhysRegs
        // stores subregs_inclusive, so a live 64-bit pair like
        // \c $sgpr12_sgpr13 also carries $sgpr12, $sgpr13, and their 16-bit
        // halves. Preserving all of them is redundant, and preserving the
        // 16-bit halves in particular produces ill-formed SDWA-style COPYs
        // (SGPR_LO16/HI16 have an allocatable cross-copy class but no valid
        // whole-lane COPY expansion). Emitting one save/restore for the
        // widest live reg containing the unit covers the entire live range.
        const InjectedPayloadSideEffects &Acc =
            FAM.getResult<InjectedPayloadSideEffectsAnalysis>(*PayloadDef);
        llvm::SmallVector<llvm::MCPhysReg, 16> Preserve;
        for (llvm::MCPhysReg R : Live) {
          if (Acc.reads_contains(R) || Acc.writes_contains(R))
            continue;
          bool HasSuperLive = false;
          for (llvm::MCPhysReg Super : TargetTRI.superregs(R)) {
            if (Live.contains(Super)) {
              HasSuperLive = true;
              break;
            }
          }
          if (HasSuperLive)
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
      // EXEC / FLAT_SCR / XNACK_MASK are wave/context registers the payload's
      // own PEI (InjectedPayloadPEIPass) and the atomic-optimizer-emitted
      // wave-scan handle by construction — the app doesn't expect them to
      // survive the instrumentation boundary. Everything else — including VCC
      // and SCC — is genuine app state and has to be saved/restored: AMDGPU
      // ISel routinely picks vcc as the carry of v_add_co_u32/v_addc_u32_e64
      // and scc as the flag of every scalar arithmetic op, so an app-uniform
      // atomicAdd payload that lands v_cmp/vcc_lo across a live carry (very
      // common for 64-bit pointer arithmetic) will corrupt the following
      // address computation and page-fault the wave. SIRegisterInfo lowers
      // COPY $vcc via s_mov_b32/s_mov_b64 and COPY $scc via the
      // S_CSELECT_B32 / S_CMP_LG_U32 pair (see SIRegisterInfo::copyPhysReg).
      bool IsUnpreservableArchReg = false;
      for (llvm::MCPhysReg ArchReg : {llvm::AMDGPU::EXEC,
                                       llvm::AMDGPU::FLAT_SCR,
                                       llvm::AMDGPU::XNACK_MASK}) {
        if (TRI->regsOverlap(PhysReg, ArchReg)) {
          IsUnpreservableArchReg = true;
          break;
        }
      }
      if (IsUnpreservableArchReg) {
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
      // A non-allocatable cross-copy class means the target has no way to
      // materialize this reg into a vreg. VCC / SCC do NOT hit this branch on
      // AMDGPU: SIRegisterInfo::getCrossCopyRegClass returns SReg_32 for SCC
      // and passes VCC through as SReg_64, both of which are allocatable.
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
