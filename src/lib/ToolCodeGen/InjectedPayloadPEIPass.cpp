//===-- InjectedPayloadPEIPass.cpp ----------------------------------------===//
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
///
/// \file
/// Implements Luthier's Injected Payload Prologue and Epilogue insertion pass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InjectedPayloadPEIPass.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/PrePostAmbleEmitter.h"
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include "luthier/ToolCodeGen/StateValueArrayStorage.h"
#include "luthier/ToolCodeGen/WrapperAnalysisPasses.h"

#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-injected-payload-pei"

namespace luthier {

namespace {

/// Returns the set of phys-regs that, when used by an injected payload, must
/// be saved into SVA lanes on prologue and restored on epilogue: the per-SA
/// frame regs the kernel prolog set up (FLAT_SCR_LO/HI for absolute-FS
/// targets, SGPR32 for the instrumentation SP, etc.). For now, the lane
/// mapping for each phys-reg is the one StateValueArraySpecs records via
/// its fixed-position constants (StackPointerRegSpillLane,
/// FramePointerRegSpillLane, etc.). When the user finalizes the
/// per-phys-reg lane mapping, the dictionary below should move into
/// StateValueArraySpecs.
llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>>
getFrameSpillSlotsForTarget(const llvm::GCNSubtarget &ST,
                            const StateValueArraySpecs &Specs) {
  llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>> Out;
  Out.push_back({llvm::AMDGPU::SGPR32, Specs.getStackPointerRegSpillLane()});
  if (!ST.hasArchitectedFlatScratch()) {
    Out.push_back(
        {llvm::AMDGPU::FLAT_SCR_LO, Specs.getFramePointerRegSpillLane()});
  }
  return Out;
}

/// Returns the lanes from which the instrumentation reads the frame regs
/// it needs to use during the payload. Symmetric counterpart to
/// getFrameSpillSlotsForTarget — kernel prolog populated these lanes; the
/// payload prologue copies them out into SGPR32 / FLAT_SCR.
llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>>
getFrameLoadSlotsForTarget(const llvm::GCNSubtarget &ST,
                           const StateValueArraySpecs &Specs) {
  llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>> Out;
  Out.push_back({llvm::AMDGPU::SGPR32, Specs.getStackPointerStoreLane()});
  if (!ST.hasArchitectedFlatScratch()) {
    if (auto FrameLane = Specs.getFrameRsrcOrScratchStoreLaneIfExists())
      Out.push_back({llvm::AMDGPU::FLAT_SCR_LO, *FrameLane});
  }
  return Out;
}

} // namespace

llvm::PreservedAnalyses
InjectedPayloadPEIPass::run(llvm::MachineFunction &MF,
                            llvm::MachineFunctionAnalysisManager &MFAM) {
  // Skip anything that isn't a Luthier injected payload. Hooks (callees
  // of payloads) keep their normal LLVM-emitted frame and don't need
  // custom SVA setup.
  llvm::Function &F = MF.getFunction();
  if (!F.hasFnAttribute(InjectedPayloadAttribute)) {
    LLVM_DEBUG(luthier::dbgs()
               << F.getName() << " is not an injected payload; skipping.\n");
    return llvm::PreservedAnalyses::all();
  }

  // Defensive: payloads MUST be marked Naked. If somebody bypassed
  // InjectedPayloadCreationPass::assignToInject and forgot the attribute,
  // stock PEI already ran and emitted a frame we'd be doubling up on.
  assert(
      F.hasFnAttribute(llvm::Attribute::Naked) &&
      "Injected payload must carry Attribute::Naked so stock PEI is a no-op");

  LLVM_DEBUG(luthier::dbgs()
             << "Running InjectedPayloadPEIPass on " << F.getName() << "\n");

  llvm::LLVMContext &Ctx = F.getContext();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto *TII = ST.getInstrInfo();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  llvm::MachineFrameInfo &MFI = MF.getFrameInfo();

  // The MF being processed lives in the instrumentation module; the MAM
  // outer proxy exposes cached module-level analyses (IPIP,
  // TargetAppModuleAndMAM) populated by upstream instrumentation-module
  // passes.
  llvm::Module &IModule = *F.getParent();
  const auto &MAMProxy =
      MFAM.getResult<llvm::ModuleAnalysisManagerMachineFunctionProxy>(MF);

  const auto *IPIP =
      MAMProxy.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule);
  if (!IPIP) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "InjectedPayloadAndInstPointAnalysis is required but not cached.")));
    return llvm::PreservedAnalyses::all();
  }
  if (!IPIP->contains(F)) {
    LLVM_DEBUG(luthier::dbgs()
               << F.getName()
               << " has no recorded insertion point; skipping.\n");
    return llvm::PreservedAnalyses::all();
  }
  const llvm::MachineInstr *TargetMI = IPIP->at(F);

  auto *TargetMAMRes =
      MAMProxy.getCachedResult<TargetAppModuleAndMAMAnalysis>(IModule);
  if (!TargetMAMRes) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "TargetAppModuleAndMAMAnalysis is required but not cached.")));
    return llvm::PreservedAnalyses::all();
  }
  auto &TargetModule = TargetMAMRes->getTargetAppModule();

  // SVStorageAndLoadLocationsAnalysis is an Prototype-level
  // analysis, and FunctionPreambleDescriptorAnalysis is likewise IP-level.
  // The outer IPAM proxy on this MF exposes cached IPAM lookups.
  //
  // TODO(NPM): the proxy indexes cached results by Prototype*,
  // but a MachineFunction pass has no natural handle to its containing IP.
  // Wiring the SVStorage/FPD lookups from an MF pass requires the driver
  // to expose the enclosing IP through a small MAM accessor (mirroring
  // TargetAppModuleAndMAMAnalysis). Until that landing, callers of this
  // pass are broken.
  const auto &IPAMProxy =
      MFAM.getResult<PrototypeAnalysisManagerMachineFunctionProxy>(
          MF);
  (void)IPAMProxy;

  const SVStorageAndLoadLocations *StateValueLocations = nullptr;
  const InstPointSVALoadPlan *LoadPlan =
      StateValueLocations
          ? StateValueLocations->getStateValueArrayLoadPlanForInstPoint(
                *TargetMI)
          : nullptr;
  if (!LoadPlan) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "No SVA load plan recorded for instrumentation point in {0}",
        F.getName()))));
    return llvm::PreservedAnalyses::all();
  }
  auto &StateValueStorage = LoadPlan->StateValueStorageLocation;

  // Pull the finalized SVA specs (set in IntrinsicMIRLoweringPass).
  auto SpecsPtr = StateValueArraySpecs::getSVASpecs(IModule, MF.getTarget());
  if (!SpecsPtr) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to read StateValueArraySpecs from IModule metadata")));
    return llvm::PreservedAnalyses::all();
  }
  const StateValueArraySpecs &Specs = *SpecsPtr;

  // The SVA physreg is the load-plan's canonical destination VGPR. The
  // pin pass (SVAPhysVGPRPinPass) guarantees that whichever VGPR
  // SVStorageAndLoadLocations picked here is what the WWM regalloc
  // assigns to our LaneVGPR in this payload MF — so this read is the
  // single source of truth for "where the SVA lives at this IP."
  // No V_READLANE_B32-walk needed.
  llvm::MCRegister SVAVGPR = LoadPlan->StateValueArrayLoadVGPR;

  // ---- Decide whether this payload actually uses the SVA -----------------
  //
  // Sources of SVA use:
  //   1. The SVA VGPR has any explicit non-implicit read/write — meaning the
  //      payload (or its inlined children) referenced an SA via the lowered
  //      V_READLANE_B32.
  //   2. The payload uses a frame register (SGPR32 for SP, or FLAT_SCR_*
  //      for absolute-FS targets) — those values come from / go to SVA lanes.
  //   3. The MF has a non-empty frame (RA chose to spill) — implies stack
  //      usage, which needs the instrumentation's SP loaded.
  //   4. The MF makes calls to other functions — also needs the frame setup.
  auto FrameSpillSlots = getFrameSpillSlotsForTarget(ST, Specs);
  auto FrameLoadSlots = getFrameLoadSlotsForTarget(ST, Specs);

  bool UsesSVA = false;
  if (SVAVGPR && MRI.isPhysRegUsed(SVAVGPR)) {
    LLVM_DEBUG(luthier::dbgs()
               << "  SVA VGPR " << llvm::printReg(SVAVGPR, ST.getRegisterInfo())
               << " is used\n");
    UsesSVA = true;
  }
  if (!UsesSVA) {
    for (const auto &[PhysReg, _] : FrameSpillSlots) {
      if (MRI.isPhysRegUsed(PhysReg)) {
        LLVM_DEBUG(luthier::dbgs()
                   << "  frame reg "
                   << llvm::printReg(PhysReg, ST.getRegisterInfo())
                   << " is used\n");
        UsesSVA = true;
        break;
      }
    }
  }
  if (!UsesSVA && MFI.hasStackObjects()) {
    LLVM_DEBUG(luthier::dbgs() << "  MFI has stack objects\n");
    UsesSVA = true;
  }
  if (!UsesSVA && MFI.hasCalls()) {
    LLVM_DEBUG(luthier::dbgs() << "  MFI has calls\n");
    UsesSVA = true;
  }
  if (!UsesSVA) {
    LLVM_DEBUG(luthier::dbgs()
               << F.getName() << " doesn't use the SVA; skipping PEI.\n");
    return llvm::PreservedAnalyses::all();
  }

  // If the payload uses the SVA but we never discovered an SVA VGPR from
  // V_READLANE_B32 (meaning no SAs were requested), we still need a VGPR
  // to materialize the SVA into for frame-reg load/spill. For now, error
  // out — the design path for "frame-only" usage (calls/spills but no SAs)
  // needs additional plumbing the user has not yet specified.
  if (!SVAVGPR) {
    Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "{0} uses stack/frame regs but requests no SAs; the SVA VGPR "
        "is therefore unallocated by IntrinsicMIRLoweringPass. This "
        "case requires the payload to consume at least one SA so the "
        "SVA VGPR exists, OR explicit no-SA SVA allocation support "
        "(not yet implemented).",
        F.getName()))));
    return llvm::PreservedAnalyses::all();
  }

  // ---- Update the FunctionPreambleDescriptor on the target side ----------
  // PrePostAmbleEmitter consults this to decide which kernel/device-fn
  // prologues need scratch+stack setup. Mark the target accordingly.
  //
  // TODO(NPM): FunctionPreambleDescriptorAnalysis is now an IP-level
  // analysis. Reaching its result from this MF pass has the same missing-IP
  // problem as the SVStorage lookup above; the descriptor update is left
  // broken until the accessor lands.
  FunctionPreambleDescriptor *PKInfo = nullptr;
  const llvm::MachineFunction *TargetMF = TargetMI->getMF();
  bool RequiresAccessToStack = false;
  if (StateValueStorage.getStateValueStorageReg() == 0) {
    // SVA is spilled — payload necessarily needs FS to load it.
    RequiresAccessToStack = true;
  }
  if (MFI.hasStackObjects() || MFI.hasCalls())
    RequiresAccessToStack = true;
  if (RequiresAccessToStack && PKInfo) {
    if (TargetMF->getFunction().getCallingConv() ==
        llvm::CallingConv::AMDGPU_KERNEL) {
      PKInfo->Kernels[TargetMF].RequiresScratchAndStackSetup = true;
    } else {
      PKInfo->DeviceFunctions[TargetMF].RequiresScratchAndStackSetup = true;
    }
  }
  (void)TargetModule;

  // ---- Emit the prologue ------------------------------------------------
  llvm::MachineBasicBlock &EntryMBB = MF.front();
  auto EntryInsertPt = EntryMBB.SkipPHIsAndLabels(EntryMBB.begin());

  // If the SVS isn't already a free VGPR, load the SVA into the SVA VGPR.
  // emitCodeToLoadSVA is a no-op for VGPRStateValueArrayStorage; for the
  // other schemes (spilled / AGPR-based) it materializes the SVA into the
  // destination VGPR. See project_sva_storage_audit memory note for which
  // schemes are known-correct as of this commit.
  if (StateValueStorage.requiresLoadAndStoreBeforeUse()) {
    StateValueStorage.emitCodeToLoadSVA(*EntryInsertPt, SVAVGPR);
  }

  // Spill app frame regs into the SVA lanes the kernel prolog reserved.
  for (const auto &[PhysReg, SpillLane] : FrameSpillSlots) {
    if (!MRI.isPhysRegUsed(PhysReg))
      continue;
    llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::V_WRITELANE_B32), SVAVGPR)
        .addReg(PhysReg, llvm::RegState::Kill)
        .addImm(SpillLane)
        .addReg(SVAVGPR);
  }

  // If the payload needs stack, read the instrumentation's frame regs out
  // of the SVA lanes the kernel prolog populated.
  if (RequiresAccessToStack) {
    for (const auto &[PhysReg, LoadLane] : FrameLoadSlots) {
      llvm::BuildMI(EntryMBB, EntryInsertPt, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_READLANE_B32), PhysReg)
          .addReg(SVAVGPR)
          .addImm(LoadLane);
    }
  }

  // ---- Emit the symmetric epilogue at every return block ----------------
  for (llvm::MachineBasicBlock &MBB : MF) {
    if (!MBB.isReturnBlock())
      continue;
    auto FirstTerm = MBB.getFirstTerminator();
    // Reverse order: frame-reg restore, then SVS store.
    for (const auto &[PhysReg, SpillLane] : FrameSpillSlots) {
      if (!MRI.isPhysRegUsed(PhysReg))
        continue;
      llvm::BuildMI(MBB, FirstTerm, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_READLANE_B32), PhysReg)
          .addReg(SVAVGPR)
          .addImm(SpillLane);
      // Tag the terminator so the live-out is visible to anything that
      // walks operand-level liveness post-PEI.
      if (FirstTerm != MBB.end()) {
        FirstTerm->addOperand(llvm::MachineOperand::CreateReg(
            PhysReg, /*isDef=*/false, /*isImp=*/true));
      }
    }
    if (StateValueStorage.requiresLoadAndStoreBeforeUse()) {
      // Emit at FirstTerm of THIS return block, not at the entry point —
      // fixing a long-standing prolog/epilog asymmetry bug in the prior
      // implementation.
      if (FirstTerm != MBB.end()) {
        StateValueStorage.emitCodeToStoreSVA(*FirstTerm, SVAVGPR);
      }
    }
  }

  LLVM_DEBUG({
    luthier::dbgs() << "After InjectedPayloadPEIPass on " << F.getName()
                    << ":\n";
    MF.print(luthier::dbgs());
  });

  return llvm::PreservedAnalyses::none();
}

} // namespace luthier
