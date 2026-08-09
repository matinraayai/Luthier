//===-- StateValueLocationIntervalsPass.cpp -------------------------------===//
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
/// This file implements the State Value Location Intervals Pass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/SVStorageAndLoadLocations.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/LuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessIModulePass.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/StateValueArrayStorage.h"
#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Support/CommandLine.h>

#include <utility>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lr-state-value-storage-and-load"

namespace luthier {

/// Allow the SVA-storage scavenger to pick physical registers whose
/// hardware index exceeds the target function's declared
/// `amdgpu-num-{vgpr,sgpr}` attribute. Default off so that
/// instrumentation lives inside the original allocation footprint; the
/// user must opt in (--luthier-exceed-num-regs) for tightly pressured
/// targets where instrumentation can't fit otherwise. The option is
/// declared here because the scavenger is the only consumer; the
/// driver re-exports its current value via a thin getter for diagnostics.
static llvm::cl::opt<bool> ExceedNumRegs(
    "luthier-exceed-num-regs", llvm::cl::init(false),
    llvm::cl::desc("Allow SVA storage scavenging to pick V/SGPRs above the "
                   "target function's amdgpu-num-{vgpr,sgpr} attribute."));

namespace {

/// Hardware-index of an AMDGPU 32-bit V/A/SGPR within its bank, or
/// \c std::nullopt when \p Reg is not in one of the three 32-bit
/// allocatable classes the SVA scavenger considers. AGPRs share the
/// VGPR budget on the gfx9+ accumulator-style configurations (the
/// AMDGPU `amdgpu-num-vgpr` attribute names them together), so for
/// cap-checking purposes AGPR indices are also bounded by the
/// VGPR cap.
std::optional<std::pair<unsigned, bool /*isSGPR*/>>
hwIndexAndKind(llvm::MCRegister Reg) {
  if (Reg >= llvm::AMDGPU::VGPR0 && Reg <= llvm::AMDGPU::VGPR255)
    return std::pair<unsigned, bool>{Reg - llvm::AMDGPU::VGPR0, false};
  if (Reg >= llvm::AMDGPU::AGPR0 && Reg <= llvm::AMDGPU::AGPR255)
    return std::pair<unsigned, bool>{Reg - llvm::AMDGPU::AGPR0, false};
  if (Reg >= llvm::AMDGPU::SGPR0 && Reg <= llvm::AMDGPU::SGPR105)
    return std::pair<unsigned, bool>{Reg - llvm::AMDGPU::SGPR0, true};
  return std::nullopt;
}

/// True if picking \p Reg as SVA storage in \p MF would stay within the
/// MF's declared `amdgpu-num-{vgpr,sgpr}` cap (or if \c --luthier-exceed-num-regs
/// is enabled, or if the MF carries no such attribute). VGPR/AGPR indices
/// are bounded by `amdgpu-num-vgpr`; SGPR indices by `amdgpu-num-sgpr`.
bool isWithinDeclaredCap(const llvm::MachineFunction &MF, llvm::MCRegister Reg) {
  if (ExceedNumRegs)
    return true;
  auto IdxAndKind = hwIndexAndKind(Reg);
  if (!IdxAndKind)
    return true; // not in a budgeted class — caller's responsibility.
  auto [Index, IsSGPR] = *IdxAndKind;
  llvm::StringRef AttrName =
      IsSGPR ? llvm::StringRef("amdgpu-num-sgpr")
             : llvm::StringRef("amdgpu-num-vgpr");
  if (!MF.getFunction().hasFnAttribute(AttrName))
    return true; // no declared cap → no constraint.
  unsigned Cap = MF.getFunction().getFnAttributeAsParsedInteger(AttrName);
  return Index < Cap;
}

/// Cross-MF cap check. Equivalent to \c isWithinDeclaredCap applied to
/// every related function: the scavenger conservatively refuses \p Reg
/// when any MF's declared cap excludes it.
bool isWithinDeclaredCap(llvm::ArrayRef<llvm::MachineFunction *> Functions,
                         llvm::MCRegister Reg) {
  if (ExceedNumRegs)
    return true;
  for (const llvm::MachineFunction *MF : Functions)
    if (!isWithinDeclaredCap(*MF, Reg))
      return false;
  return true;
}



/// Build a non-owning set of phys-regs that must NOT be clobbered at the
/// given AppMI: the union of \c Active and \c Inactive lane live sets
/// across every injected payload attached to \p AppMI. Each payload's
/// per-lane sets already incorporate that payload's declared Reads/Writes
/// via \c stepBackwardOverPayload inside
/// \c IModuleIPPredicatedLivenessAnalysis.
llvm::DenseSet<llvm::MCPhysReg>
liveAtAppMI(const llvm::MachineInstr &AppMI,
            const InjectedPayloadAndInstPoint &IPIP,
            const llvm::DenseMap<const llvm::Function *, PayloadLiveSets>
                &PayloadLiveSetsByFn) {
  llvm::DenseSet<llvm::MCPhysReg> Out;
  if (!IPIP.contains(AppMI))
    return Out;
  for (const llvm::Function *Payload : IPIP.at(AppMI)) {
    auto It = PayloadLiveSetsByFn.find(Payload);
    if (It == PayloadLiveSetsByFn.end())
      continue;
    for (llvm::MCPhysReg R : It->second.Active)
      Out.insert(R);
    for (llvm::MCPhysReg R : It->second.Inactive)
      Out.insert(R);
  }
  return Out;
}

/// Build the conservative union across every payload's live set in
/// \p PayloadLiveSetsByFn. Used by the function-start scavenger paths
/// that don't have an AppMI in hand.
llvm::DenseSet<llvm::MCPhysReg>
liveAcrossAllPayloads(const llvm::DenseMap<const llvm::Function *,
                                           PayloadLiveSets>
                          &PayloadLiveSetsByFn) {
  llvm::DenseSet<llvm::MCPhysReg> Out;
  for (const auto &[F, LS] : PayloadLiveSetsByFn) {
    for (llvm::MCPhysReg R : LS.Active)
      Out.insert(R);
    for (llvm::MCPhysReg R : LS.Inactive)
      Out.insert(R);
  }
  return Out;
}

} // namespace

/// Scavenges \p NumRegs registers of class \p RC that are:
///   - allocatable per \p MRI,
///   - not already used in \p MRI (RA didn't touch them),
///   - not in \p DoNotClobber.
///
/// \p DoNotClobber is the union of every "must-preserve" set the caller
/// cares about — typically the per-payload \c Active ∪ \c Inactive live
/// sets at the relevant IPs, plus (for function-start scavenging) the
/// entry-block live-ins of the MF being scavenged.
static void
scavengeFreeRegister(const llvm::MachineFunction &MF,
                     const llvm::TargetRegisterClass &RC,
                     const llvm::DenseSet<llvm::MCPhysReg> &DoNotClobber,
                     int NumRegs,
                     llvm::SmallVectorImpl<llvm::MCRegister> &ScavengedRegs) {
  const auto &MRI = MF.getRegInfo();
  int NumRegsFound = 0;
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
        !DoNotClobber.contains(Reg) && isWithinDeclaredCap(MF, Reg)) {
      ScavengedRegs.push_back(Reg);
      NumRegsFound++;
      if (NumRegsFound == NumRegs)
        return;
    }
  }
}

/// Single-register variant of the scavenger above. Returns the first
/// candidate register satisfying the same predicates, or \c MCRegister{}
/// if none is found.
static llvm::MCRegister
scavengeFreeRegister(const llvm::MachineFunction &MF,
                     const llvm::TargetRegisterClass &RC,
                     const llvm::DenseSet<llvm::MCPhysReg> &DoNotClobber) {
  const auto &MRI = MF.getRegInfo();
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
        !DoNotClobber.contains(Reg) && isWithinDeclaredCap(MF, Reg)) {
      return Reg;
    }
  }
  return {};
}

/// Cross-MF variant. Scavenges \p NumRegs registers of class \p RC unused
/// across every MachineFunction in \p Functions and not in \p DoNotClobber.
static void
scavengeFreeRegister(llvm::ArrayRef<llvm::MachineFunction *> Functions,
                     const llvm::TargetRegisterClass *RC,
                     const llvm::DenseSet<llvm::MCPhysReg> &DoNotClobber,
                     unsigned int NumRegs,
                     llvm::SmallVectorImpl<llvm::MCRegister> &Regs) {
  unsigned int NumRegFound = 0;
  for (llvm::MCRegister Reg : *RC) {
    if (!isWithinDeclaredCap(Functions, Reg))
      continue;
    bool IsUnused = llvm::all_of(Functions, [&](llvm::MachineFunction *MF) {
      auto &MRI = MF->getRegInfo();
      return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
             !DoNotClobber.contains(Reg);
    });
    if (IsUnused) {
      Regs.push_back(Reg);
      NumRegFound++;
      if (NumRegFound == NumRegs)
        return;
    }
  }
}

llvm::MCRegister
scavengeFreeRegister(llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions,
                     const llvm::TargetRegisterClass *RC,
                     const llvm::DenseSet<llvm::MCPhysReg> &DoNotClobber) {
  for (llvm::MCRegister Reg : *RC) {
    if (!isWithinDeclaredCap(RelatedFunctions, Reg))
      continue;
    bool IsUnused =
        llvm::all_of(RelatedFunctions, [&](llvm::MachineFunction *MF) {
          auto &MRI = MF->getRegInfo();
          return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
                 !DoNotClobber.contains(Reg);
        });
    if (IsUnused)
      return Reg;
  }
  return {};
}

/// Selects a VGPR to load the state value array into for use for the
/// injected payload of \p InstPoint
/// \param InstPoint instrumentation point for which we are selecting a VGPR
/// to load the state value array into
/// \param SVS the state value array storage at the location of \p InstPoint
/// \param InstPointLiveRegs a set of physical registers that are live before
/// the \p InstPoint
/// \param AccessedPhysicalRegsNotInLiveIns a set of physical registers
/// accessed in injected payloads that aren't in the live-ins set of their
/// instrumentation point at the point of access
/// \param ScavengeDeadAVGPRs if \c true then it will try to scavenge a dead
/// A/VGPR that is not used at the instrumentation point; This flag is only
/// set when the state value array storage is fixed
/// \return a pair, with the first element indicating the VGPR selected, and
/// the second element indicating whether the selected VGPR will clobber a
/// live register of the app and needs preserving
static std::pair<llvm::MCRegister, bool>
selectVGPRLoadLocationForInjectedPayload(
    const llvm::MachineInstr &InstPoint, StateValueArrayStorage &SVS,
    const llvm::DenseSet<llvm::MCPhysReg> &InstPointDoNotClobber,
    bool ScavengeDeadAVGPRs) {
  llvm::MCRegister AVGPRLocation{0};
  bool ClobbersAppRegister{false};
  // if the state value array already in a VGPR, then select the same VGPR
  // to be the load destination
  if (!SVS.requiresLoadAndStoreBeforeUse())
    AVGPRLocation = SVS.getStateValueStorageReg();
  else {
    if (!ScavengeDeadAVGPRs) {
      AVGPRLocation = llvm::AMDGPU::VGPR0;
      ClobbersAppRegister = true;
    } else {
      auto &InstrumentedMF = *InstPoint.getParent()->getParent();
      // Scavenge a dead VGPR to hold the state value array
      AVGPRLocation = scavengeFreeRegister(InstrumentedMF,
                                           llvm::AMDGPU::VGPR_32RegClass,
                                           InstPointDoNotClobber);
      // Fall back to a dead AGPR
      if (AVGPRLocation == 0)
        AVGPRLocation = scavengeFreeRegister(InstrumentedMF,
                                             llvm::AMDGPU::AGPR_32RegClass,
                                             InstPointDoNotClobber);
      if (AVGPRLocation == 0) {
        // Last resort: clobber a register the app uses but the payload
        // doesn't depend on at this IP. The PEI will spill it.
        ClobbersAppRegister = true;
        auto &InstrumentedMFRI = InstrumentedMF.getRegInfo();
        for (llvm::MCRegister Reg : llvm::AMDGPU::VGPR_32RegClass) {
          if (InstrumentedMFRI.isPhysRegUsed(Reg) &&
              !InstPointDoNotClobber.contains(Reg)) {
            AVGPRLocation = Reg;
            break;
          }
        }
        if (AVGPRLocation == 0)
          AVGPRLocation = llvm::AMDGPU::VGPR0;
      }
    }
  }
  return {AVGPRLocation, ClobbersAppRegister};
}

/// \brief Tries to find a fixed location for storing the state value array
/// \details The order of searching for the storage location is as follows:
/// 1. Find an unused VGPR. This is the ideal scenario, as no further action
/// is required in the prologue/epilogue of an injected payload to load/store
/// the state value array\n
/// 2. If no unused VGPRs are found, then this routine will find the next
/// unused AGPR. This usually comes at no cost to the occupancy, as the app
/// will get the same amount of AGPRs as it gets VGPRs. In gfx90A-, since
/// AGPRs cannot be used directly by vector instructions and have to be moved
/// to a VGPR, a single application VGPR must be spilled. Preference is
/// given to finding another free AGPR to act as a spill slot. If no other
/// free AGPR is found, then three free SGPRs must be found to spill the
/// app's VGPR into an emergency spill slot in the instrumentation stack.\n
/// 3. If no unused V/AGPRs are found in the kernel or a free AGPR is found
/// but allocation of the spill registers is unsuccessful on gfx90A-,
/// then as a last resort, this function tries to find three free SGPRs
/// that can be used to spill an app's VGPR onto the stack, and load the
/// state value array from the stack
/// TODO: This function must take an argument indicating whether the tool
/// writer wants to respect the original kernel's granulated register usage
/// or not.
static std::shared_ptr<StateValueArrayStorage> findFixedStateValueArrayStorage(
    llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions,
    llvm::ArrayRef<StateValueArrayStorage::StorageKind> SupportedStorage,
    int MaxAGPRsUsedByAllStorage, int MaxSGPRsUsedByAllStorage,
    const llvm::DenseSet<llvm::MCPhysReg> &DoNotClobber) {
  // Find the next VGPR available to hold the value state array
  llvm::MCRegister StateValueArrayFixedVGPRLocation = scavengeFreeRegister(
      RelatedFunctions, &llvm::AMDGPU::VGPR_32RegClass, DoNotClobber);
  // If we failed to find a free VGPR, we then have to scavenge for all
  // possible SGPRs and AGPRs that can be used in storing the state value
  // array
  if (StateValueArrayFixedVGPRLocation == 0) {
    llvm::SmallVector<llvm::MCRegister, 3> SGPRsScavenged;
    llvm::SmallVector<llvm::MCRegister, 2> AGPRsScavenged;
    scavengeFreeRegister(RelatedFunctions, &llvm::AMDGPU::AGPR_32RegClass,
                         DoNotClobber, MaxAGPRsUsedByAllStorage,
                         AGPRsScavenged);
    scavengeFreeRegister(RelatedFunctions, &llvm::AMDGPU::SGPR_32RegClass,
                         DoNotClobber, MaxSGPRsUsedByAllStorage,
                         SGPRsScavenged);

    LLVM_DEBUG(

        luthier::dbgs()
            << "Number of AGPRs scavenged for fixed location SVA storage: "
            << AGPRsScavenged.size() << "\n";
        luthier::dbgs()
        << "Number of SGPRs scavenged for fixed location SVA storage: "
        << SGPRsScavenged.size() << "\n";

    );

    // Loop over all possible supported storage schemes and select the best
    // preferred one which we can use
    for (const auto &StorageScheme : SupportedStorage) {
      if (StorageScheme == StateValueArrayStorage::SVS_SINGLE_VGPR)
        continue;
      LLVM_DEBUG(luthier::dbgs() << "Evaluating fixed " << StorageScheme
                                 << " storage scheme.\n";);
      int NumAGPRsUsedByStorage =
          StateValueArrayStorage::getNumAGPRsUsed(StorageScheme);
      int NumSGPRsUsedByStorage =
          StateValueArrayStorage::getNumSGPRsUsed(StorageScheme);
      LLVM_DEBUG(luthier::dbgs() << "Number of ARGPs required by the scheme: "
                                 << NumAGPRsUsedByStorage << "\n";
                 luthier::dbgs() << "Number of SGPRs required by the scheme: "
                                 << NumSGPRsUsedByStorage << "\n";);
      if (NumSGPRsUsedByStorage <= SGPRsScavenged.size() &&
          NumAGPRsUsedByStorage <= AGPRsScavenged.size()) {
        LLVM_DEBUG(luthier::dbgs()
                       << "Found a suitable fixed storage scheme!\n";);
        auto Out = StateValueArrayStorage::createSVAStorage(
            {}, AGPRsScavenged, SGPRsScavenged, StorageScheme);
        if (Out.takeError()) {
          return nullptr;
        }
        return std::move(*Out);
      }
    }
    // If we made it out of the loop, we weren't able to find a fixed location
    // for the state value array, so we return nullptr
    return nullptr;
  } else
    return std::make_shared<VGPRStateValueArrayStorage>(
        StateValueArrayFixedVGPRLocation);
}

static std::shared_ptr<StateValueArrayStorage> findStateValueArrayStorageAtMI(
    const llvm::MachineFunction &MF,
    const llvm::DenseSet<llvm::MCPhysReg> &DoNotClobber,
    llvm::ArrayRef<StateValueArrayStorage::StorageKind> SupportedStorage,
    int MaxAGPRsUsedByAllStorage, int MaxSGPRsUsedByAllStorage) {
  // Find the next VGPR available to hold the value state array
  llvm::MCRegister StateValueArrayVGPRLocation =
      scavengeFreeRegister(MF, llvm::AMDGPU::VGPR_32RegClass, DoNotClobber);
  // If we failed to find a free VGPR, we then have to scavenge for all
  // possible SGPRs and AGPRs that can be used in storing the state value
  // array
  if (StateValueArrayVGPRLocation == 0) {
    llvm::SmallVector<llvm::MCRegister, 3> SGPRsScavenged;
    llvm::SmallVector<llvm::MCRegister, 2> AGPRsScavenged;
    scavengeFreeRegister(MF, llvm::AMDGPU::AGPR_32RegClass, DoNotClobber,
                         MaxAGPRsUsedByAllStorage, AGPRsScavenged);

    scavengeFreeRegister(MF, llvm::AMDGPU::SGPR_32RegClass, DoNotClobber,
                         MaxSGPRsUsedByAllStorage, SGPRsScavenged);

    LLVM_DEBUG(

        luthier::dbgs()
            << "Number of AGPRs scavenged for location SVA storage: "
            << AGPRsScavenged.size() << "\n";
        luthier::dbgs()
        << "Number of SGPRs scavenged for location SVA storage: "
        << SGPRsScavenged.size() << "\n";

    );

    // Loop over all possible supported storage schemes and select the best
    // preferred one which we can use
    for (const auto &StorageScheme : SupportedStorage) {
      if (StorageScheme == StateValueArrayStorage::SVS_SINGLE_VGPR)
        continue;
      int NumAGPRsUsedByStorage =
          StateValueArrayStorage::getNumAGPRsUsed(StorageScheme);
      int NumSGPRsUsedByStorage =
          StateValueArrayStorage::getNumSGPRsUsed(StorageScheme);
      if (NumSGPRsUsedByStorage <= SGPRsScavenged.size() &&
          NumAGPRsUsedByStorage <= AGPRsScavenged.size()) {
        auto Out = StateValueArrayStorage::createSVAStorage(
            {}, AGPRsScavenged, SGPRsScavenged, StorageScheme);
        if (Out.takeError()) {
          return nullptr;
        }
        return std::move(*Out);
      }
    }
    // If we made it out of the loop, we weren't able to find a location
    // for the state value array, so we return nullptr
    return nullptr;
  } else
    return std::make_shared<VGPRStateValueArrayStorage>(
        StateValueArrayVGPRLocation);
}

llvm::ArrayRef<StateValueStorageSegment>
SVStorageAndLoadLocations::getStorageIntervals(
    const llvm::MachineBasicBlock &MBB) const {
  auto It = StateValueStorageIntervals.find(&MBB);
  if (It == StateValueStorageIntervals.end())
    return {};
  else
    return It->second;
}

const InstPointSVALoadPlan *
SVStorageAndLoadLocations::getStateValueArrayLoadPlanForInstPoint(
    const llvm::MachineInstr &MI) const {
  auto It = InstPointSVSLoadPlans.find(&MI);
  if (It == InstPointSVSLoadPlans.end())
    return nullptr;
  else
    return &It->second;
}

llvm::Error SVStorageAndLoadLocations::calculate(
    const llvm::MachineModuleInfo &TargetMMI, const llvm::Module &TargetM,
    llvm::FunctionAnalysisManager &TargetFAM,
    llvm::MachineFunctionAnalysisManager &TargetMFAM,
    const InjectedPayloadAndInstPoint &IPIP, FunctionPreambleDescriptor &FPD,
    const llvm::DenseMap<const llvm::Function *, PayloadLiveSets>
        &PayloadLiveSetsByFn) {
  // Module-wide "do not clobber" set: every register that's live in any
  // payload's Active or Inactive lane set, anywhere in the module. Used by
  // the fixed-storage scavenger and as a starting point for per-IP queries.
  llvm::DenseSet<llvm::MCPhysReg> AllPayloadsDoNotClobber =
      liveAcrossAllPayloads(PayloadLiveSetsByFn);

  // Target MFs live in the new-PM target FAM cache (CodeDiscoveryPass
  // populates them via FAM.getResult<MachineFunctionAnalysis>(F).getMF()
  // — MachineFunctionAnalysis is a Function-level analysis registered
  // in FAM). Pull via TargetFAM rather than the legacy IMMI we receive
  // (which only contains the IModule's payloads/hooks).
  llvm::SmallVector<llvm::MachineFunction *, 4> MFs;
  for (llvm::Function &F : const_cast<llvm::Module &>(TargetM)) {
    if (F.isDeclaration())
      continue;
    MFs.push_back(&TargetFAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF());
  }

  // Early exit if no MF is present in the LCO of LR
  if (MFs.empty())
    return llvm::Error::success();
  // Get all the possible state value array storage for the sub-target being
  // used and check if we have at least only one method for storage
  const auto &ST = MFs[0]->getSubtarget<llvm::GCNSubtarget>();
  llvm::SmallVector<StateValueArrayStorage::StorageKind, 6> SupportedStorage;
  getSupportedSVAStorageList(ST, SupportedStorage);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !SupportedStorage.empty(),
      llvm::formatv("Failed to find compatible state value array storage "
                    "for ST {0}, CPU {1}.",
                    ST.getTargetTriple().str(), ST.getCPU())));
  // Query the maximum number of SGPRs and AGPRs in all storage methods;
  // This saves us time during register scavenging
  int MaxNumAGPRsUsedByAllStorage = 0;
  int MaxNumSGPRsUsedByAllStorage = 0;
  for (const auto &StorageScheme : SupportedStorage) {
    int MaxNumAGPRsUsedByStorage =
        StateValueArrayStorage::getNumAGPRsUsed(StorageScheme);
    if (MaxNumAGPRsUsedByStorage > MaxNumAGPRsUsedByAllStorage)
      MaxNumAGPRsUsedByAllStorage = MaxNumAGPRsUsedByStorage;
    int MaxNumSGPRsUsedByStorage =
        StateValueArrayStorage::getNumSGPRsUsed(StorageScheme);
    if (MaxNumSGPRsUsedByStorage > MaxNumSGPRsUsedByAllStorage)
      MaxNumSGPRsUsedByAllStorage = MaxNumSGPRsUsedByStorage;
  }

  // Try to find a fixed location to store the state value array
  auto StateValueFixedLocation = findFixedStateValueArrayStorage(
      MFs, SupportedStorage, MaxNumAGPRsUsedByAllStorage,
      MaxNumSGPRsUsedByAllStorage, AllPayloadsDoNotClobber);

  if (StateValueFixedLocation != nullptr) {
    // If a fixed location was found, then all MBB intervals inside all MFs
    // will get the fixed state value location
    // Also in a fixed storage case, there is no need to emit any kind of
    // preamble code to any device functions involved inside the lifted
    // representation
    for (const auto &MF : MFs) {
      for (const auto &MBB : *MF) {
        auto &Segments =
            StateValueStorageIntervals
                .insert({&MBB, llvm::SmallVector<StateValueStorageSegment>{}})
                .first->getSecond();
        Segments.emplace_back(TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(*MF).getMBBStartIdx(&MBB),
                              TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(*MF).getMBBEndIdx(&MBB),
                              StateValueFixedLocation);
      }
      if (MF->getFunction().getCallingConv() !=
          llvm::CallingConv::AMDGPU_KERNEL) {
        FPD.DeviceFunctions[MF].RequiresPreAndPostAmble = false;
      }
    }
    for (const auto &[InsertionPointMI, HookFunctions] : IPIP.mi_payloads()) {
      llvm::DenseSet<llvm::MCPhysReg> IPDoNotClobber =
          liveAtAppMI(*InsertionPointMI, IPIP, PayloadLiveSetsByFn);
      auto [VGPRLocation, ClobbersAppReg] =
          selectVGPRLoadLocationForInjectedPayload(
              *InsertionPointMI, *StateValueFixedLocation, IPDoNotClobber,
              true);

      InstPointSVSLoadPlans.insert(
          {InsertionPointMI, InstPointSVALoadPlan{VGPRLocation, ClobbersAppReg,
                                                  *StateValueFixedLocation}});
    }
  } else {
    // If not, we'll have to shuffle between possible state value array
    // storage schemes
    for (const auto &MF : MFs) {
      if (MF->getFunction().getCallingConv() ==
          llvm::CallingConv::AMDGPU_KERNEL) {
        FPD.DeviceFunctions[MF].RequiresPreAndPostAmble = true;
      }
      auto &MRI = MF->getRegInfo();
      // Pick the highest numbered VGPR not accessed by the Hooks
      // to hold the value state
      // TODO: is there a more informed way to initialize this?
      // TODO: if an argument is passed specifying to keep the register
      // usage of the kernel the same as before, this needs to be initialized
      // to the last available SGPR/VGPR/AGPR
      llvm::DenseSet<llvm::MCPhysReg> EntryDoNotClobber =
          AllPayloadsDoNotClobber;
      if (!MF->empty())
        for (const auto &LI : MF->front().liveins())
          EntryDoNotClobber.insert(LI.PhysReg);

      // The current location of the state value register
      std::shared_ptr<StateValueArrayStorage> SVS =
          findStateValueArrayStorageAtMI(
              *MF, EntryDoNotClobber, SupportedStorage,
              MaxNumAGPRsUsedByAllStorage, MaxNumSGPRsUsedByAllStorage);

      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          SVS != nullptr,
          llvm::formatv("Failed to get a state value array storage for MI {0}.",
                        *MF->begin()->begin())));

      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          llvm::isa<VGPRStateValueArrayStorage>(SVS.get()) ||
              llvm::isa<SingleAGPRStateValueArrayStorage>(SVS.get()),
          "The entry SVS must be stored in a VGPR or an AGPR."));

      // A set of hook insertion points that fall into the current interval
      llvm::SmallDenseSet<const llvm::MachineInstr *, 4>
          HookInsertionPointsInCurrentSegment{};
      for (const auto &MBB : *MF) {
        // Marks the beginning of the current interval we are in this loop
        llvm::SlotIndex CurrentIntervalBegin =
            TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(*MF).getMBBStartIdx(&MBB);

        auto &CurrentMBBSegments =
            StateValueStorageIntervals.insert({&MBB, {}}).first->getSecond();
        for (const auto &MI : MBB) {
          if (IPIP.contains(MI))
            HookInsertionPointsInCurrentSegment.insert(&MI);
          // Per-MI "do not clobber" set: union of every payload-live set
          // at this MI (if it's an instrumentation point; empty
          // otherwise). For non-IP MIs we have no live-reg query in the
          // new design — the conservative AllPayloadsDoNotClobber set
          // is used as a fallback, mirroring the old code's behavior
          // when InstrLiveRegs were unavailable.
          llvm::DenseSet<llvm::MCPhysReg> MIDoNotClobber =
              IPIP.contains(MI)
                  ? liveAtAppMI(MI, IPIP, PayloadLiveSetsByFn)
                  : AllPayloadsDoNotClobber;
          // - If we have spilled the state value reg and this instruction
          // will require a hook to be inserted, then we try to relocate the
          // SVS. In this instance, since the hook will have to load the value
          // state register anyway, we try and see if after loading it, we can
          // store it in a V/AGPR.
          // - If the SVS registers are going to be used, we must relocate
          // the SVS.
          // - Otherwise, we keep the SVS in its place.
          bool TryRelocatingValueStateReg =
              SVS->getStateValueStorageReg() == 0 && IPIP.contains(MI);
          llvm::SmallVector<llvm::MCRegister, 4> SVSRegs;
          SVS->getAllStorageRegisters(SVSRegs);
          bool MustRelocateStateValue =
              llvm::any_of(SVSRegs, [&](llvm::MCRegister Reg) {
                return MIDoNotClobber.contains(Reg);
              });
          // If we have to relocate something, then create a new interval
          // for it;
          // Note that reg scavenging might conclude that the values remain
          // where they are, and that's okay
          // Also create a new interval if we reach the end of a MBB
          if (&MI == &MBB.back() || TryRelocatingValueStateReg ||
              MustRelocateStateValue) {
            auto NextIndex = &MI == &MBB.back()
                                 ? TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(*MF).getMBBEndIdx(&MBB)
                                 : TargetMFAM.getResult<llvm::SlotIndexesAnalysis>(*MF).getInstructionIndex(MI);
            CurrentMBBSegments.emplace_back(CurrentIntervalBegin, NextIndex,
                                            SVS);
            for (const auto &HookMI : HookInsertionPointsInCurrentSegment) {
              llvm::DenseSet<llvm::MCPhysReg> HookDoNotClobber =
                  liveAtAppMI(*HookMI, IPIP, PayloadLiveSetsByFn);
              auto [HookSVGPR, ClobbersAppReg] =
                  selectVGPRLoadLocationForInjectedPayload(
                      *HookMI, *SVS, HookDoNotClobber, false);
              InstPointSVSLoadPlans.insert(
                  {HookMI, {HookSVGPR, ClobbersAppReg, *SVS}});
            }
            HookInsertionPointsInCurrentSegment.clear();
            CurrentIntervalBegin = NextIndex;
          }
          if (TryRelocatingValueStateReg || MustRelocateStateValue) {
            SVS = findStateValueArrayStorageAtMI(
                *MF, MIDoNotClobber, SupportedStorage,
                MaxNumAGPRsUsedByAllStorage, MaxNumSGPRsUsedByAllStorage);
            LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
                SVS != nullptr, "Failed to relocate the SVA storage."));
          }
        }
      }
    }
  }
  return llvm::Error::success();
}

llvm::AnalysisKey SVStorageAndLoadLocationsAnalysis::Key;

SVStorageAndLoadLocationsAnalysis::Result
SVStorageAndLoadLocationsAnalysis::run(
    Prototype &IP, PrototypeAnalysisManager &IPAM) {
  Result Out;

  llvm::Module &TargetModule = IP.getTargetModule();
  llvm::Module &IModule = IP.getInstrumentationModule();

  // The IP-side proxy hands out the outer ModuleAnalysisManager; the same
  // MAM caches results for both modules, keyed by the module reference.
  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<ModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager();

  llvm::FunctionAnalysisManager &TargetFAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();
  llvm::MachineFunctionAnalysisManager &TargetMFAM =
      MAM.getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(
                TargetModule)
          .getManager();

  const llvm::MachineModuleInfo &TargetMMI =
      MAM.getResult<llvm::MachineModuleAnalysis>(TargetModule).getMMI();

  const InjectedPayloadAndInstPoint &IPIP =
      IPAM.getResult<InjectedPayloadAndInstPointAnalysis>(IP);

  FunctionPreambleDescriptor &FPD =
      IPAM.getResult<FunctionPreambleDescriptorAnalysis>(IP);

  // TODO(NPM): source PayloadLiveSetsByFn from a migrated
  // IP-level liveness analysis. IModuleIPPredicatedLivenessAnalysis is still
  // a legacy ModulePass and cannot be pulled through IPAM; consumers of
  // this analysis are broken until that migration lands.
  llvm::DenseMap<const llvm::Function *, PayloadLiveSets> PayloadLiveSetsByFn;

  if (auto Err = Out.calculate(TargetMMI, TargetModule, TargetFAM, TargetMFAM,
                               IPIP, FPD, PayloadLiveSetsByFn))
    TargetModule.getContext().emitError(llvm::toString(std::move(Err)));

  return Out;
}

} // namespace luthier