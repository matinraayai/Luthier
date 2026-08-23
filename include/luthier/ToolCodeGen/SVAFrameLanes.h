//===-- SVAFrameLanes.h ----------------------------------------*- C++ -*-===//
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
/// \file SVAFrameLanes.h
/// Per-physreg → SVA-lane mapping for the kernel-prolog frame setup and the
/// injected-payload PEI's app-frame preservation. Replaces the defunct
/// \c namespace stateValueArray family of compile-time helpers; the lane
/// numbers come directly from the \c StateValueArraySpecs layout that
/// \c StateValueArraySpecsAnalysis produces from the instrumentation
/// module's \c luthier::readSVA call sites.
///
/// The kernel prolog (PrePostAmbleEmitter) temporarily stashes four
/// kernarg-derived registers — \c SGPR0, \c SGPR1, \c FLAT_SCR_LO,
/// \c FLAT_SCR_HI — into SVA lanes 0-3 while it computes per-wave scratch
/// offsets, then restores them. The injected-payload prologue
/// (InjectedPayloadPEIPass) writes the app's \c SGPR32 / \c FLAT_SCR_LO
/// (and \c FLAT_SCR_HI on non-architected-FS) into the same set of lanes
/// so the payload can reuse those physregs for its own frame, and reads
/// the instrumentation-private frame values back from the "store" lanes
/// the prolog populated.
///
/// Layout (matches what \c StateValueArraySpecsAnalysis produces):
///
///   Lane | Spilled phys-reg (kernarg / app value)
///   -----|----------------------------------------------------------------
///     0  | \c SGPR0 (= PRIVATE_SEGMENT_BUFFER sub0). Per StateValueArraySpecs
///        | this is \c StackPointerRegSpillLane.
///     1  | \c SGPR1 (= PRIVATE_SEGMENT_BUFFER sub1).
///        | \c FramePointerRegSpillLane.
///     2  | \c FLAT_SCR_LO on absolute-FS targets, OR the second buffer-rsrc
///        | sub-word on PRIVATE_RSRC targets. Per
///        | \c StateValueArraySpecs::StackPointerStoreLane (named for the
///        | SGPR32-store role; the lane is shared with FS_LO).
///     3  | \c FLAT_SCR_HI on absolute-FS, OR third buffer-rsrc sub-word.
///        | \c StateValueArraySpecs::getFrameRsrcOrScratchStoreLaneIfExists().
///     4+ | Per-SA lanes (start point depends on whether FS or buffer rsrc
///        | took 2 or 4 lanes).
///
/// The "store" slots — where the kernel prolog writes the per-wave-computed
/// values that the payload reads back — are at the same lane indices but
/// referenced via the *load* direction. The kernel prolog overwrites lanes
/// 0-3 with the post-S_ADD per-wave values, and the payload prologue reads
/// them out. See \c PrePostAmbleEmitter::emitCodeToSetupScratch for the
/// kernel-prolog side.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_SVA_FRAME_LANES_H
#define LUTHIER_TOOL_CODE_GEN_SVA_FRAME_LANES_H
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include <AMDGPU.h>
#include <MCTargetDesc/AMDGPUMCTargetDesc.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/MC/MCRegister.h>
#include <optional>

namespace luthier {

/// Returns the SVA lane index for the kernarg-derived register \p PhysReg
/// in the kernel-prolog spill slot — i.e., where the prolog writes
/// \p PhysReg's original value before doing the per-wave scratch
/// arithmetic. Returns \c std::nullopt if \p PhysReg is not one of the
/// four kernel-prolog-managed registers.
inline std::optional<uint8_t>
getKernelPrologFrameSpillLane(llvm::MCRegister PhysReg,
                              const StateValueArraySpecs &Specs) {
  if (PhysReg == llvm::AMDGPU::SGPR0)
    return Specs.getStackPointerRegSpillLane();
  if (PhysReg == llvm::AMDGPU::SGPR1)
    return Specs.getFramePointerRegSpillLane();
  if (PhysReg == llvm::AMDGPU::FLAT_SCR_LO)
    return Specs.getStackPointerStoreLane();
  if (PhysReg == llvm::AMDGPU::FLAT_SCR_HI) {
    auto Base = Specs.getFrameRsrcOrScratchStoreLaneIfExists();
    if (!Base)
      return std::nullopt;
    return *Base;
  }
  return std::nullopt;
}

/// Returns the SVA lane index where the instrumentation-private value of
/// \p PhysReg lives — i.e., the lane the payload prologue reads to bring
/// the per-wave-computed instrumentation frame register into the physical
/// register \p PhysReg. By the kernel-prolog/PEI lane-sharing convention
/// described in this file, this is the same lane as the spill slot
/// (the prolog overwrites the spilled value with the per-wave value).
inline std::optional<uint8_t>
getInstrumentationFrameStoreLane(llvm::MCRegister PhysReg,
                                 const StateValueArraySpecs &Specs) {
  return getKernelPrologFrameSpillLane(PhysReg, Specs);
}

/// The 4-register ordered table the kernel prolog walks to spill / restore
/// the kernarg state registers.
inline llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>, 4>
getKernelPrologFrameSpillSlots(const StateValueArraySpecs &Specs) {
  llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>, 4> Out;
  for (llvm::MCRegister R :
       {llvm::MCRegister(llvm::AMDGPU::SGPR0),
        llvm::MCRegister(llvm::AMDGPU::SGPR1),
        llvm::MCRegister(llvm::AMDGPU::FLAT_SCR_LO),
        llvm::MCRegister(llvm::AMDGPU::FLAT_SCR_HI)}) {
    if (auto L = getKernelPrologFrameSpillLane(R, Specs))
      Out.push_back({R, *L});
  }
  return Out;
}

/// Symmetric "store-side" table: same physregs, same lanes, but
/// conceptually the lane is read FROM (not written TO) at this point in
/// the kernel prolog (after S_ADD has updated the physical register, the
/// prolog writes the new value back to the same lane the spill came from).
inline llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>, 4>
getKernelPrologFrameStoreSlots(const StateValueArraySpecs &Specs) {
  return getKernelPrologFrameSpillSlots(Specs);
}

/// The app-frame registers the injected-payload PEI must spill into the
/// SVA on prologue and restore on epilogue. SGPR32 is the app's stack
/// pointer; FLAT_SCR_LO is the FP-analogue on absolute-FS targets.
inline llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>, 2>
getPayloadAppFrameSpillSlots(const StateValueArraySpecs &Specs) {
  llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>, 2> Out;
  Out.push_back({llvm::AMDGPU::SGPR32, Specs.getStackPointerRegSpillLane()});
  if (auto FPLane = Specs.getFramePointerRegSpillLane();
      true /* always emitted; lane is constexpr */) {
    Out.push_back({llvm::AMDGPU::FLAT_SCR_LO, FPLane});
  }
  return Out;
}

/// Symmetric "load-side" table the payload prologue consults to bring
/// instrumentation-private SP/FS values out of the SVA into physregs.
inline llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>, 2>
getPayloadAppFrameLoadSlots(const StateValueArraySpecs &Specs) {
  llvm::SmallVector<std::pair<llvm::MCRegister, uint8_t>, 2> Out;
  Out.push_back({llvm::AMDGPU::SGPR32, Specs.getStackPointerStoreLane()});
  if (auto FrameLane = Specs.getFrameRsrcOrScratchStoreLaneIfExists())
    Out.push_back({llvm::AMDGPU::FLAT_SCR_LO, *FrameLane});
  return Out;
}

} // namespace luthier

#endif
