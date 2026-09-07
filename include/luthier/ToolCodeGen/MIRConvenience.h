//===-- MIRConvenience.h ----------------------------------------*- C++ -*-===//
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
/// This file contains a set of high-level convenience functions used to write
/// MIR instructions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_MIR_CONVENIENCE_H
#define LUTHIER_TOOL_CODE_GEN_MIR_CONVENIENCE_H
#include <llvm/CodeGen/MachineBasicBlock.h>

namespace llvm {

class MCRegister;
class GCNSubtarget;

}

namespace luthier {

/// Returns true if \p MI implicitly reads the EXEC mask — i.e. it is a
/// VALU (excluding the uniform read/write-lane family) or any
/// non-scalar/non-SMRD instruction. Used as the canonical predicate for
/// classifying MIs as vector (EXEC-dependent) vs scalar (uniform).
///
/// Mirrors the matching helper in \c CodeDiscoveryPass that drives its
/// pure-scalar vs pure-vector MBB splitting. Promoted here so other
/// passes (e.g. liveness) can ask the same question.
bool shouldImplicitReadExec(const llvm::MachineInstr &MI);

/// Returns true if \p MBB is a vector machine basic block (its instructions
/// execute only on lanes selected by the current EXEC mask), false if it
/// is scalar (its instructions are uniform / execute on all lanes
/// regardless of EXEC).
///
/// Determined by asking \c shouldImplicitReadExec on the first
/// non-debug instruction of \p MBB. \c CodeDiscoveryPass guarantees each
/// MBB is pure-scalar or pure-vector, so the first instruction suffices.
/// Empty MBBs return \c false (scalar — there are no instructions to
/// read EXEC).
bool isVectorMBB(const llvm::MachineBasicBlock &MBB);

/// Swaps the value between \p ScrSGPR and \p DestSGPR by inserting 3
/// <tt>S_XOR_B32</tt>s before \p InsertionPoint
void emitSGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR);

/// MBB-appending overload: inserts at the end of \p MBB. Safe to call on an
/// empty MBB (does not dereference a sentinel iterator, unlike the
/// iterator-based overload).
void emitSGPRSwap(llvm::MachineBasicBlock &MBB, llvm::MCRegister SrcSGPR,
                  llvm::MCRegister DestSGPR);

/// Exchange the wave's live \c FLAT_SCR pair with the pair of SGPRs a state
/// value array storage scheme keeps the instrumentation's flat scratch base
/// in, appending the moves at the end of \p MBB . Calling it a second time
/// undoes the first, so a scratch access is bracketed by two identical calls.
///
/// The mechanism is generation-dependent, which is why the two absolute-flat-
/// scratch storage schemes are split:
///   * **GFX9** — \c FLAT_SCR_LO / \c FLAT_SCR_HI are ordinary addressable
///     SGPRs, so each half is exchanged with a three-\c S_XOR_B32 swap and
///     \p TempSGPR is not touched.
///   * **GFX10+** — \c FLAT_SCR is reachable only through
///     \c S_GETREG_B32 / \c S_SETREG_B32 (see
///     \c SIFrameLowering::emitEntryFunctionFlatScratchInit ), and neither
///     can name a second register, so there is no XOR trick available. Each
///     half becomes a three-move rotation through \p TempSGPR :
///     <tt>getreg TempSGPR</tt>, <tt>setreg InstFS</tt>,
///     <tt>InstFS = TempSGPR</tt>.
///
/// \param MBB block to append the swap to
/// \param InstFSLo SGPR holding the lo half of the instrumentation's flat
/// scratch base on entry, and the application's on exit
/// \param InstFSHi as \p InstFSLo , for the hi half
/// \param TempSGPR a scratch SGPR, clobbered; only read on GFX10+
void emitFlatScratchSwap(llvm::MachineBasicBlock &MBB,
                         llvm::MCRegister InstFSLo, llvm::MCRegister InstFSHi,
                         llvm::MCRegister TempSGPR);

/// Park the wave's live \c FLAT_SCR pair in SVA lanes
/// <tt>[Lane, Lane + 1]</tt> of \p SVAVGPR , appending at the end of
/// \p MBB .
///
/// \p TempSGPR is only read on GFX10+, where \c FLAT_SCR cannot be a
/// \c V_WRITELANE_B32 source (nor an \c S_MOV_B32 one) and has to be pulled
/// out with \c S_GETREG_B32 first. On GFX9 the lane moves name \c FLAT_SCR_LO
/// and \c FLAT_SCR_HI directly and \p TempSGPR may be a null register.
void emitFlatScratchSaveToSVALanes(llvm::MachineBasicBlock &MBB,
                                   llvm::MCRegister SVAVGPR, unsigned Lane,
                                   llvm::MCRegister TempSGPR);

/// Inverse of \c emitFlatScratchSaveToSVALanes : install \c FLAT_SCR from SVA
/// lanes <tt>[Lane, Lane + 1]</tt> of \p SVAVGPR . Used both to restore a
/// parked pair and to install the instrumentation's own base out of the SVA's
/// \c FLAT_SCRATCH scalar-argument lanes.
void emitFlatScratchLoadFromSVALanes(llvm::MachineBasicBlock &MBB,
                                     llvm::MCRegister SVAVGPR, unsigned Lane,
                                     llvm::MCRegister TempSGPR);

/// Swaps the value between \p ScrVGPR and \p DestVGPR by inserting 3
/// <tt>V_XOR_B32_e32</tt>s before \p InsertionPoint
void emitVGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR);

/// MBB-appending overload — see \c emitSGPRSwap.
void emitVGPRSwap(llvm::MachineBasicBlock &MBB, llvm::MCRegister SrcVGPR,
                  llvm::MCRegister DestVGPR);

/// Emits an instruction that flips the exec mask before \p MI
/// Clobbers the SCC bit
void emitExecMaskFlip(llvm::MachineBasicBlock::iterator MI);

/// MBB-appending overload — see \c emitSGPRSwap.
void emitExecMaskFlip(llvm::MachineBasicBlock &MBB);

/// Explicit (block, insertion point) overload. Unlike the iterator-only form
/// this one never dereferences \p InsertionPoint , so it is safe to pass
/// <tt>MBB.end()</tt> or the result of \c getFirstTerminator on a block with
/// no terminator.
void emitExecMaskFlip(llvm::MachineBasicBlock &MBB,
                      llvm::MachineBasicBlock::iterator InsertionPoint);

void emitMoveFromVGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR,
                            bool KillSource);

void emitMoveFromVGPRToVGPR(llvm::MachineBasicBlock &MBB,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR,
                            bool KillSource);

void emitMoveFromSGPRToSGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR,
                            bool KillSource);

void emitMoveFromSGPRToSGPR(llvm::MachineBasicBlock &MBB,
                            llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR,
                            bool KillSource);

void emitMoveFromAGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcAGPR, llvm::MCRegister DestVGPR,
                            bool KillSource);

void emitMoveFromAGPRToVGPR(llvm::MachineBasicBlock &MBB,
                            llvm::MCRegister SrcAGPR, llvm::MCRegister DestVGPR,
                            bool KillSource);

void emitMoveFromVGPRToAGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestAGPR,
                            bool KillSource = true);

void emitMoveFromVGPRToAGPR(llvm::MachineBasicBlock &MBB,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestAGPR,
                            bool KillSource = true);

void emitMoveFromSGPRToVGPRLane(llvm::MachineBasicBlock::iterator MI,
                                llvm::MCRegister SrcSGPR,
                                llvm::MCRegister DestVGPR, unsigned int Lane,
                                bool KillSource);

void emitMoveFromVGPRLaneToSGPR(llvm::MachineBasicBlock::iterator MI,
                                llvm::MCRegister SrcVGPR,
                                llvm::MCRegister DestSGPR, unsigned int Lane,
                                bool KillSource);

/// MBB-appending overload — see \c emitMoveFromSGPRToVGPRLane.
void emitMoveFromSGPRToVGPRLane(llvm::MachineBasicBlock &MBB,
                                llvm::MCRegister SrcSGPR,
                                llvm::MCRegister DestVGPR, unsigned int Lane,
                                bool KillSource);

/// MBB-appending overload — see \c emitMoveFromVGPRLaneToSGPR.
void emitMoveFromVGPRLaneToSGPR(llvm::MachineBasicBlock &MBB,
                                llvm::MCRegister SrcVGPR,
                                llvm::MCRegister DestSGPR, unsigned int Lane,
                                bool KillSource);

/// Generates a set of MBBs that ensures the \c llvm::AMDGPU::SCC bit does not
/// get clobbered due to the sequence of instructions built by \p MIBuilder
/// before the insertion point \p MI
/// This is a common pattern used when loading and storing the state value
/// array that allows flipping the exec mask without clobbering the
/// \c SCC bit and not requiring temporary registers
/// \returns the iterator where all paths emitted converge together
llvm::MachineBasicBlock::iterator createSCCSafeSequenceOfMIs(
    llvm::MachineBasicBlock::iterator MI,
    const std::function<void(llvm::MachineBasicBlock &,
                             const llvm::TargetInstrInfo &)> &MIBuilder);

/// The two emergency slots live at absolute offsets 0 and 4 of the wavefront's
/// private segment, because the instrumentation stack starts there — see
/// \c InstrumentationSlotsReservation . Nothing has to be added to reach them,
/// so these accesses need no address register at all where the hardware can
/// encode a SADDR-less \c SCRATCH_* .
///
/// \p SADDRScratchSGPR is that one exception. GFX9 absolute-flat-scratch
/// targets support neither the ST addressing mode
/// ( \c GCNSubtarget::hasFlatScratchSTMode , which needs gfx10.3 or gfx940 )
/// nor the \c null SGPR, so a real register holding zero has to be handed to
/// the SADDR operand. Each caller passes the scratch SGPR its storage scheme
/// reserves for exactly this; the emitter zeroes it immediately before use.
/// On every other target the argument is ignored and may be a null register.
void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister DestVGPR);

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister DestVGPR);

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister DestVGPR);

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister DestVGPR);

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister SrcVGPR, bool KillSource);

/// Byte size of the instrumentation stack's reserved two-slot carve-out.
///
/// Luthier's instrumentation stack is pinned to the start of the wavefront's
/// private segment — the instrumentation stack pointer is the constant zero —
/// and these two slots sit at the very bottom of it:
///   * \c [0, 4) — emergency \c VGPR0 courier slot.
///   * \c [4, 8) — emergency state-value-array slot.
/// A payload's own frame therefore begins at
/// \c InstrumentationSlotsReservation , and the application's frame begins
/// above the whole instrumentation reserve —
/// \c WORK_ITEM_INSTRUMENTATION_PRIVATE_SEGMENT_SIZE bytes up — which is the
/// displacement \c RebaseAppScratchAccessesPass adds to every application
/// scratch access.
inline constexpr unsigned InstrumentationSlotsReservation = 8;

/// Mirrors \c getScratchScaleFactor in LLVM's \c SIFrameLowering.cpp: a stack
/// pointer register holds a plain byte offset when flat scratch is enabled, and
/// a wave-swizzled offset (bytes times the wavefront size) when scratch is
/// addressed through the private segment buffer instead. Luthier's own
/// emergency-slot accesses no longer need this — they address absolute offsets
/// with a flat \c SCRATCH_* instruction — but a payload's \c SGPR32 and any
/// value shared with LLVM-generated frame code still has to agree with the
/// subtarget's convention.
unsigned getScratchScaleFactor(const llvm::GCNSubtarget &ST);

/// Emits an \c S_WAITCNT before \p MI with the given per-counter
/// encoding \p Encoding.
void emitWaitCnt(llvm::MachineBasicBlock::iterator MI, unsigned Encoding = 0);

void emitWaitCnt(llvm::MachineBasicBlock &MBB, unsigned Encoding = 0);

} // namespace luthier

#endif