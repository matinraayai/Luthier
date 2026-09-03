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

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR);

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR);

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR);

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR);

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource);

/// Emits an \c S_WAITCNT before \p MI with the given per-counter
/// encoding \p Encoding.
void emitWaitCnt(llvm::MachineBasicBlock::iterator MI, unsigned Encoding = 0);

void emitWaitCnt(llvm::MachineBasicBlock &MBB, unsigned Encoding = 0);

} // namespace luthier

#endif