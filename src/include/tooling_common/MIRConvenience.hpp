//===-- MIRConvenience.hpp ------------------------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
#ifndef LUTHIER_TOOLING_COMMON_MIR_CONVENIENCE_HPP
#define LUTHIER_TOOLING_COMMON_MIR_CONVENIENCE_HPP
#include <llvm/CodeGen/MachineBasicBlock.h>

namespace llvm {

class MCRegister;

}

namespace luthier {

/// Swaps the value between \p ScrSGPR and \p DestSGPR by inserting 3
/// <tt>S_XOR_B32</tt>s before \p InsertionPoint
void emitSGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR);

/// Swaps the value between \p ScrVGPR and \p DestVGPR by inserting 3
/// <tt>V_XOR_B32_e32</tt>s before \p InsertionPoint
void emitVGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR);

/// Emits an instruction that flips the exec mask before \p MI
/// Clobbers the SCC bit
void emitExecMaskFlip(llvm::MachineBasicBlock::iterator MI);

void emitMoveFromVGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR,
                            bool KillSource);

void emitMoveFromSGPRToSGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR,
                            bool KillSource);

void emitMoveFromAGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcAGPR, llvm::MCRegister DestVGPR,
                            bool KillSource);

void emitMoveFromVGPRToAGPR(llvm::MachineBasicBlock::iterator MI,
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

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR);

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource);

void emitWaitCnt(llvm::MachineBasicBlock::iterator MI);

} // namespace luthier

#endif