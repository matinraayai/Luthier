//===-- CodeGenHelpers.cpp ------------------------------------------------===//
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
/// \file This file implements the CodeGen helper functions.
//===----------------------------------------------------------------------===//
#include "luthier/LLVM/CodeGenHelpers.h"
#include "SIInstrInfo.h"

namespace luthier {

bool isScalar(const llvm::MachineInstr &MI) {
  return llvm::SIInstrInfo::isSOP1(MI) || llvm::SIInstrInfo::isSOP2(MI) ||
         llvm::SIInstrInfo::isSOPK(MI) || llvm::SIInstrInfo::isSOPC(MI) ||
         llvm::SIInstrInfo::isSOPP(MI) || llvm::SIInstrInfo::isSMRD(MI);
}

bool isLaneAccess(const llvm::MachineInstr &MI) {
  return MI.getOpcode() == llvm::AMDGPU::V_READFIRSTLANE_B32 ||
         MI.getOpcode() == llvm::AMDGPU::V_READLANE_B32 ||
         MI.getOpcode() == llvm::AMDGPU::V_WRITELANE_B32;
}

bool isVector(const llvm::MachineInstr &MI) {
  return !(isScalar(MI) || isLaneAccess(MI));
}

} // namespace luthier