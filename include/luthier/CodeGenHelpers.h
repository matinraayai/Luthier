//===-- CodeGenHelpers.h ----------------------------------------*- C++ -*-===//
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
/// \file This file includes a set of frequently-used helper functions regarding
/// LLVM's target independent code generator.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INCLUDE_LUTHIER_LLVMCODEGENHELPERS_H
#define LUTHIER_INCLUDE_LUTHIER_LLVMCODEGENHELPERS_H

namespace llvm {

class MachineInstr;

}

namespace luthier {

/// \return if \p MI is a scalar instruction
bool isScalar(const llvm::MachineInstr &MI);

/// \return \c true if \p MI is a lane access instruction (e.g. V_READLANE_B32),
/// \c false otherwise
bool isLaneAccess(const llvm::MachineInstr &MI);

/// \return \c true if \p MI is vector instruction (i.e. not a scalar or a
/// lane access instruction), \c false otherwise
bool isVector(const llvm::MachineInstr &MI);

} // namespace luthier

#endif
