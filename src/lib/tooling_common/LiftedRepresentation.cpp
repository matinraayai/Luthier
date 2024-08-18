//===-- LiftedRepresentation.cpp - Lifted Representation  -----------------===//
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
/// This file implements Luthier's Lifted Representation, which contains the
/// \c llvm::Module and \c llvm::MachineModuleInfo of a lifted HSA primitive (a
/// kernel or an executable), as well as a mapping between the HSA primitives
/// and LLVM IR primitives involved.
//===----------------------------------------------------------------------===//
#include <luthier/LiftedRepresentation.h>
#include <luthier/hsa/LoadedCodeObjectDeviceFunction.h>
#include <luthier/hsa/LoadedCodeObjectExternSymbol.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include <luthier/hsa/LoadedCodeObjectVariable.h>

namespace luthier {

llvm::ArrayRef<llvm::MachineInstr *> LiftedRepresentation::getUsesOfGlobalValue(
    const hsa::LoadedCodeObjectSymbol &GV) const {
  if (llvm::isa<hsa::LoadedCodeObjectVariable>(GV) ||
      llvm::isa<hsa::LoadedCodeObjectExternSymbol>(GV))
    return getUsesOfGlobalValue(*RelatedGlobalVariables.at(&GV));
  else if (llvm::isa<hsa::LoadedCodeObjectKernel>(GV) ||
           llvm::isa<hsa::LoadedCodeObjectDeviceFunction>(GV))
    return getUsesOfGlobalValue(RelatedFunctions.at(&GV)->getFunction());
  else
    return {};
}

LiftedRepresentation::LiftedRepresentation() = default;
} // namespace luthier