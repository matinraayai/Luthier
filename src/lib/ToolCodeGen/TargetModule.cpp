//===-- TargetModule.cpp --------------------------------------------------===//
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
/// \file
/// Implements \c TargetModule.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetModule.h"
#include <cassert>
#include <llvm/IR/LLVMContext.h>

namespace luthier {

TargetModule::TargetModule(llvm::StringRef ModuleID, llvm::LLVMContext &C,
                           Prototype &Parent)
    : M(std::make_unique<llvm::Module>(ModuleID, C)), Parent(Parent) {}

TargetModule::TargetModule(std::unique_ptr<llvm::Module> M, Prototype &Parent)
    : M(std::move(M)), Parent(Parent) {
  assert(this->M && "TargetModule cannot wrap a null llvm::Module");
}

} // namespace luthier
