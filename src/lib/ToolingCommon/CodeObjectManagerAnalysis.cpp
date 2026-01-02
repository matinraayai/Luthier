//===-- CodeObjectManagerAnalysis.cpp -------------------------------------===//
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
/// \file
/// Implements the \c CodeObjectManagerAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/CodeObjectManagerAnalysis.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include <llvm/Object/Binary.h>

namespace luthier {

llvm::Expected<llvm::MemoryBuffer &>
CodeObjectManagerAnalysis::Result::readCodeObjectFromFile(
    llvm::StringRef Path) {
  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>> BinaryOrErr =
      llvm::object::createBinary(Path);
  LUTHIER_RETURN_ON_ERROR(BinaryOrErr.takeError());

  const llvm::object::Binary *Bin = BinaryOrErr->getBinary();

  if (auto *AMDGCNObj =
          llvm::dyn_cast_if_present<object::AMDGCNObjectFile>(&Bin)) {
    if (AMDGCNObj->isRelocatableObject())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Path {0} is a relocatable object, not a shared object.", Path));
    auto [OwningObjFile, OwningBuffer] = BinaryOrErr->takeBinary();
    const std::unique_ptr<llvm::MemoryBuffer> &BuffInVector =
        CodeObjects.emplace_back(std::move(OwningBuffer));
    return *BuffInVector;
  }
  return LUTHIER_MAKE_GENERIC_ERROR(
      llvm::formatv("Path {0} is not an AMDGCN object file", Path));
}

llvm::Expected<llvm::MemoryBuffer &>
CodeObjectManagerAnalysis::Result::takeOwnershipOfCodeObject(
    std::unique_ptr<llvm::MemoryBuffer> CodeObject) {
  llvm::Expected<std::unique_ptr<llvm::object::Binary>> BinaryOrErr =
      llvm::object::createBinary(*CodeObject);
  LUTHIER_RETURN_ON_ERROR(BinaryOrErr.takeError());

  if (auto *AMDGCNObj = llvm::dyn_cast_if_present<object::AMDGCNObjectFile>(
          BinaryOrErr->get())) {
    if (AMDGCNObj->isRelocatableObject())
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Code object is a relocatable object, not a shared object.");

    const std::unique_ptr<llvm::MemoryBuffer> &BuffInVector =
        CodeObjects.emplace_back(std::move(CodeObject));
    return *BuffInVector;
  }
  return LUTHIER_MAKE_GENERIC_ERROR("Code object is not an AMDGCN object file");
}
} // namespace luthier