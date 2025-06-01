//===-- CodeObjectReader.cpp ----------------------------------------------===//
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
/// \file
/// Implements a set of commonly used functionality for the
/// \c hsa_code_object_reader_t handle in HSA.
//===----------------------------------------------------------------------===//
#include <llvm/ADT/StringExtras.h>
#include <luthier/hsa/CodeObjectReader.h>
#include <luthier/hsa/HsaError.h>

namespace luthier::hsa {

llvm::Expected<hsa_code_object_reader_t> codeObjectReaderCreateFromMemory(
    const decltype(hsa_code_object_reader_create_from_memory)
        &HsaCodeObjectReaderCreateFromMemoryFn,
    llvm::StringRef Elf) {
  hsa_code_object_reader_t Reader;
  if (hsa_status_t Status = HsaCodeObjectReaderCreateFromMemoryFn(
          Elf.data(), Elf.size(), &Reader);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        llvm::formatv("Failed to create a new code object reader handle from "
                      "memory for code object:\n {0}",
                      Elf),
        Status);
  }
  return Reader;
}

llvm::Error
codeObjectReaderDestroy(hsa_code_object_reader_t COR,
                        const decltype(hsa_code_object_reader_destroy)
                            &HsaCodeObjectReaderDestroyFn) {
  if (hsa_status_t Status = HsaCodeObjectReaderDestroyFn(COR);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        "Failed to destroy code object reader with handle {0:x}", COR.handle,
        Status);
  }
  return llvm::Error::success();
}

} // namespace luthier::hsa
