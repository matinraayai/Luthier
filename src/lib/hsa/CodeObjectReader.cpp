//===-- CodeObjectReader.hpp - HSA Code Object Reader Wrapper -------------===//
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
/// This file implements the \c CodeObjectReader class under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//
#include "hsa/CodeObjectReader.hpp"
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/HsaError.h"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Expected<CodeObjectReader> CodeObjectReader::createFromMemory(
    const decltype(hsa_code_object_reader_create_from_memory)
        *HsaCodeObjectReaderCreateFromMemoryFn,
    llvm::StringRef Elf) {
  hsa_code_object_reader_t Reader;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaCodeObjectReaderCreateFromMemoryFn(Elf.data(), Elf.size(), &Reader)));
  return CodeObjectReader{Reader};
}

llvm::Expected<CodeObjectReader> CodeObjectReader::createFromMemory(
    const decltype(hsa_code_object_reader_create_from_memory)
        *HsaCodeObjectReaderCreateFromMemoryFn,
    llvm::ArrayRef<uint8_t> Elf) {
  return createFromMemory(HsaCodeObjectReaderCreateFromMemoryFn,
                          llvm::toStringRef(Elf));
}

llvm::Error
CodeObjectReader::destroy(const decltype(hsa_code_object_reader_destroy)
                              *HsaCodeObjectReaderDestroyFn) {
  return LUTHIER_HSA_SUCCESS_CHECK(HsaCodeObjectReaderDestroyFn(asHsaType()));
}

} // namespace luthier::hsa
