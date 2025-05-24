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
/// This file implements a set of commonly used functionality for the
/// \c hsa_code_object_reader_t handle in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/CodeObjectReader.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/HsaError.h"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Expected<hsa_code_object_reader_t> createCodeObjectReaderFromMemory(
    const decltype(hsa_code_object_reader_create_from_memory)
        *HsaCodeObjectReaderCreateFromMemoryFn,
    llvm::StringRef Elf) {
  hsa_code_object_reader_t Reader;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaCodeObjectReaderCreateFromMemoryFn(Elf.data(), Elf.size(), &Reader)));
  return Reader;
}

llvm::Error
destroyCodeObjectReader(hsa_code_object_reader_t COR,
                        const decltype(hsa_code_object_reader_destroy)
                            *HsaCodeObjectReaderDestroyFn) {
  return LUTHIER_HSA_SUCCESS_CHECK(HsaCodeObjectReaderDestroyFn(COR));
}

} // namespace luthier::hsa
