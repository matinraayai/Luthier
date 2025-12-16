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
#include "luthier/HSA/CodeObjectReader.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/HsaError.h"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Expected<hsa_code_object_reader_t> codeObjectReaderCreateFromMemory(
    const ApiTableContainer<::CoreApiTable> &CoreApi, llvm::StringRef Elf) {
  hsa_code_object_reader_t Reader;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_code_object_reader_create_from_memory>(
          Elf.data(), Elf.size(), &Reader),
      "Failed to create code object reader from memory"));
  return Reader;
}

llvm::Error
codeObjectReaderDestroy(hsa_code_object_reader_t COR,
                        const ApiTableContainer<::CoreApiTable> &CoreApi) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_code_object_reader_destroy>(COR),
      "Failed to destroy the code object reader"));
  return llvm::Error::success();
}

} // namespace luthier::hsa