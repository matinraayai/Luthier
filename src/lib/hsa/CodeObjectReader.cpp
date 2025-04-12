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
/// This file implements the \c hsa::CodeObjectReader class.
//===----------------------------------------------------------------------===//
#include "hsa/CodeObjectReader.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "luthier/hsa/HsaError.h"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Expected<CodeObjectReader>
CodeObjectReader::createFromMemory(llvm::StringRef CodeObject) {
  const auto &CoreApiTable =
      hsa::HsaRuntimeInterceptor::instance().getSavedApiTableContainer().core;
  hsa_code_object_reader_t Reader;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      CoreApiTable.hsa_code_object_reader_create_from_memory_fn(
          CodeObject.data(), CodeObject.size(), &Reader)));
  return CodeObjectReader{Reader};
}

llvm::Expected<CodeObjectReader>
CodeObjectReader::createFromMemory(llvm::ArrayRef<uint8_t> Elf) {
  return createFromMemory(llvm::toStringRef(Elf));
}

llvm::Error CodeObjectReader::destroy() {
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_code_object_reader_destroy_fn(asHsaType()));
}

} // namespace luthier::hsa
