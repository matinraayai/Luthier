//===-- CodeObjectReader.hpp ----------------------------------------------===//
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
/// This file implements the concrete portions of the \c hsa::CodeObjectReader
/// interface.
//===----------------------------------------------------------------------===//
#include "hsa/CodeObjectReader.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "luthier/hsa/HsaError.h"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Error CodeObjectReader::createFromMemory(llvm::ArrayRef<uint8_t> Elf) {
  return createFromMemory(llvm::toStringRef(Elf));
}

} // namespace luthier::hsa
