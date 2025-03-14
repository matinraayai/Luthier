//===-- LoadedCodeObjectSymbol.cpp - Loaded Code Object Symbol ------------===//
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
/// This file implements the concrete portions of the
/// \c hsa::LoadedCodeObjectSymbol interface.
//===----------------------------------------------------------------------===//
#include <luthier/hsa/LoadedCodeObjectSymbol.h>

namespace luthier::hsa {

char LoadedCodeObjectSymbol::ID = 0;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void hsa::LoadedCodeObjectSymbol::dump() const {
  print(llvm::dbgs());
}
#endif

} // namespace luthier::hsa
