//===-- LoadedCodeObjectExternSymbol.h - LCO External Symbol ----*- C++ -*-===//
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
/// This file defines the \c LoadedCodeObjectExternSymbol under the
/// \c luthier::hsa namespace, which represents all symbols declared
/// inside a \c hsa::LoadedCodeObject but not defined.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_EXTERN_SYMBOL_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_EXTERN_SYMBOL_H
#include "LoadedCodeObjectSymbol.h"

namespace luthier::hsa {

/// \brief an external variable \c LoadedCodeObjectSymbol
class LoadedCodeObjectExternSymbol
    : public llvm::RTTIExtends<LoadedCodeObjectExternSymbol,
                               LoadedCodeObjectSymbol> {
public:
  static char ID;

};

} // namespace luthier::hsa

#endif