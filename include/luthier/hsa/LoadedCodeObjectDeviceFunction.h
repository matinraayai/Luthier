//===-- LoadedCodeObjectDeviceFunction.h - LCO Device Function --*- C++ -*-===//
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
/// This file defines the \c LoadedCodeObjectDeviceFunction interface under the
/// \c luthier::hsa namespace, which represents all device (non-kernel)
/// functions inside a \c hsa::LoadedCodeObject.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LOADED_CODE_OBJECT_DEVICE_FUNCTION_H
#define LUTHIER_LOADED_CODE_OBJECT_DEVICE_FUNCTION_H
#include "LoadedCodeObjectSymbol.h"

namespace luthier::hsa {

class LoadedCodeObjectDeviceFunction
    : public llvm::RTTIExtends<LoadedCodeObjectDeviceFunction,
                               LoadedCodeObjectSymbol> {
public:
  static char ID;
};

} // namespace luthier::hsa

#endif