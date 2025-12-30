//===-- IntrinsicProcessorRegistry.h ----------------------------*- C++ -*-===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// Describes Luthier's Intrinsic Processor registry singleton and its
/// associated analysis pass.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INTRINSIC_PROCESSOR_REGISTRY_H
#define LUTHIER_TOOLING_INTRINSIC_PROCESSOR_REGISTRY_H
#include "luthier/Common/Singleton.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/ADT/StringMap.h>

namespace luthier {

class IntrinsicProcessorRegistry final
    : public Singleton<IntrinsicProcessorRegistry> {
private:
  llvm::StringMap<IntrinsicProcessor> Processors;

public:
  IntrinsicProcessorRegistry();

  [[nodiscard]] bool
  isIntrinsicProcessorRegistered(llvm::StringRef Name) const {
    return Processors.contains(Name);
  }

  void registerIntrinsicProcessor(llvm::StringRef Name,
                                  IntrinsicProcessor Processor) {
    Processors.insert({Name, std::move(Processor)});
  }

  void unregisterIntrinsicProcessor(llvm::StringRef Name) {
    Processors.erase(Name);
  }

  [[nodiscard]] IntrinsicProcessor
  getIntrinsicProcessor(llvm::StringRef Name) const {
    return Processors.at(Name);
  }
};

} // namespace luthier

#endif