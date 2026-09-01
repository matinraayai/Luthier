//===-- IntrinsicProcessorRegistry.h ----------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file IntrinsicProcessorRegistry.h
/// Describes Luthier's Intrinsic Processor registry.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INTRINSIC_PROCESSOR_REGISTRY_H
#define LUTHIER_TOOL_CODE_GEN_INTRINSIC_PROCESSOR_REGISTRY_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"

#include <llvm/ADT/StringMap.h>
#include <mutex>
#include <shared_mutex>

namespace luthier {

class IntrinsicProcessorRegistry final {
private:
  mutable std::shared_mutex Mutex;

  llvm::StringMap<IntrinsicProcessor> Processors;

public:
  IntrinsicProcessorRegistry();

  [[nodiscard]] bool
  isIntrinsicProcessorRegistered(llvm::StringRef Name) const {
    std::shared_lock Lock(Mutex);
    return Processors.contains(Name);
  }

  void registerIntrinsicProcessor(llvm::StringRef Name,
                                  IntrinsicProcessor Processor) {
    std::unique_lock Lock(Mutex);
    (void)Processors.insert({Name, std::move(Processor)});
  }

  void unregisterIntrinsicProcessor(llvm::StringRef Name) {
    std::unique_lock Lock(Mutex);
    (void)Processors.erase(Name);
  }

  [[nodiscard]] std::optional<IntrinsicProcessor>
  getIntrinsicProcessorIfRegistered(llvm::StringRef Name) const {
    std::shared_lock Lock(Mutex);
    if (const auto It = Processors.find(Name); It != Processors.end()) {
      return It->second;
    }
    return std::nullopt;
  }
};

/// \brief Per-tool trait that owns the \c IntrinsicProcessorRegistry the tool's
/// instrumentation pipeline consults during IR/MIR intrinsic lowering. The
/// registry default-constructs (its ctor auto-registers the built-in Luthier
/// intrinsics from \c IntrinsicRegistry.def), so the trait needs no
/// constructor arguments and never fails.
template <typename Derived> class IntrinsicProcessorRegistryTraitBase {
  IntrinsicProcessorRegistry Registry;

public:
  IntrinsicProcessorRegistry &getIntrinsicProcessorRegistry() {
    return Registry;
  }

  const IntrinsicProcessorRegistry &getIntrinsicProcessorRegistry() const {
    return Registry;
  }
};

} // namespace luthier

#endif