//===-- IObjectCache.hpp - Object Cache Interface -------------------------===//
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
// See the License for the specific languae governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file This file defines the \c IObjectCache interface, which can be used
/// to cache object files that are inspected by Luthier at runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_OBJECT_CACHE_HPP
#define LUTHIER_COMMON_OBJECT_CACHE_HPP
#include "common/ObjectUtils.hpp"
#include "common/Singleton.hpp"
#include "luthier/hsa/DenseMapInfo.h"
#include "luthier/hsa/LoadedCodeObjectDeviceFunction.h"
#include "luthier/hsa/LoadedCodeObjectExternSymbol.h"
#include "luthier/hsa/LoadedCodeObjectKernel.h"
#include "luthier/hsa/LoadedCodeObjectVariable.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringMap.h>
#include <mutex>
#include <shared_mutex>

namespace luthier {

/// \brief Interface in charge of retaining a copy of object files inspected
/// by the tooling library and their parsed \c llvm::object::ObjectFile
/// representations
/// Primarily used by the \c CodeLifter so that it can cache its disassembled
/// and lifted representations
class IObjectCache {
public:
  typedef std::function<void(const llvm::object::ObjectFile &)>
      object_invalidation_callback_t;

  /// Adds a callback function to be invoked when a cached object file is
  /// about to get destroyed inside the cache
  /// Any component inside the Luthier tooling library can add a callback so
  /// that they can be notified when a cached code object is about to be
  /// destroyed
  virtual void
  addInvalidationCallback(const object_invalidation_callback_t &CB) = 0;

  /// \return \c true if \p ObjFile is stored inside the cache, \c false
  /// otherwise
  [[nodiscard]] virtual bool
  isObjectFileCached(const llvm::object::ObjectFile &ObjFile) const = 0;
};


} // namespace luthier

#endif