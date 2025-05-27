//===-- Singleton.h - Luthier Singleton Interface ---------------*- C++ -*-===//
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
/// This file contains the interface inherited by all Singleton objects in
/// Luthier.
/// It was inspired by OGRE's Singleton implementation here:
/// https://github.com/OGRECave/ogre/blob/master/OgreMain/include/OgreSingleton.h
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_SINGLETON_H
#define LUTHIER_COMMON_SINGLETON_H
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"

namespace luthier {

/// \brief Interface inherited by all Singleton objects in Luthier
/// \tparam T The concrete Singleton object itself
template <typename T> class Singleton {
private:
  static T *Instance;

public:
  /// Constructor for explicit initialization of the Singleton instance \n
  /// Instead of hiding initialization away in the \c instance() method,
  /// this design allows passing additional arguments to the constructor
  /// of Singleton if required
  /// The constructor is \b not thread-safe, and is meant to be allocated
  /// on the heap with the \c new operator for better control over its lifetime
  Singleton() {
    if (Instance != nullptr) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_CREATE_ERROR("Called the Singleton constructor twice."));
    }
    Instance = static_cast<T *>(this);
  }

  /// Destructor for explicit initialization of the Singleton Instance \n
  /// The destructor is \b not thread-safe, and is meant to be used directly
  /// with the \p delete operator for better control over its lifetime
  virtual ~Singleton() { Instance = nullptr; }

  /// Disallowed copy construction
  Singleton(const Singleton &) = delete;

  /// Disallowed assignment operation
  Singleton &operator=(const Singleton &) = delete;

  /// \return a reference to the Singleton instance
  static inline T &instance() {
    if (Instance == nullptr)
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_CREATE_ERROR("Singleton is not initialized"));
    return *Instance;
  }

  static inline bool isInitialized() { return Instance != nullptr; }
};

#ifdef __clang__
// Template definition of the Instance pointer to suppress clang warnings
// regarding translation units
template <typename T> T *luthier::Singleton<T>::Instance{nullptr};
#endif

} // namespace luthier

#endif