//===-- singleton.hpp - Singleton Interface -------------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the interface inherited by all Singleton objects in
/// Luthier.
/// It was inspired by OGRE's Singleton implementation here:
/// https://github.com/OGRECave/ogre/blob/master/OgreMain/include/OgreSingleton.h
//===----------------------------------------------------------------------===//
#ifndef SINGLETON_HPP
#define SINGLETON_HPP
#include <llvm/Support/ErrorHandling.h>

namespace luthier {

///
/// \tparam T
template <typename T> class Singleton {
private:
  static T *Instance;

public:
  Singleton() {
    if (Instance != nullptr) {
      llvm::report_fatal_error("Called the singleton constructor twice.");
    }
    Instance = static_cast<T *>(this);
  }
  ~Singleton() { Instance = nullptr; }

  Singleton(const Singleton &) = delete;

  Singleton &operator=(const Singleton &) = delete;

  static inline T &instance() {
    if (Instance == nullptr) {
      llvm::report_fatal_error("Singleton is not initialized.");
    }
    return *Instance;
  }
};

#ifdef __clang__
// template definition of the Instance pointer to suppress clang warnings
template <typename T> T *luthier::Singleton<T>::Instance{nullptr};
#endif

} // namespace luthier

#endif