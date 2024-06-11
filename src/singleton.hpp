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
#include <mutex>

namespace luthier {

/// \brief Interface inherited by all Singleton objects in Luthier
/// \tparam T The concrete Singleton object itself
template <typename T> class Singleton {
private:
  static T *Instance;

public:
  Singleton() {
    static std::once_flag OnceFlag;
    std::call_once(OnceFlag, [&]() { Instance = static_cast<T *>(this); });
  }

  ///
  ~Singleton() {
    static std::once_flag OnceFlag;
    std::call_once(OnceFlag, [&]() { Instance = nullptr; });
  }

  /// Disallowed copy construction
  Singleton(const Singleton &) = delete;

  /// Disallowed assignment operation
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