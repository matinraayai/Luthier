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
  /// Constructor for explicit initialization of the Singleton instance \n
  /// Instead of hiding initialization away in the \c instance() method,
  /// this design allows passing additional arguments to the constructor
  /// of Singleton if required
  /// The constructor is \b not thread-safe, and is meant to be allocated
  /// on the heap with the \c new operator for better control over its lifetime
  Singleton() {
    if (Instance != nullptr) {
      llvm::report_fatal_error("Called the Singleton constructor twice.");
    }
    Instance = static_cast<T *>(this);
  }

  /// Destructor for explicit initialization of the Singleton Instance \n
  /// The destructor is \b not thread-safe, and is meant to be used directly
  /// with the \p delete operator for better control over its lifetime
  ~Singleton() { Instance = nullptr; }

  /// Disallowed copy construction
  Singleton(const Singleton &) = delete;

  /// Disallowed assignment operation
  Singleton &operator=(const Singleton &) = delete;

  /// \return a reference to the Singleton instance
  static inline T &instance() {
    if (Instance == nullptr)
      llvm::report_fatal_error("Singleton is not initialized");
    return *Instance;
  }
};

#ifdef __clang__
// Template definition of the Instance pointer to suppress clang warnings
// regarding translation untis
template <typename T> T *luthier::Singleton<T>::Instance{nullptr};
#endif

} // namespace luthier

#endif