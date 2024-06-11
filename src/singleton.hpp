#ifndef SINGLETON_HPP
#define SINGLETON_HPP
#include <llvm/Support/ErrorHandling.h>

namespace luthier {

class GlobalSingletonManager;

template <typename T> class Singleton {
private:
  friend class GlobalSingletonManager;
  static T *Instance;

protected:
  Singleton() {
    if (Instance != nullptr) {
      llvm::report_fatal_error("Called the singleton constructor twice.");
    }
    Instance = static_cast<T *>(this);
  }
  ~Singleton() { Instance = nullptr; }

public:
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