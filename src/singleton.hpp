#ifndef SINGLETON_HPP
#define SINGLETON_HPP
#include <llvm/Support/ErrorHandling.h>
#include <shared_mutex>

namespace luthier {

template <typename T> class Singleton {

private:
  static T *Instance;

public:
  Singleton(const Singleton &) = delete;

  Singleton &operator=(const Singleton &) = delete;

  Singleton() {
    if (Instance != nullptr) {
      llvm::report_fatal_error("Called the singleton constructor twice.");
    }
    Instance = static_cast<T *>(this);
  }
  ~Singleton() { Instance = nullptr; }

  static inline T &instance() {
    if (Instance == nullptr) {
      llvm::report_fatal_error("Singleton is not initialized.");
    }
    return *Instance;
  }
};

template <typename T> T *luthier::Singleton<T>::Instance{nullptr};

} // namespace luthier

#endif