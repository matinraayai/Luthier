#ifndef HSA_PLATFORM_HPP
#define HSA_PLATFORM_HPP
#include <llvm/ADT/DenseMap.h>
#include <llvm/Object/ELFObjectFile.h>
#include <luthier/kernel_descriptor.h>

#include <mutex>

namespace luthier::hsa {

class Platform;

class Executable;

class ExecutableSymbol;

class LoadedCodeObject;

/**
 * \brief an interface implemented by any item or attribute related to an
 * \b Executable.
 * Instances of classes that implement this interface will have their values
 * persist at some point of time during the lifetime of the executable,
 * and become stale once the executable is destroyed.
 * The luthier internal HSA callback \b luthier::hsa::internalApiCallback
 * will notify the \b Platform at appropriate events for caching instance
 * (or instances) of this interface an executable has been frozen or before
 * an executable is about to be destroyed. \b Platform will then either cache
 * all the items related to the executable that implement this interface or
 * invalidates them. The \b Executable class will also notify the \b Platform
 * on freezing and destruction of any executables by Luthier itself.
 *
 * Each class of this interface is in charge of maintaining a cache. Since
 * their state is tied to a single event (freezing and destruction of an
 * executable) they must be updated at the same time by the \b Platform
 * For thread safety, a recursive mutex is shared among all instances of this
 * class.
 *
 * As all executables dealt by Luthier should (at least in theory) be captured
 * by this mechanism; Therefore, any instances of this interface can return an
 * \b llvm::Error if a functionality that relies on cached information cannot
 * find the required information in the cache.
 */
class ExecutableBackedCachable {
  friend Platform;

private:
  static std::recursive_mutex CacheMutex;
  virtual llvm::Error cache() const = 0;

  [[nodiscard]] virtual bool isCached() const = 0;

  virtual llvm::Error invalidate() const = 0;

protected:
  static std::recursive_mutex &getCacheMutex() { return CacheMutex; }
};

/**
 * \brief Singleton in charge of caching useful information about the HSA
 * runtime state
 * The HSA API internal callback is used to inform this class about changes in
 * the application state
 */
class Platform {

public:
  Platform(const Platform &) = delete;
  Platform &operator=(const Platform &) = delete;

  static inline Platform &instance() {
    static Platform Instance;
    return Instance;
  }

  /**
   * An HSA event handler that records (caches) all
   * \c hsa::LoadedCodeObject's of \p Exec, if they are not already cached.
   * \param Exec
   * \return
   */
  llvm::Error cacheExecutableOnLoadedCodeObjectCreation(const Executable &Exec);

  llvm::Error cacheExecutableOnExecutableFreeze(const Executable &Exec);

  llvm::Error invalidateExecutableOnExecutableDestroy(const Executable &Exec);

  llvm::Expected<std::optional<hsa::ExecutableSymbol>>
  getSymbolFromLoadedAddress(luthier::address_t Address);

private:
  Platform() = default;
  ~Platform() = default;

  llvm::DenseMap<luthier::address_t, hsa_executable_symbol_t>
      AddressToSymbolMap{};
};

} // namespace luthier::hsa

#endif