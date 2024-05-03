#ifndef HSA_PLATFORM_HPP
#define HSA_PLATFORM_HPP
#include "hsa.hpp"
#include "hsa_executable_symbol.hpp"
#include <llvm/ADT/DenseMap.h>
#include <luthier/kernel_descriptor.h>

namespace luthier::hsa {

/**
 * \brief in charge of caching useful information about the HSA runtime state
 */
class Platform {

public:
  Platform(const Platform &) = delete;
  Platform &operator=(const Platform &) = delete;

  static inline Platform &instance() {
    static Platform Instance;
    return Instance;
  }

  llvm::Error registerFrozenExecutable(hsa_executable_t Exec);

  llvm::Error unregisterFrozenExecutable(hsa_executable_t Exec);

private:
  Platform() = default;
  ~Platform() = default;

  llvm::DenseSet<decltype(hsa_executable_t::handle)> FrozenExecs{};

  llvm::DenseMap<decltype(hsa_executable_t::handle),
                 llvm::DenseSet<decltype(hsa_executable_symbol_t::handle)>>
      AgentSymbolsOfFrozenExecs{};


  llvm::DenseMap<decltype(hsa_executable_t::handle),
                 llvm::DenseSet<decltype(hsa_executable_symbol_t::handle)>>
      ProgramSymbolsOfFrozenExecs{};

  llvm::DenseMap<decltype(hsa_executable_symbol_t::handle),
                 hsa_loaded_code_object_t>
      LCOofSymbols{};

  /**
   * \brief Keeps track of the Indirect functions
   * encountered so far, in order to expose them to the tool writer
   * seamlessly as an \b hsa_executable_symbol_t
   */
//  llvm::DenseMap<decltype(hsa_executable_symbol_t::handle), DeviceFunctionInfo>
//      DeviceFunctionHandles{};
  // TODO: Invalidate this cache

  llvm::DenseMap<KernelDescriptor *, hsa_executable_symbol_t> KDToSymbolMap{};
};

} // namespace luthier::hsa

#endif