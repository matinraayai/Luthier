#ifndef HSA_PLATFORM_HPP
#define HSA_PLATFORM_HPP
#include "hsa.hpp"
#include "hsa_executable_symbol.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/Object/ELFObjectFile.h>
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

  llvm::object::ELF64LEObjectFile &
  getStorgeELFofLCO(hsa_loaded_code_object_t LCO);

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

  llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                 std::unique_ptr<llvm::object::ELF64LEObjectFile>>
      StorageELFOfLCOs{};

  /**
   * \brief Keeps track of the Indirect functions
   * encountered so far, in order to expose them to the tool writer
   * seamlessly as an \b hsa_executable_symbol_t
   */
  //  llvm::DenseMap<decltype(hsa_executable_symbol_t::handle),
  //  DeviceFunctionInfo>
  //      DeviceFunctionHandles{};
  // TODO: Invalidate this cache

  llvm::DenseMap<luthier::address_t *, hsa_executable_symbol_t>
      DeviceAddressToSymbolMap{};
};

} // namespace luthier::hsa

#endif