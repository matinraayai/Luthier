#ifndef HSA_LOADED_CODE_OBJECT_HPP
#define HSA_LOADED_CODE_OBJECT_HPP
#include "hsa_handle_type.hpp"
#include "hsa_platform.hpp"
#include "object_utils.hpp"
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Error.h>

namespace object = llvm::object;

namespace luthier::hsa {

class GpuAgent;

class ExecutableSymbol;

class ISA;

/**
 * Notes:
 * 1. Even though the API mentions both file-backed and
 * memory-backed Loaded Code Objects exists, only memory-backed ones are
 * actually implemented. Therefore, querying the storage type and querying the
 * FD of the storage has been removed.
 */

class LoadedCodeObject : public ExecutableBackedCachable,
                         public HandleType<hsa_loaded_code_object_t> {
  /*****************************************************************************
   * \brief Public-facing methods
   ****************************************************************************/
public:
  explicit LoadedCodeObject(hsa_loaded_code_object_t LCO);

  /**
   * Queries the \b Executable associated with this \b LoadedCodeObject.
   * Performs an HSA call to complete this operation
   * \return the \b Executable of this \b LoadedCodeObject, or an \b llvm::Error
   * reporting any HSA errors occurred during this operation
   */
  [[nodiscard]] llvm::Expected<Executable> getExecutable() const;

  /**
   * Queries the \b GpuAgent associated with this \b LoadedCodeObject.
   * Performs an HSA call to complete this operation
   * \return the \b GpuAgent of this \b LoadedCodeObject, or an \b llvm::Error
   * reporting any HSA errors occurred during this operation
   */
  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  /**
   * Returns the \b luthier::AMDGCNObjectFile of this \b LoadedCodeObject,
   * obtained by parsing its Storage memory
   * This operation relies on the object being cached beforehand
   * \return the \b luthier::AMDGCNObjectFile of this \b LoadedCodeObject, or
   * an \b llvm::Error if this \b LoadedCodeObject has not been cached properly
   */
  [[nodiscard]] llvm::Expected<luthier::AMDGCNObjectFile &>
  getStorageELF() const;

  [[nodiscard]] llvm::Expected<long> getLoadDelta() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getLoadedMemory() const;

  [[nodiscard]] llvm::Expected<std::string> getUri() const;

  [[nodiscard]] llvm::Expected<ISA> getISA() const;

  /**
   *
   * @return
   */
  [[nodiscard]] llvm::Expected<std::vector<ExecutableSymbol>>
  getExecutableSymbols() const;

  [[nodiscard]] llvm::Expected<std::optional<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name) const;

  /*****************************************************************************
   * \brief Private functionality specific to \b LoadedCodeObject only
   ****************************************************************************/
private:
  /**
   * Queries where the ELF of this \b LoadedCodeObject is stored, and its size
   * from HSA
   * This function is not public, since the primary reason for querying this
   * is to inspect the ELF of the \b LoadedCodeObject, which is exposed through
   * \b getStorageELF. The storage memory can be obtained externally from the
   * \b luthier::AMDGCNObjectFile if needed.
   * \return An \b llvm::ArrayRef pointing to the beginning and the end of the
   * storage memory on success, or an \b llvm::Error reporting any HSA errors
   * encountered during this operation
   */
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getStorageMemory() const;

  /**
   * Tries to find the \b hsa_executable_symbol_t of a symbol given its name
   * It queries the \b hsa_executable_t of this LCO using the HSA API
   * \b hsa_executable_get_symbol_by_name
   * Primarily used to find a connection between the
   * \b llvm::object::ELFSymbolRef found by inspecting the ELF of the storage
   * memory
   * \param Name Name of the symbol
   * \return if the symbol is found, an \b hsa_executable_symbol_t. If not,
   * an \b std::nullopt. If an error is occurred, an \b llvm::Error of
   * appropriate kind.
   */
  [[nodiscard]] llvm::Expected<std::optional<hsa_executable_symbol_t>>
  getHSASymbolHandleByNameFromExecutable(llvm::StringRef Name) const;

  /*****************************************************************************
   * \brief Implementation of \b hsa::ExecutableBackedCachable
   ****************************************************************************/
private:
  static llvm::DenseSet<decltype(hsa_loaded_code_object_t::handle)> CachedLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        std::unique_ptr<luthier::AMDGCNObjectFile>>
      StorageELFOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle), hsa_isa_t>
      ISAOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<object::ELFSymbolRef>>
      KernelDescSymbols;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<object::ELFSymbolRef>>
      KernelFuncSymbols;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<object::ELFSymbolRef>>
      DeviceFuncSymbols;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<object::ELFSymbolRef>>
      VariableSymbols;

  llvm::Error cache() const override;

  bool isCached() const override;

  llvm::Error invalidate() const override;
};

} // namespace luthier::hsa

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::LoadedCodeObject> {
  static inline luthier::hsa::LoadedCodeObject getEmptyKey() {
    return luthier::hsa::LoadedCodeObject({DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::LoadedCodeObject getTombstoneKey() {
    return luthier::hsa::LoadedCodeObject({DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::LoadedCodeObject &ISA) {
    return DenseMapInfo<decltype(hsa_loaded_code_object_t::handle)>::
        getHashValue(ISA.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::LoadedCodeObject &lhs,
                      const luthier::hsa::LoadedCodeObject &rhs) {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace llvm

namespace std {

template <> struct hash<luthier::hsa::LoadedCodeObject> {
  size_t operator()(const luthier::hsa::LoadedCodeObject &obj) const {
    return hash<unsigned long>()(obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &lhs,
                  const luthier::hsa::LoadedCodeObject &rhs) const {
    return lhs.hsaHandle() < rhs.hsaHandle();
  }
};

template <> struct less_equal<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &lhs,
                  const luthier::hsa::LoadedCodeObject &rhs) const {
    return lhs.hsaHandle() <= rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &Lhs,
                  const luthier::hsa::LoadedCodeObject &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

template <> struct not_equal_to<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &lhs,
                  const luthier::hsa::LoadedCodeObject &rhs) const {
    return lhs.hsaHandle() != rhs.hsaHandle();
  }
};

template <> struct greater<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &lhs,
                  const luthier::hsa::LoadedCodeObject &rhs) const {
    return lhs.hsaHandle() > rhs.hsaHandle();
  }
};

template <> struct greater_equal<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &lhs,
                  const luthier::hsa::LoadedCodeObject &rhs) const {
    return lhs.hsaHandle() >= rhs.hsaHandle();
  }
};

} // namespace std

#endif