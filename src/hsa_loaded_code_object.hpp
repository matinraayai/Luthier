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
 * \brief Wraps functionality related to \c hsa_loaded_code_object_t in Luthier
 *
 * \note This wrapper is implemented with the following assumptions based on
 * the current state of ROCr:
 * 1. Even though the ROCr HSA vendor loader API
 * (under <hsa/hsa_ven_amd_loader.h>) acknowledge both file-backed and
 * memory-backed Loaded Code Objects exists, only memory-backed ones are
 * actually implemented. Therefore, querying the storage type and querying the
 * FD of the storage is not included. Luthier assumes all Loaded Code Objects
 * have a memory storage in order to return its associated ELF.
 * In the event that file-backed storage is implemented in the loader, the
 * code needs to be updated.
 * 2. Program Loaded Code Objects has been deprecated and are not used anywhere
 * in the ROCm stack. ROCr does not even allow using Loaded Code Objects with
 * program allocations. Therefore, it is safe to assume all Loaded Code Objects
 * are backed by a \c GpuAgent.
 * 3. Internally, the vendor loader API keeps track of loaded segments, and
 * allows for querying these segments via the
 * \c hsa_ven_amd_loader_query_segment_descriptors function. As of right now,
 * Luthier does not use this information to locate the load address of the
 * symbols, and instead relies on the \c luthier::getSymbolLMA and
 * \c luthier::getSectionLMA functions to calculate the load address.
 *
 * \note This wrapper relies on cached functionality as described by the
 * \c hsa::ExecutableBackedCachable interface and backed by the \c hsa::Platform
 * Singleton.
 */
class LoadedCodeObject : public ExecutableBackedCachable,
                         public HandleType<hsa_loaded_code_object_t> {
  /*****************************************************************************
   * Public-facing methods
   ****************************************************************************/
public:
  /**
   * Primary constructor for Loaded Code Objects.
   * \param LCO HSA handle of the Loaded Code Object
   */
  explicit LoadedCodeObject(hsa_loaded_code_object_t LCO);

  /**
   * Queries the \c Executable associated with this \c LoadedCodeObject.
   * \return the \c Executable of this \c LoadedCodeObject, or an \c llvm::Error
   * reporting any HSA errors occurred
   * \note Performs an HSA call to complete this operation
   * \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE
   */
  [[nodiscard]] llvm::Expected<Executable> getExecutable() const;

  /**
   * Queries the \c GpuAgent associated with this \c LoadedCodeObject
   * \return the \c GpuAgent of this \c LoadedCodeObject, or an \c llvm::Error
   * reporting any HSA errors occurred during this operation
   * \note As Loaded Code Objects of program allocation are deprecated in ROCr,
   * it is safe to assume all Loaded Code Objects have agent allocation, and
   * therefore, are backed by an HSA Agent
   * \note performs an HSA call to complete this operation
   * \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT
   */
  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  /**
   * Returns a reference to the \c luthier::AMDGCNObjectFile of
   * the ELF associated with this \c LoadedCodeObject
   * The ELF is obtained by parsing the Loaded Code Objects'
   * Storage memory.
   * \note This operation relies on cached information
   * \return the \c luthier::AMDGCNObjectFile of this \c LoadedCodeObject, or
   * an \c llvm::Error if this \c LoadedCodeObject has not been cached
   * \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
   * HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE
   */
  [[nodiscard]] llvm::Expected<luthier::AMDGCNObjectFile &>
  getStorageELF() const;

  /**
   * \return the Load Delta of this Loaded Code Object, or an \c llvm::Error
   * indicating an HSA error
   * \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA
   * \note performs an HSA call to complete this operation
   */
  [[nodiscard]] llvm::Expected<long> getLoadDelta() const;

  /**
   * \return an \c llvm::ArrayRef to the portion of GPU memory that
   * this code object has been loaded onto, \c llvm::Error
   * indicating an HSA error
   * \note performs an HSA call to complete this operation
   * \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
   * HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE
   */
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getLoadedMemory() const;

  /**
   * \return The URI describing the origins of this \c LoadedCodeObject
   * \note performs an HSA call to complete this operation
   * \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
   * HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI
   */
  [[nodiscard]] llvm::Expected<std::string> getUri() const;

  [[nodiscard]] llvm::Expected<ISA> getISA() const;

  [[nodiscard]] llvm::Expected<const hsa::md::Metadata &> getMetadata() const;

  /**
   * Appends all the \c hsa::ExecutableSymbols in this LCO that are of type
   * \c hsa::KERNEL to \c Out
   * \param Out [out] an \c llvm::SmallVectorImpl pointing to a sequential
   * container that will contain the symbols returned by the operation
   * \return \c llvm::Error describing whether the operation has succeeded or
   * not
   * \note this operation relies on cached information
   */
  [[nodiscard]] llvm::Error
  getKernelSymbols(llvm::SmallVectorImpl<hsa::ExecutableSymbol> &Out) const;

  /**
   * Appends all the \c hsa::ExecutableSymbols in this LCO that are of type
   * \c hsa::VARIABLE to \c Out
   * \param Out [out] an \c llvm::SmallVectorImpl pointing to a sequential
   * container that will contain the symbols returned by the operation
   * \return \c llvm::Error describing whether the operation has succeeded or
   * not
   * \note this operation relies on cached information
   */
  [[nodiscard]] llvm::Error
  getVariableSymbols(llvm::SmallVectorImpl<hsa::ExecutableSymbol> &Out) const;

  /**
   * Appends all the \c hsa::ExecutableSymbols in this LCO that are of type
   * \c hsa::DEVICE_FUNCTION to \c Out
   * \param Out [out] an \c llvm::SmallVectorImpl pointing to a sequential
   * container that will contain the symbols returned by the operation
   * \return \c llvm::Error describing whether the operation has succeeded or
   * not
   * \note this operation requires the LCO to be cached.
   */
  [[nodiscard]] llvm::Error getDeviceFunctionSymbols(
      llvm::SmallVectorImpl<hsa::ExecutableSymbol> &Out) const;

  /**
   *
   * @return
   */
  [[nodiscard]] llvm::Error
  getExecutableSymbols(llvm::SmallVectorImpl<ExecutableSymbol> &Out) const;

  [[nodiscard]] llvm::Expected<std::optional<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name) const;

  /*****************************************************************************
   * Private functionality specific to \c LoadedCodeObject
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

  /**
   * Looks up the given \p Name in the internal kernel symbol information cache
   * and constructs an \c hsa::ExecutableSymbol for it
   * \param NameWithKDAtTheEnd Name of the kernel symbol, with ".kd" at the end
   * \return on success, an \c hsa::ExecutableSymbol associated with the given
   * kernel name, or an \c llvm::Error describing the failure
   */
  [[nodiscard]] llvm::Expected<hsa::ExecutableSymbol>
  constructKernelSymbolUsingName(llvm::StringRef NameWithKDAtTheEnd) const;

  /**
   * Looks up the given \p Name in the internal variable symbol information
   * cache and constructs an \c hsa::ExecutableSymbol for it
   * \param Name Name of the variable symbol
   * \return on success, an \c hsa::ExecutableSymbol associated with the given
   * \c Name, or an \c llvm::Error describing the failure. If the variable
   * symbol is external to this LCO and HSA returns no handles for the symbol,
   * then it returns \c std::nullopt
   */
  [[nodiscard]] llvm::Expected<std::optional<hsa::ExecutableSymbol>>
  constructVariableSymbolUsingName(llvm::StringRef Name) const;

  /**
   * Looks up the given \p Name in the internal device function symbol
   * information cache and constructs an \c hsa::ExecutableSymbol for it
   * \param Name Name of the device function symbol
   * \return on success, an \c hsa::ExecutableSymbol associated with the given
   * \c Name, or an \c llvm::Error describing the failure
   */
  [[nodiscard]] llvm::Expected<hsa::ExecutableSymbol>
  constructDeviceFunctionSymbolUsingName(llvm::StringRef Name) const;

  /*****************************************************************************
   * Implementation of \c hsa::ExecutableBackedCachable
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
      KernelDescSymbolsOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<object::ELFSymbolRef>>
      KernelFuncSymbolsOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<object::ELFSymbolRef>>
      DeviceFuncSymbolsOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<object::ELFSymbolRef>>
      VariableSymbolsOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        hsa::md::Metadata>
      MetadataOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        llvm::StringMap<hsa::md::Kernel::Metadata *>>
      KernelSymbolsMetadataOfLCOs;

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