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

class LoadedCodeObject : public ExecutableBackedCachable,
                         public HandleType<hsa_loaded_code_object_t> {
public:
  explicit LoadedCodeObject(hsa_loaded_code_object_t LCO);

  [[nodiscard]] llvm::Expected<Executable> getExecutable() const;

  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  [[nodiscard]] llvm::Expected<hsa_ven_amd_loader_code_object_storage_type_t>
  getStorageType() const;

  [[nodiscard]] llvm::Expected<luthier::AMDGCNObjectFile &>
  getStorageELF() const;

  [[nodiscard]] llvm::Expected<std::vector<ExecutableSymbol>>
  getExecutableSymbols() const;

  [[nodiscard]] llvm::Expected<std::optional<ExecutableSymbol>>
  getExecutableSymbolByName(llvm::StringRef Name) const;

  [[nodiscard]] llvm::Expected<int> getStorageFile() const;

  [[nodiscard]] llvm::Expected<long> getLoadDelta() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getLoadedMemory() const;

  [[nodiscard]] llvm::Expected<std::string> getUri() const;

  [[nodiscard]] llvm::Expected<ISA> getISA() const;

private:
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getStorageMemory() const;

  [[nodiscard]] llvm::Expected<hsa_executable_symbol_t>
  getSymbolByNameFromExecutable(llvm::StringRef Name) const;

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