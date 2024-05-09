#ifndef HSA_LOADED_CODE_OBJECT_HPP
#define HSA_LOADED_CODE_OBJECT_HPP
#include "hsa_handle_type.hpp"
#include "hsa_platform.hpp"
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

class GpuAgent;

class ISA;

class LoadedCodeObject : public ExecutableBackedCachableItem,
                         public HandleType<hsa_loaded_code_object_t> {
public:
  explicit LoadedCodeObject(hsa_loaded_code_object_t LCO);

  [[nodiscard]] llvm::Expected<Executable> getExecutable() const;

  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  [[nodiscard]] llvm::Expected<hsa_ven_amd_loader_code_object_storage_type_t>
  getStorageType() const;

  [[nodiscard]] llvm::Expected<hsa_ven_amd_loader_loaded_code_object_kind_t>
  getKind();

  [[nodiscard]] llvm::Expected<llvm::object::ELF64LEObjectFile &>
  getStorageELF() const;

  [[nodiscard]] llvm::Expected<int> getStorageFile() const;

  [[nodiscard]] llvm::Expected<long> getLoadDelta() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getLoadedMemory() const;

  [[nodiscard]] llvm::Expected<std::string> getUri() const;

  llvm::Expected<ISA> getISA() const;

private:
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getStorageMemory() const;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
                        std::unique_ptr<llvm::object::ELF64LEObjectFile>>
      StorageELFOfLCOs;

  static llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle), hsa_isa_t>
      ISAOfLCOs;

  llvm::Error cache() const override;

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