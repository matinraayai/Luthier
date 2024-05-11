#ifndef HSA_EXECUTABLE_SYMBOL_HPP
#define HSA_EXECUTABLE_SYMBOL_HPP
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>

#include <optional>
#include <string>

#include "hsa_handle_type.hpp"
#include "hsa_loaded_code_object.hpp"
#include "hsa_platform.hpp"
#include <luthier/kernel_descriptor.h>
#include <luthier/types.h>

namespace luthier::hsa {

class GpuAgent;

class ExecutableSymbol final : public ExecutableBackedCachable,
                               public HandleType<hsa_executable_symbol_t> {
private:
  typedef struct {
    hsa_loaded_code_object_t LCO;
    llvm::object::ELFSymbolRef Symbol;
    std::optional<llvm::object::ELFSymbolRef> KernelFunctionSymbol;
  } ExecutableSymbolInfo; // < Information required to represent an indirect
                          // function in HSA, since it is not implemented in
                          // ROCr yet

  ExecutableSymbolInfo SymbolInfo;

  /*****************************************************************************
   * \brief implementation of the \b ExecutableBackedCachableItem interface
   ****************************************************************************/
private:
  /**
   * \brief Keeps track of the Indirect functions
   * encountered so far, in order to expose them to the tool writer seamlessly
   * as an \ref hsa_executable_symbol
   */
  static llvm::DenseMap<decltype(hsa_executable_symbol_t::handle),
                        ExecutableSymbolInfo>
      SymbolHandleCache;

  llvm::Error cache() const override;

  bool isCached() const override;

  llvm::Error invalidate() const override;

public:
  ExecutableSymbol(hsa_executable_symbol_t Handle, hsa_loaded_code_object_t LCO,
                   const llvm::object::ELFSymbolRef &ELFSymbol,
                   std::optional<const llvm::object::ELFSymbolRef>
                       KernelFunctionSymbol = std::nullopt)
      : HandleType<hsa_executable_symbol_t>(Handle),
        SymbolInfo({LCO, ELFSymbol, KernelFunctionSymbol}) {}

  static llvm::Expected<ExecutableSymbol> createDeviceFunctionSymbol(
      hsa_loaded_code_object_t LCO,
      const object::ELFSymbolRef &DeviceFunctionELFSymbol);

  ExecutableSymbol(const ExecutableSymbol &Symbol)
      : HandleType<hsa_executable_symbol_t>(Symbol.asHsaType()),
        SymbolInfo(Symbol.SymbolInfo){};

  ExecutableSymbol &operator=(const ExecutableSymbol &Other) {
    Type<hsa_executable_symbol_t>::operator=(Other);
    this->SymbolInfo = Other.SymbolInfo;
    return *this;
  }

  ExecutableSymbol &operator=(ExecutableSymbol &&Other) noexcept {
    HandleType<hsa_executable_symbol_t>::operator=(Other);
    this->SymbolInfo = Other.SymbolInfo;
    return *this;
  }

  static llvm::Expected<ExecutableSymbol>
  fromHandle(hsa_executable_symbol_t Symbol);

  static llvm::Expected<ExecutableSymbol>
  fromKernelDescriptor(const KernelDescriptor *KD);

  [[nodiscard]] SymbolKind getType() const;

  [[nodiscard]] llvm::Expected<llvm::StringRef> getName() const;

  [[nodiscard]] llvm::Expected<hsa_symbol_linkage_t> getLinkage() const;

  [[nodiscard]] llvm::Expected<hsa_variable_allocation_t>
  getVariableAllocation() const;

  [[nodiscard]] llvm::Expected<luthier::address_t> getVariableAddress() const;

  [[nodiscard]] llvm::Expected<const KernelDescriptor *>
  getKernelDescriptor() const;

  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  [[nodiscard]] llvm::Expected<hsa::Executable> getExecutable() const;

  [[nodiscard]] llvm::Expected<LoadedCodeObject> getLoadedCodeObject() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getMachineCode() const;
};

} // namespace luthier::hsa

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::ExecutableSymbol> {
  static inline luthier::hsa::ExecutableSymbol getEmptyKey() {
    hsa_executable_symbol_t SymbolHandle{
        DenseMapInfo<decltype(hsa_executable_symbol_t::handle)>::getEmptyKey()};
    hsa_loaded_code_object_t LCOHandle{DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getEmptyKey()};
    return luthier::hsa::ExecutableSymbol{
        SymbolHandle, LCOHandle, llvm::object::SymbolRef{}, std::nullopt};
  }

  static inline luthier::hsa::ExecutableSymbol getTombstoneKey() {
    hsa_executable_symbol_t SymbolHandle{DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getTombstoneKey()};
    return luthier::hsa::ExecutableSymbol{
        SymbolHandle, {0}, llvm::object::SymbolRef{}, std::nullopt};
  }

  static unsigned getHashValue(const luthier::hsa::ExecutableSymbol &ISA) {
    return DenseMapInfo<decltype(hsa_executable_symbol_t::handle)>::
        getHashValue(ISA.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::ExecutableSymbol &Lhs,
                      const luthier::hsa::ExecutableSymbol &Rhs) {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace llvm

namespace std {

template <> struct hash<luthier::hsa::ExecutableSymbol> {
  size_t operator()(const luthier::hsa::ExecutableSymbol &Obj) const {
    return hash<unsigned long>()(Obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::ExecutableSymbol> {
  bool operator()(const luthier::hsa::ExecutableSymbol &Lhs,
                  const luthier::hsa::ExecutableSymbol &Rhs) const {
    return Lhs.hsaHandle() < Rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::ExecutableSymbol> {
  bool operator()(const luthier::hsa::ExecutableSymbol &Lhs,
                  const luthier::hsa::ExecutableSymbol &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace std

#endif