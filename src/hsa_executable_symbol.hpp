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

class ExecutableSymbol final : public ExecutableBackedCachableItem,
                               public HandleType<hsa_executable_symbol_t> {
private:
  typedef struct {
    hsa_loaded_code_object_t LCO;
    std::string Name;
    llvm::ArrayRef<uint8_t> Code;
  } DeviceFunctionInfo; // < Information required to represent an indirect
                        // function in HSA, since it is not implemented in
                        // ROCr yet

  std::optional<DeviceFunctionInfo> DFO{std::nullopt}; // < Will only be used
                                                       // if the symbol is
                                                       // an indirect function

  /*****************************************************************************
   * \brief implementation of the \b ExecutableBackedCachableItem interface
   ****************************************************************************/
private:
  /**
   * \brief Keeps track of the Indirect functions
   * encountered so far, in order to expose them to the tool writer seamlessly
   * as an \ref hsa_executable_symbol
   */
  static std::unordered_map<decltype(hsa_executable_symbol_t::handle),
                            DeviceFunctionInfo>
      DeviceFunctionHandleCache;

  llvm::Error cache() const override;

  llvm::Error invalidate() const override;

  explicit ExecutableSymbol(hsa_executable_symbol_t Symbol)
      : HandleType<hsa_executable_symbol_t>(Symbol){};

public:
  ExecutableSymbol(std::string IndirectFunctionName,
                   llvm::ArrayRef<uint8_t> IndirectFunctionCode,
                   const hsa::LoadedCodeObject &LCO)
      : HandleType<hsa_executable_symbol_t>(
            {reinterpret_cast<decltype(hsa_executable_symbol_t::handle)>(
                IndirectFunctionCode.data())}) {
    DFO.emplace(LCO.asHsaType(), std::move(IndirectFunctionName),
                IndirectFunctionCode);
    // Cache the indirect function calls to allow conversion from hsa handles
    {
      std::lock_guard Lock(getMutex());
      DeviceFunctionHandleCache.insert({hsaHandle(), *DFO});
    }
  }

  ExecutableSymbol(const ExecutableSymbol &Symbol)
      : HandleType<hsa_executable_symbol_t>(Symbol.asHsaType()),
        DFO(Symbol.DFO){};

  ExecutableSymbol &operator=(const ExecutableSymbol &Other) {
    Type<hsa_executable_symbol_t>::operator=(Other);
    this->DFO = Other.DFO;
    return *this;
  }

  ExecutableSymbol &operator=(ExecutableSymbol &&Other) noexcept {
    HandleType<hsa_executable_symbol_t>::operator=(Other);
    this->DFO = Other.DFO;
    return *this;
  }

  static ExecutableSymbol fromHandle(hsa_executable_symbol_t Symbol);

  static llvm::Expected<ExecutableSymbol>
  fromKernelDescriptor(const KernelDescriptor *KD);

  [[nodiscard]] llvm::Expected<hsa_symbol_kind_t> getType() const;

  [[nodiscard]] llvm::Expected<std::string> getName() const;

  [[nodiscard]] llvm::Expected<hsa_symbol_linkage_t> getLinkage() const;

  [[nodiscard]] llvm::Expected<hsa_variable_allocation_t>
  getVariableAllocation() const;

  [[nodiscard]] llvm::Expected<luthier::address_t> getVariableAddress() const;

  [[nodiscard]] llvm::Expected<const KernelDescriptor *>
  getKernelDescriptor() const;

  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  [[nodiscard]] llvm::Expected<hsa::Executable> getExecutable() const;

  [[nodiscard]] llvm::Expected<std::optional<LoadedCodeObject>>
  getLoadedCodeObject() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getMachineCode() const;
};

} // namespace luthier::hsa

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::ExecutableSymbol> {
  static inline luthier::hsa::ExecutableSymbol getEmptyKey() {
    return luthier::hsa::ExecutableSymbol::fromHandle({DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::ExecutableSymbol getTombstoneKey() {
    return luthier::hsa::ExecutableSymbol::fromHandle({DenseMapInfo<
        decltype(hsa_executable_symbol_t::handle)>::getTombstoneKey()});
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