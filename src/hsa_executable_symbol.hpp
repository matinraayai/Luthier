#ifndef HSA_EXECUTABLE_SYMBOL_HPP
#define HSA_EXECUTABLE_SYMBOL_HPP
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>

#include <optional>
#include <string>

#include "hsa_handle_type.hpp"
#include "hsa_kernel_descriptor.hpp"
#include "luthier_types.h"

namespace luthier::hsa {

class GpuAgent;

class Executable;

class ExecutableSymbol : public HandleType<hsa_executable_symbol_t> {
private:
  hsa_agent_t Agent;
  hsa_executable_t Executable;
  std::optional<std::string> IndirectFunctionName{std::nullopt};
  std::optional<llvm::ArrayRef<uint8_t>> IndirectFunctionCode{std::nullopt};

public:
  ExecutableSymbol(hsa_executable_symbol_t Symbol, hsa_agent_t Agent,
                   hsa_executable_t Executable)
      : HandleType<hsa_executable_symbol_t>(Symbol), Agent(Agent),
        Executable(Executable){};

  ExecutableSymbol(std::string IndirectFunctionName,
                   llvm::ArrayRef<uint8_t> IndirectFunctionCode,
                   hsa_agent_t Agent, hsa_executable_t Executable)
      : HandleType<hsa_executable_symbol_t>({0}), Agent(Agent),
        Executable(Executable), IndirectFunctionCode(IndirectFunctionCode),
        IndirectFunctionName(std::move(IndirectFunctionName)){};

  ExecutableSymbol(const ExecutableSymbol &Symbol)
      : HandleType<hsa_executable_symbol_t>(Symbol.asHsaType()),
        Agent(Symbol.Agent), Executable(Symbol.Executable),
        IndirectFunctionCode(Symbol.IndirectFunctionCode),
        IndirectFunctionName(Symbol.IndirectFunctionName){};

  ExecutableSymbol &operator=(const ExecutableSymbol &Other) {
    Type<hsa_executable_symbol_t>::operator=(Other);
    this->Executable = Other.Executable;
    this->Agent = Other.Agent;
    this->IndirectFunctionCode = Other.IndirectFunctionCode;
    this->IndirectFunctionName = Other.IndirectFunctionName;
    return *this;
  }

  ExecutableSymbol &operator=(ExecutableSymbol &&Other) noexcept {
    HandleType<hsa_executable_symbol_t>::operator=(Other);
    this->Executable = Other.Executable;
    this->Agent = Other.Agent;
    this->IndirectFunctionCode = Other.IndirectFunctionCode;
    this->IndirectFunctionName = Other.IndirectFunctionName;
    return *this;
  }

  [[nodiscard]] uint64_t hsaHandle() const override {
    if (HandleType<hsa_executable_symbol_t>::hsaHandle() == 0 &&
        IndirectFunctionCode.has_value()) {
      return reinterpret_cast<luthier_address_t>(IndirectFunctionCode->data());
    } else
      return HandleType<hsa_executable_symbol_t>::hsaHandle();
  }

  static llvm::Expected<ExecutableSymbol>
  fromKernelDescriptor(const hsa::KernelDescriptor *KD);

  [[nodiscard]] llvm::Expected<hsa_symbol_kind_t> getType() const;

  [[nodiscard]] llvm::Expected<std::string> getName() const;

  [[nodiscard]] hsa_symbol_linkage_t getLinkage() const;

  [[nodiscard]] llvm::Expected<luthier_address_t> getVariableAddress() const;

  [[nodiscard]] llvm::Expected<const KernelDescriptor *>
  getKernelDescriptor() const;

  [[nodiscard]] GpuAgent getAgent() const;

  [[nodiscard]] hsa::Executable getExecutable() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getMachineCode() const;
};

} // namespace luthier::hsa

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::ExecutableSymbol> {
  static inline luthier::hsa::ExecutableSymbol getEmptyKey() {
    return luthier::hsa::ExecutableSymbol(
        {DenseMapInfo<
            decltype(hsa_executable_symbol_t::handle)>::getEmptyKey()},
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getEmptyKey()},
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::ExecutableSymbol getTombstoneKey() {
    return luthier::hsa::ExecutableSymbol(
        {DenseMapInfo<
            decltype(hsa_executable_symbol_t::handle)>::getTombstoneKey()},
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getTombstoneKey()},
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::ExecutableSymbol &ISA) {
    return DenseMapInfo<decltype(hsa_executable_symbol_t::handle)>::
        getHashValue(ISA.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::ExecutableSymbol &lhs,
                      const luthier::hsa::ExecutableSymbol &rhs) {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace llvm

namespace std {

template <> struct hash<luthier::hsa::ExecutableSymbol> {
  size_t operator()(const luthier::hsa::ExecutableSymbol &obj) const {
    return hash<unsigned long>()(obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::ExecutableSymbol> {
  bool operator()(const luthier::hsa::ExecutableSymbol &lhs,
                  const luthier::hsa::ExecutableSymbol &rhs) const {
    return lhs.hsaHandle() < rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::ExecutableSymbol> {
  bool operator()(const luthier::hsa::ExecutableSymbol &lhs,
                  const luthier::hsa::ExecutableSymbol &rhs) const {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace std

#endif