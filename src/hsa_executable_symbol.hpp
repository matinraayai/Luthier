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
  hsa_agent_t agent_;
  hsa_executable_t executable_;
  std::optional<std::string> indirectFunctionName_{std::nullopt};
  std::optional<llvm::ArrayRef<uint8_t>> indirectFunctionCode_{std::nullopt};

public:
  ExecutableSymbol(hsa_executable_symbol_t symbol, hsa_agent_t agent,
                   hsa_executable_t executable)
      : HandleType<hsa_executable_symbol_t>(symbol), agent_(agent),
        executable_(executable){};

  ExecutableSymbol(std::string indirectFunctionName,
                   llvm::ArrayRef<uint8_t> indirectFunctionCode,
                   hsa_agent_t agent, hsa_executable_t executable)
      : HandleType<hsa_executable_symbol_t>({0}), agent_(agent),
        executable_(executable), indirectFunctionCode_(indirectFunctionCode),
        indirectFunctionName_(std::move(indirectFunctionName)){};

  [[nodiscard]] uint64_t hsaHandle() const override {
    if (HandleType<hsa_executable_symbol_t>::hsaHandle() == 0 &&
        indirectFunctionCode_.has_value()) {
      return reinterpret_cast<luthier_address_t>(indirectFunctionCode_->data());
    } else
      return HandleType<hsa_executable_symbol_t>::hsaHandle();
  }

  static llvm::Expected<ExecutableSymbol>
  fromKernelDescriptor(const hsa::KernelDescriptor *kd);

  [[nodiscard]] llvm::Expected<hsa_symbol_kind_t> getType() const;

  [[nodiscard]] llvm::Expected<std::string> getName() const;

  [[nodiscard]] hsa_symbol_linkage_t getLinkage() const;

  [[nodiscard]] llvm::Expected<luthier_address_t> getVariableAddress() const;

  [[nodiscard]] llvm::Expected<const KernelDescriptor *>
  getKernelDescriptor() const;

  [[nodiscard]] GpuAgent getAgent() const;

  [[nodiscard]] Executable getExecutable() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getMachineCode() const;
};

} // namespace luthier::hsa

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