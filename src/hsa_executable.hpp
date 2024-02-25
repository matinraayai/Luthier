#ifndef HSA_EXECUTABLE_HPP
#define HSA_EXECUTABLE_HPP
#include <optional>
#include <vector>

#include "hsa_code_object_reader.hpp"
#include "hsa_handle_type.hpp"

namespace luthier {

class CodeObjectManager;

namespace hsa {

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

class Executable : public HandleType<hsa_executable_t> {
  friend class luthier::CodeObjectManager;

private:
  static llvm::Expected<Executable>
  create(hsa_profile_t profile = HSA_PROFILE_FULL,
         hsa_default_float_rounding_mode_t defaultFloatRoundingMode =
             HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
         const char *options = "");

  llvm::Expected<hsa::LoadedCodeObject>
  loadCodeObject(hsa::CodeObjectReader reader, hsa::GpuAgent agent);

  llvm::Error freeze(const char *options = "");

  llvm::Error destroy();

public:
  explicit Executable(hsa_executable_t executable);

  llvm::Expected<hsa_profile_t> getProfile();

  llvm::Expected<hsa_executable_state_t> getState();

  llvm::Expected<hsa_default_float_rounding_mode_t> getRoundingMode();

  [[nodiscard]] llvm::Expected<std::vector<ExecutableSymbol>>
  getSymbols(const luthier::hsa::GpuAgent &agent) const;

  [[nodiscard]] llvm::Expected<std::optional<ExecutableSymbol>>
  getSymbolByName(const luthier::hsa::GpuAgent &agent,
                  const std::string &name) const;

  [[nodiscard]] llvm::Expected<std::vector<LoadedCodeObject>>
  getLoadedCodeObjects() const;

  [[nodiscard]] llvm::Expected<std::vector<hsa::GpuAgent>> getAgents() const;
};

} // namespace hsa

} // namespace luthier

namespace std {

template <> struct hash<luthier::hsa::Executable> {
  size_t operator()(const luthier::hsa::Executable &obj) const {
    return hash<unsigned long>()(obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::Executable> {
  bool operator()(const luthier::hsa::Executable &lhs,
                  const luthier::hsa::Executable &rhs) const {
    return lhs.hsaHandle() < rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::Executable> {
  bool operator()(const luthier::hsa::Executable &lhs,
                  const luthier::hsa::Executable &rhs) const {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace std
#endif