#ifndef HSA_EXECUTABLE_HPP
#define HSA_EXECUTABLE_HPP
#include <optional>
#include <vector>

#include <llvm/ADT/DenseMapInfo.h>

#include "hsa_code_object_reader.hpp"
#include "hsa_handle_type.hpp"
#include "hsa_platform.hpp"

namespace luthier {

class ToolExecutableManager;

namespace hsa {

class GpuAgent;

class ExecutableSymbol;

class LoadedCodeObject;

class Executable final : public HandleType<hsa_executable_t> {
  /*****************************************************************************
   * \brief Instrumented \b hsa_executable_t creation and destruction
   * functionality
   * Creation and destruction of Executables is only allowed through
   * \b CodeObjectManager
   ****************************************************************************/
  friend class luthier::ToolExecutableManager;

private:
  static llvm::Expected<Executable>
  create(hsa_profile_t Profile = HSA_PROFILE_FULL,
         hsa_default_float_rounding_mode_t DefaultFloatRoundingMode =
             HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
         const char *Options = "");

  llvm::Expected<hsa::LoadedCodeObject>
  loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                      const hsa::GpuAgent &Agent, llvm::StringRef Options = "");

  llvm::Error defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                                llvm::StringRef SymbolName,
                                                void *Address);

  llvm::Error defineAgentReadOnlyVariable(const hsa::ExecutableSymbol &Symbol);

  llvm::Error freeze(const char *Options = "");

  llvm::Expected<bool> validate(llvm::StringRef Options = "");

  llvm::Error destroy();

public:
  explicit Executable(hsa_executable_t Exec);

  llvm::Expected<hsa_profile_t> getProfile();

  [[nodiscard]] llvm::Expected<hsa_executable_state_t> getState() const;

  llvm::Expected<hsa_default_float_rounding_mode_t> getRoundingMode();

  /**
   * \return the \p hsa::LoadedCodeObject's in the executable
   */
  [[nodiscard]] llvm::Expected<std::vector<LoadedCodeObject>>
  getLoadedCodeObjects() const;
};

} // namespace hsa

} // namespace luthier

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::Executable> {
  static inline luthier::hsa::Executable getEmptyKey() {
    return luthier::hsa::Executable(
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::Executable getTombstoneKey() {
    return luthier::hsa::Executable(
        {DenseMapInfo<decltype(hsa_executable_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::Executable &Executable) {
    return DenseMapInfo<decltype(hsa_executable_t::handle)>::getHashValue(
        Executable.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::Executable &lhs,
                      const luthier::hsa::Executable &rhs) {
    return lhs.hsaHandle() == rhs.hsaHandle();
  }
};

} // namespace llvm

namespace std {

template <> struct hash<luthier::hsa::Executable> {
  size_t operator()(const luthier::hsa::Executable &Obj) const {
    return hash<unsigned long>()(Obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::Executable> {
  bool operator()(const luthier::hsa::Executable &Lhs,
                  const luthier::hsa::Executable &Rhs) const {
    return Lhs.hsaHandle() < Rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::Executable> {
  bool operator()(const luthier::hsa::Executable &Lhs,
                  const luthier::hsa::Executable &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace std

#endif