#include "hsa_executable.hpp"

#include <llvm/BinaryFormat/ELF.h>

#include "error.hpp"
#include "hsa_agent.hpp"
#include "hsa_code_object_reader.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_loaded_code_object.hpp"

namespace luthier::hsa {

llvm::Expected<Executable>
Executable::create(hsa_profile_t Profile,
                   hsa_default_float_rounding_mode_t DefaultFloatRoundingMode,
                   const char *Options) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(hsa_executable_create_alt(
      Profile, DefaultFloatRoundingMode, Options, &Exec)));
  return Executable{Exec};
}

llvm::Error Executable::freeze(const char *Options) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_freeze_fn(asHsaType(), Options)));
  return Platform::instance().cacheExecutableOnExecutableFreeze(*this);
}

Executable::Executable(hsa_executable_t Exec)
    : HandleType<hsa_executable_t>(Exec) {}

llvm::Expected<hsa_profile_t> Executable::getProfile() {
  hsa_profile_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_PROFILE, &Out)));
  return Out;
}

llvm::Expected<hsa_executable_state_t> Executable::getState() const {
  hsa_executable_state_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_STATE, &Out)));
  return Out;
}

llvm::Expected<hsa_default_float_rounding_mode_t>
Executable::getRoundingMode() {
  hsa_default_float_rounding_mode_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &Out)));
  return Out;
}

llvm::Expected<std::vector<LoadedCodeObject>>
Executable::getLoadedCodeObjects() const {
  std::vector<LoadedCodeObject> LoadedCodeObjects;
  auto Iterator = [](hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
                     void *Data) -> hsa_status_t {
    auto Out = reinterpret_cast<std::vector<LoadedCodeObject> *>(Data);
    Out->emplace_back(LCO);
    return HSA_STATUS_SUCCESS;
  };
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable()
          .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
              this->asHsaType(), Iterator, &LoadedCodeObjects)));
  return LoadedCodeObjects;
}

llvm::Expected<hsa::LoadedCodeObject>
Executable::loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                                const hsa::GpuAgent &Agent,
                                llvm::StringRef Options) {
  hsa_loaded_code_object_t LCO;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_load_agent_code_object_fn(
          asHsaType(), Agent.asHsaType(), Reader.asHsaType(), Options.data(),
          &LCO)));
  LUTHIER_RETURN_ON_ERROR(
      Platform::instance().cacheExecutableOnLoadedCodeObjectCreation(*this));
  return LoadedCodeObject{LCO};
}

llvm::Error
Executable::defineExternalAgentGlobalVariable(const ExecutableSymbol &Symbol) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Symbol.getType() == VARIABLE));

  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(luthier::address_t, VariableAddress,
                                   Symbol.getVariableAddress());
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(std::string, VariableName, Symbol.getName());

  auto Agent = Symbol.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_agent_global_variable_define_fn(
          asHsaType(), Agent->asHsaType(), VariableName.c_str(),
          reinterpret_cast<void *>(
              *reinterpret_cast<uint64_t *>(VariableAddress)))));
  return llvm::Error::success();
}

llvm::Error
Executable::defineAgentReadOnlyVariable(const hsa::ExecutableSymbol &Symbol) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Symbol.getType() == VARIABLE));

  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(luthier::address_t, VariableAddress,
                                   Symbol.getVariableAddress());
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(std::string, VariableName, Symbol.getName());

  auto Agent = Symbol.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_readonly_variable_define_fn(
          asHsaType(), Agent->asHsaType(), VariableName.c_str(),
          reinterpret_cast<void *>(VariableAddress))));
  return llvm::Error::success();
}

llvm::Error Executable::destroy() {
  LUTHIER_RETURN_ON_ERROR(
      Platform::instance().invalidateExecutableOnExecutableDestroy(*this));
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_destroy_fn(asHsaType()));
}
llvm::Expected<bool> Executable::validate(llvm::StringRef Options) {
  uint32_t Result;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_validate_alt_fn(
          asHsaType(), Options.data(), &Result)));
  return Result == 0;
}

} // namespace luthier::hsa
