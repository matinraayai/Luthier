#include <luthier/luthier.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <optional>

#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "error.hpp"
#include "hip_intercept.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "log.hpp"
#include "target_manager.hpp"
#include <luthier/instr.hpp>

namespace luthier {

void hipApiInternalCallback(void *CBData, ApiEvtPhase Phase, int ApiId,
                            bool *SkipFunc, std::optional<std::any> *Out) {
  LUTHIER_LOG_FUNCTION_CALL_START
  if (Phase == API_EVT_PHASE_ENTER) {
    if (ApiId == HIP_PRIVATE_API_ID___hipRegisterFunction) {
      auto &COM = CodeObjectManager::instance();
      auto LastRFuncArgs =
          reinterpret_cast<hip___hipRegisterFunction_api_args_t *>(CBData);
      // If the function doesn't have __luthier_wrap__ in its name then it
      // belongs to the instrumented application or HIP can manage it on its own
      // since no device function is present to strip from it
      if (llvm::StringRef(LastRFuncArgs->deviceFunction)
              .find(luthier::DeviceFunctionWrap) != llvm::StringRef::npos) {
        COM.registerInstrumentationFunctionWrapper(
            LastRFuncArgs->hostFunction, LastRFuncArgs->deviceFunction);
      }
    }
  }
  LUTHIER_LOG_FUNCTION_CALL_END
}

void hsaApiInternalCallback(hsa_api_evt_args_t *CBData, ApiEvtPhase Phase,
                            hsa_api_evt_id_t ApiId, bool *SkipFunction) {
  LUTHIER_LOG_FUNCTION_CALL_START
  LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((constructor)) void init() {
  LUTHIER_LOG_FUNCTION_CALL_START
  auto &HipInterceptor = HipInterceptor::instance();
  LUTHIER_CHECK_WITH_MSG(HipInterceptor.isEnabled(),
                         "HIP Interceptor failed to initialize");
  HipInterceptor.setInternalCallback(luthier::hipApiInternalCallback);
  HipInterceptor.setUserCallback(luthier::hipApiUserCallback);
  HipInterceptor.enableInternalCallback(
      HIP_PRIVATE_API_ID___hipRegisterFunction);
  LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((destructor)) void finalize() {
  LUTHIER_LOG_FUNCTION_CALL_START
  luthier::atHsaApiTableUnload();
  LUTHIER_LOG_FUNCTION_CALL_END
}

llvm::Expected<const std::vector<Instr> &>
disassembleKernel(hsa_executable_symbol_t Symbol) {
  return luthier::CodeLifter::instance().disassemble(
      hsa::ExecutableSymbol::fromHandle(Symbol));
}

llvm::Expected<std::tuple<std::unique_ptr<llvm::Module>,
                          std::unique_ptr<llvm::MachineModuleInfoWrapperPass>,
                          luthier::LiftedSymbolInfo>>
liftSymbol(hsa_executable_symbol_t Symbol) {
  return luthier::CodeLifter::instance().liftSymbol(
      hsa::ExecutableSymbol::fromHandle(Symbol));
}

llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet) {
  auto Symbol = luthier::hsa::ExecutableSymbol::fromKernelDescriptor(
      reinterpret_cast<const luthier::KernelDescriptor *>(
          Packet.kernel_object));
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());

  const auto InstrumentedKernel =
      luthier::CodeObjectManager::instance().getInstrumentedKernel(*Symbol);

  auto InstrumentedKD = InstrumentedKernel->getKernelDescriptor();

  LUTHIER_RETURN_ON_ERROR(InstrumentedKD.takeError());

  Packet.kernel_object = reinterpret_cast<uint64_t>(*InstrumentedKD);
  return llvm::Error::success();
}

} // namespace luthier

extern "C" {
const HsaApiTable *luthier_get_hsa_table() {
  return &luthier::hsa::Interceptor::instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s *luthier_get_hsa_ven_amd_loader() {
  return &luthier::hsa::Interceptor::instance().getHsaVenAmdLoaderTable();
}

void *luthier_get_hip_function(const char *funcName) {
  return luthier::HipInterceptor::instance().getHipFunction(funcName);
}

luthier_status_t
luthier_create_instrumented_kernel(hsa_executable_symbol_t symbol) {
  auto Error = luthier::CodeGenerator::instance().createInstrumentationTask(
      luthier::hsa::ExecutableSymbol::fromHandle(symbol));
  return luthier::convertErrorToStatusCode(Error);
}

luthier_status_t luthier_insert_call(luthier_instruction_t instr,
                                     const void *dev_func,
                                     luthier_ipoint_t point) {
  auto Error = luthier::CodeGenerator::instance().insertCall(
      *luthier::hsa::Instr::fromHandle(instr), dev_func, point);
  return luthier::convertErrorToStatusCode(Error);
}

void luthier_add_call_arg_const_val32(luthier_instruction_t instr,
                                      uint32_t val) {
  luthier::CodeGenerator::instance().addArgToLastCall();
}

void luthier_add_call_arg_const_val64(luthier_instruction_t instr,
                                      uint64_t val) {
  luthier::CodeGenerator::instance().addArgToLastCall();
}

void luthier_enable_hsa_op_callback(hsa_api_evt_id_t op) {
  luthier::hsa::Interceptor::instance().enableUserCallback(op);
}

void luthier_disable_hsa_op_callback(hsa_api_evt_id_t op) {
  luthier::hsa::Interceptor::instance().disableUserCallback(op);
}

void luthier_enable_all_hsa_callbacks() {
  luthier::hsa::Interceptor::instance().enableAllUserCallbacks();
}

void luthier_disable_all_hsa_callbacks() {
  luthier::hsa::Interceptor::instance().disableAllUserCallbacks();
}

void luthier_enable_hip_op_callback(uint32_t op) {
  luthier::HipInterceptor::instance().enableUserCallback(op);
}

void luthier_disable_hip_op_callback(uint32_t op) {
  luthier::HipInterceptor::instance().disableUserCallback(op);
}

void luthier_enable_all_hip_callbacks() {
  luthier::HipInterceptor::instance().enableAllUserCallbacks();
}

void luthier_disable_all_hip_callbacks() {
  luthier::HipInterceptor::instance().disableAllUserCallbacks();
}

// NOLINTBEGIN

__attribute__((
    visibility("default"))) extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

__attribute__((visibility("default"))) bool
OnLoad(HsaApiTable *table, uint64_t runtime_version, uint64_t failed_tool_count,
       const char *const *failed_tool_names) {
  LUTHIER_LOG_FUNCTION_CALL_START
  [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
  bool res = luthier::hsa::Interceptor::instance().captureHsaApiTable(table);
  luthier::atHsaApiTableLoad();
  auto &hsaInterceptor = luthier::hsa::Interceptor::instance();
  hsaInterceptor.setInternalCallback(luthier::hsaApiInternalCallback);
  hsaInterceptor.setUserCallback(luthier::atHsaEvt);
  return res;
  LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((visibility("default"))) void OnUnload() {
  LUTHIER_LOG_FUNCTION_CALL_START
  luthier::hsa::Interceptor::instance().uninstallApiTables();
  LUTHIER_LOG_FUNCTION_CALL_END
}
}
// NOLINTEND