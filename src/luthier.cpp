#include <luthier/luthier.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/amd_detail/hip_api_trace.hpp>

#include <optional>

#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "error.hpp"
#include "global_singleton_manager.hpp"
#include "hip_intercept.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_platform.hpp"
#include "log.hpp"
#include "target_manager.hpp"
#include <luthier/instr.h>

namespace luthier {

static GlobalSingletonManager *GSM{nullptr};

namespace hip {

void *getHipFunctionPtr(llvm::StringRef FuncName) {
  return hip::Interceptor::instance().getHipFunction(FuncName);
}

} // namespace hip

__attribute__((constructor)) void init() {
  static std::once_flag Once{};
  std::call_once(Once, [](){
    luthier::GSM = new luthier::GlobalSingletonManager();
  });

}

void rocprofilerFinalize(void *Data) {
  LUTHIER_LOG_FUNCTION_CALL_START
  luthier::hsa::atHsaApiTableUnload();
  luthier::hsa::Interceptor::instance().uninstallApiTables();
  LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((destructor)) void finalize() {
  delete GSM;
  llvm::llvm_shutdown();
}

namespace hsa {

void internalApiCallback(hsa::ApiEvtArgs *CBData, ApiEvtPhase Phase,
                         hsa::ApiEvtID ApiId, bool *SkipFunction) {
  LUTHIER_LOG_FUNCTION_CALL_START
  if (Phase == API_EVT_PHASE_EXIT &&
      ApiId == HSA_API_EVT_ID_hsa_executable_freeze) {
    hsa::Executable Exec(CBData->hsa_executable_destroy.executable);
    // Cache the executable and its items
    if (auto Err = Platform::instance().cacheExecutableOnExecutableFreeze(Exec))
      llvm::report_fatal_error("Tool executable register failed");
    // Check if the executable belongs to the tool and not the app
    if (auto Err = CodeObjectManager::instance()
                       .checkIfLuthierToolExecutableAndRegister(Exec)) {
      llvm::report_fatal_error("Tool executable check failed");
    }
  }
  if (Phase == API_EVT_PHASE_EXIT &&
      ApiId == HSA_API_EVT_ID_hsa_executable_load_agent_code_object) {
    // because the output of hsa_executable_load_agent_code_object can be set to
    // nullptr by the app, we have to access it by iterating over the LCOs of
    // the Exec it was created for
    hsa::Executable Exec(
        CBData->hsa_executable_load_agent_code_object.executable);
    if (auto Err =
            Platform::instance().cacheExecutableOnLoadedCodeObjectCreation(
                Exec)) {
      llvm::report_fatal_error("Caching of Loaded Code Object failed!");
    }
  }
  if (Phase == API_EVT_PHASE_ENTER &&
      ApiId == HSA_API_EVT_ID_hsa_executable_destroy) {
    hsa::Executable Exec(CBData->hsa_executable_destroy.executable);
    if (auto Err =
            CodeLifter::instance().invalidateCachedExecutableItems(Exec)) {
      llvm::report_fatal_error("Executable cache invalidation failed");
    }

    if (auto Err = Platform::instance().invalidateExecutableOnExecutableDestroy(
            Exec)) {
      llvm::report_fatal_error("Executable cache invalidation failed");
    }
  }
  LUTHIER_LOG_FUNCTION_CALL_END
}

const HsaApiTable &getHsaApiTable() {
  return hsa::Interceptor::instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s &getHsaVenAmdLoaderTable() {
  return hsa::Interceptor::instance().getHsaVenAmdLoaderTable();
}

void enableHsaOpCallback(hsa::ApiEvtID Op) {
  hsa::Interceptor::instance().enableUserCallback(Op);
}

void disableHsaOpCallback(hsa::ApiEvtID Op) {
  hsa::Interceptor::instance().disableUserCallback(Op);
}

void enableAllHsaCallbacks() {
  hsa::Interceptor::instance().enableAllUserCallbacks();
}

void disableAllHsaCallbacks() {
  hsa::Interceptor::instance().disableAllUserCallbacks();
}

} // namespace hsa

llvm::Expected<const std::vector<hsa::Instr> &>
disassembleSymbol(hsa_executable_symbol_t Symbol) {
  auto SymbolWrapper = hsa::ExecutableSymbol::fromHandle(Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
  return luthier::CodeLifter::instance().disassemble(*SymbolWrapper);
}

llvm::Expected<std::tuple<std::unique_ptr<llvm::Module>,
                          std::unique_ptr<llvm::MachineModuleInfoWrapperPass>,
                          luthier::LiftedSymbolInfo>>
liftSymbol(hsa_executable_symbol_t Symbol) {
  auto SymbolWrapper = hsa::ExecutableSymbol::fromHandle(Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
  return luthier::CodeLifter::instance().liftSymbol(*SymbolWrapper);
}

llvm::Error
instrument(std::unique_ptr<llvm::Module> Module,
           std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP,
           const LiftedSymbolInfo &LSO, luthier::InstrumentationTask &ITask) {
  return CodeGenerator::instance().instrument(std::move(Module),
                                              std::move(MMIWP), LSO, ITask);
}

llvm::Expected<bool> isKernelInstrumented(hsa_executable_symbol_t Kernel) {
  auto Symbol = hsa::ExecutableSymbol::fromHandle(Kernel);
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
  return CodeObjectManager::instance().isKernelInstrumented(*Symbol);
}

llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet) {
  auto Symbol = luthier::hsa::ExecutableSymbol::fromKernelDescriptor(
      reinterpret_cast<const luthier::KernelDescriptor *>(
          Packet.kernel_object));
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());

  auto InstrumentedKernel =
      luthier::CodeObjectManager::instance().getInstrumentedKernel(*Symbol);
  LUTHIER_RETURN_ON_ERROR(InstrumentedKernel.takeError());

  auto InstrumentedKD = InstrumentedKernel->getKernelDescriptor();

  LUTHIER_RETURN_ON_ERROR(InstrumentedKD.takeError());

  Packet.kernel_object = reinterpret_cast<uint64_t>(*InstrumentedKD);
  return llvm::Error::success();
}

void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                             uint64_t LibVersion, uint64_t LibInstance,
                             void **Tables, uint64_t NumTables, void *Data) {
  if (Type == ROCPROFILER_HSA_TABLE) {
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    luthier::hsa::Interceptor::instance().captureHsaApiTable(Table);
    luthier::hsa::atHsaApiTableLoad();
    auto &hsaInterceptor = luthier::hsa::Interceptor::instance();
    hsaInterceptor.setInternalCallback(luthier::hsa::internalApiCallback);
    hsaInterceptor.setUserCallback(luthier::hsa::atHsaEvt);
    hsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_freeze);
    hsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_destroy);
    hsaInterceptor.enableInternalCallback(
        luthier::hsa::HSA_API_EVT_ID_hsa_executable_load_agent_code_object);
  }
}

} // namespace luthier

extern "C" __attribute__((used)) rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t Version, const char *RuntimeVersion,
                      uint32_t Priority, rocprofiler_client_id_t *ID) {
  ID->name = "Luthier";
  rocprofiler_at_intercept_table_registration(luthier::apiRegistrationCallback,
                                              ROCPROFILER_HSA_TABLE, nullptr);

  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr,
      &luthier::rocprofilerFinalize,
      nullptr};
  return &Cfg;
}