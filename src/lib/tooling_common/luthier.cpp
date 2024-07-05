//===-- luthier.pp - Implementation of the Luthier API --------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of functions declared over at
/// luthier.h.
/// For the controller logic of Luthier, see <tt>luthier::Controller</tt>.
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>

#include <llvm/ADT/StringRef.h>

#include <optional>

#include "tooling_common/code_generator.hpp"
#include "tooling_common/code_lifter.hpp"
#include "common/error.hpp"
#include "hip/hip_intercept.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include "hsa/hsa_intercept.hpp"
#include "hsa/hsa_platform.hpp"
#include "luthier/instrumentation_task.h"
#include "tooling_common/tool_executable_manager.hpp"
#include <luthier/instr.h>

namespace luthier {

namespace hip {

const HipCompilerDispatchTable &getSavedCompilerTable() {
  return hip::Interceptor::instance().getSavedCompilerTable();
}

} // namespace hip

namespace hsa {



void setAtHsaApiEvtCallback(
    const std::function<void(ApiEvtArgs *, ApiEvtPhase, ApiEvtID)> &Callback) {
  hsa::Interceptor::instance().setUserCallback(Callback);
}
} // namespace hsa

namespace hsa {

const HsaApiTable &getHsaApiTable() {
  return hsa::Interceptor::instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s &getHsaVenAmdLoaderTable() {
  return hsa::Interceptor::instance().getHsaVenAmdLoaderTable();
}

llvm::Expected<SymbolKind> getSymbolKind(hsa_executable_symbol_t Symbol) {
  auto SymbolWrapper = ExecutableSymbol::fromHandle(Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
  return SymbolWrapper->getType();
}

llvm::Expected<llvm::StringRef> getSymbolName(hsa_executable_symbol_t Symbol) {
  auto SymbolWrapper = ExecutableSymbol::fromHandle(Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
  return SymbolWrapper->getName();
}

llvm::Expected<hsa_executable_t>
getExecutableOfSymbol(hsa_executable_symbol_t Symbol) {
  auto SymbolWrapper = ExecutableSymbol::fromHandle(Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
  auto Exec = SymbolWrapper->getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());
  return Exec->asHsaType();
}

llvm::Expected<std::optional<hsa_loaded_code_object_t>>
getDefiningLoadedCodeObject(hsa_executable_symbol_t Symbol) {
  auto SymbolWrapper = ExecutableSymbol::fromHandle(Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
  auto DefiningLCO = SymbolWrapper->getDefiningLoadedCodeObject();
  if (DefiningLCO.has_value())
    return DefiningLCO->asHsaType();
  else
    return std::nullopt;
}

void enableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID) {
  hsa::Interceptor::instance().enableUserCallback(ApiID);
}

void disableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID) {
  hsa::Interceptor::instance().disableUserCallback(ApiID);
}

void enableAllHsaApiEvtCallbacks() {
  hsa::Interceptor::instance().enableAllUserCallbacks();
}

void disableAllHsaCallbacks() {
  hsa::Interceptor::instance().disableAllUserCallbacks();
}

} // namespace hsa

llvm::Expected<const std::vector<hsa::Instr> &>
disassemble(hsa_executable_symbol_t Func) {
  auto SymbolWrapper = hsa::ExecutableSymbol::fromHandle(Func);
  LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
  return luthier::CodeLifter::instance().disassemble(*SymbolWrapper);
}

llvm::Expected<const luthier::LiftedRepresentation &>
lift(hsa_executable_symbol_t Kernel) {
  auto KernelWrapper = hsa::ExecutableSymbol::fromHandle(Kernel);
  LUTHIER_RETURN_ON_ERROR(KernelWrapper.takeError());
  return CodeLifter::instance().lift(*KernelWrapper);
}

llvm::Expected<const luthier::LiftedRepresentation &>
lift(hsa_executable_t Executable) {
  return CodeLifter::instance().lift(hsa::Executable{Executable});
}

llvm::Error
instrumentAndLoad(hsa_executable_symbol_t Kernel,
                  const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset) {
  auto KernelWrapper = hsa::ExecutableSymbol::fromHandle(Kernel);
  LUTHIER_RETURN_ON_ERROR(KernelWrapper.takeError());
  return CodeGenerator::instance().instrument(LR, Mutator);
}

llvm::Error
instrumentAndLoad(hsa_executable_t Exec, const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset) {
  return CodeGenerator::instance().instrument(
      LR, Mutator);
}

llvm::Expected<bool> isKernelInstrumented(hsa_executable_symbol_t Kernel,
                                          llvm::StringRef Preset) {
  auto Symbol = hsa::ExecutableSymbol::fromHandle(Kernel);
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
  return ToolExecutableManager::instance().isKernelInstrumented(*Symbol,
                                                                Preset);
}

llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                                     llvm::StringRef Preset) {
  auto Symbol = luthier::hsa::ExecutableSymbol::fromKernelDescriptor(
      reinterpret_cast<const luthier::hsa::KernelDescriptor *>(
          Packet.kernel_object));
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());

  auto InstrumentedKernel =
      luthier::ToolExecutableManager::instance().getInstrumentedKernel(*Symbol,
                                                                       Preset);
  LUTHIER_RETURN_ON_ERROR(InstrumentedKernel.takeError());
  auto InstrumentedKD = InstrumentedKernel->getKernelDescriptor();

  LUTHIER_RETURN_ON_ERROR(InstrumentedKD.takeError());

  Packet.kernel_object = reinterpret_cast<uint64_t>(*InstrumentedKD);

  auto InstrumentedKernelMD = InstrumentedKernel->getKernelMetadata();
  LUTHIER_RETURN_ON_ERROR(InstrumentedKernelMD.takeError());

  Packet.private_segment_size = InstrumentedKernelMD->PrivateSegmentFixedSize;

  return llvm::Error::success();
}

} // namespace luthier
