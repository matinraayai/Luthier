//===-- luthier.pp - Implementation of the Luthier API --------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of functions declared over at
/// luthier.h.
/// For the controller logic of Luthier, \see luthier::Controller
//===----------------------------------------------------------------------===//
#include <luthier/luthier.h>

#include <llvm/ADT/StringRef.h>

#include <optional>

#include "code_generator.hpp"
#include "code_lifter.hpp"
#include "controller.hpp"
#include "error.hpp"
#include "hip_intercept.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_platform.hpp"
#include "luthier/instrumentation_task.h"
#include "tool_executable_manager.hpp"
#include <luthier/instr.h>

namespace luthier {

namespace hip {

const HipCompilerDispatchTable &getSavedCompilerTable() {
  return hip::Interceptor::instance().getSavedCompilerTable();
}

} // namespace hip

namespace hsa {

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

llvm::Error instrument(hsa_executable_symbol_t Kernel,
                       const LiftedRepresentation &LR,
                       luthier::InstrumentationTask &ITask) {
  return CodeGenerator::instance().instrument(LR, ITask);
}

llvm::Error instrument(hsa_executable_t Exec,
                       const LiftedRepresentation &LR,
                       luthier::InstrumentationTask &ITask) {
  return CodeGenerator::instance().instrument(LR, ITask);
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
      reinterpret_cast<const luthier::KernelDescriptor *>(
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
llvm::Expected<std::unique_ptr<LiftedRepresentation<hsa_executable_symbol_t>>>
cloneRepresentation(const LiftedRepresentation<hsa_executable_symbol_t> &LR) {
  return CodeLifter::instance().cloneRepresentation<hsa_executable_symbol_t>(
      LR);
}
llvm::Expected<std::unique_ptr<LiftedRepresentation<hsa_executable_t>>>
luthier::cloneRepresentation(const LiftedRepresentation<hsa_executable_t> &LR) {
  return CodeLifter::instance().cloneRepresentation<hsa_executable_t>(LR);
}

} // namespace luthier
