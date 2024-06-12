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
#include "code_object_manager.hpp"
#include "controller.hpp"
#include "disassembler.hpp"
#include "error.hpp"
#include "hip_intercept.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_platform.hpp"
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

} // namespace luthier
