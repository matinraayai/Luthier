//===-- luthier.pp - Implementation of the Luthier API --------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of functions declared over at
/// luthier.h.
/// For the controller logic of Luthier, see <tt>luthier::Controller</tt>.
//===----------------------------------------------------------------------===//
#include "luthier/luthier.h"

#include <llvm/ADT/StringRef.h>

#include <optional>

#include "common/error.hpp"
#include "hip/hip_compiler_intercept.hpp"
#include "hip/hip_runtime_intercept.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include "hsa/hsa_intercept.hpp"
#include "hsa/hsa_platform.hpp"
#include "luthier/instrumentation_task.h"
#include "tooling_common/code_generator.hpp"
#include "tooling_common/code_lifter.hpp"
#include "tooling_common/tool_executable_manager.hpp"
#include <luthier/instr.h>

namespace luthier {

namespace hip {

const HipCompilerDispatchTable &getSavedCompilerTable() {
  return hip::CompilerInterceptor::instance().getSavedCompilerTable();
}

} // namespace hip

namespace hsa {

void setAtHsaApiEvtCallback(
    const std::function<void(ApiEvtArgs *, ApiEvtPhase, ApiEvtID)> &Callback) {
  hsa::Interceptor::instance().setUserCallback(Callback);
}


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

llvm::Expected<std::unique_ptr<LiftedRepresentation>>
instrument(const LiftedRepresentation &LR,
           llvm::function_ref<llvm::Error(InstrumentationTask &,
                                          LiftedRepresentation &)>
               Mutator) {
  return CodeGenerator::instance().instrument(LR, Mutator);
}

llvm::Error printLiftedRepresentation(
    LiftedRepresentation &LR,
    llvm::SmallVectorImpl<
        std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>>
        &CompiledObjectFiles,
    llvm::CodeGenFileType FileType) {
  auto Lock = LR.getContext().getLock();
  CompiledObjectFiles.reserve(LR.size());
  for (auto &[LCO, LCOModule] : LR.modules()) {
    auto &[TSM, MMIWP] = LCOModule;
    llvm::SmallVector<char, 0> ObjectFile;
    LUTHIER_RETURN_ON_ERROR(CodeGenerator::printAssembly(
        *TSM.getModuleUnlocked(), LR.getTargetMachine<llvm::GCNTargetMachine>(),
        MMIWP, ObjectFile, FileType));
    CompiledObjectFiles.emplace_back(LCO, ObjectFile);
  }
  return llvm::Error::success();
}

llvm::Error
instrumentAndLoad(hsa_executable_symbol_t Kernel,
                  const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset) {
  // Instrument the lifted representation
  auto InstrumentedLR = CodeGenerator::instance().instrument(LR, Mutator);
  LUTHIER_RETURN_ON_ERROR(InstrumentedLR.takeError());

  // Print the assembly files of the Instrumented LR
  llvm::SmallVector<
      std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>, 1>
      Relocatables;

  LUTHIER_RETURN_ON_ERROR(printLiftedRepresentation(
      **InstrumentedLR, Relocatables, llvm::CodeGenFileType::ObjectFile));

  // Link the object files into executables
  llvm::SmallVector<
      std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>, 1>
      Executables;
  for (const auto &[LCO, Relocatable] : Relocatables) {
    llvm::SmallVector<uint8_t> Executable;
    auto ISA = hsa::LoadedCodeObject(LCO).getISA();
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());
    LUTHIER_RETURN_ON_ERROR(CodeGenerator::linkRelocatableToExecutable(
        Relocatable, *ISA, Executable));
    Executables.emplace_back(LCO, Executable);
  }
  // Create a set of extern variables used in the instrumented code

  auto KernelWrapper = hsa::ExecutableSymbol::fromHandle(Kernel);
  LUTHIER_RETURN_ON_ERROR(KernelWrapper.takeError());

  llvm::StringMap<void *> ExternVariables;
  // set of static variables used in the original kernel itself
  for (const auto &[Symbol, GV] : LR.globals()) {
    auto SymbolWrapper = hsa::ExecutableSymbol::fromHandle(Symbol);
    LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
    ExternVariables.insert({llvm::cantFail(SymbolWrapper->getName()),
                            reinterpret_cast<void *>(llvm::cantFail(
                                SymbolWrapper->getVariableAddress()))});
  }
  auto &TEM = ToolExecutableManager::instance();
  const auto &SIM = TEM.getStaticInstrumentationModule();
  auto Agent = llvm::cantFail(KernelWrapper->getAgent());
  // Set of static variables used in the instrumentation module
  for (const auto &GVName : SIM.getGlobalVariableNames()) {
    auto VarAddress = SIM.getGlobalVariablesLoadedOnAgent(GVName, Agent);
    LUTHIER_RETURN_ON_ERROR(VarAddress.takeError());
    ExternVariables.insert({GVName, reinterpret_cast<void *>(**VarAddress)});
  }
  return TEM.loadInstrumentedKernel(Executables, *KernelWrapper, Preset,
                                    ExternVariables);
}

llvm::Error
instrumentAndLoad(hsa_executable_t Exec, const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset) {
  // Instrument the lifted representation
  auto InstrumentedLR = CodeGenerator::instance().instrument(LR, Mutator);
  LUTHIER_RETURN_ON_ERROR(InstrumentedLR.takeError());

  // Print the assembly files of the Instrumented LR
  llvm::SmallVector<
      std::pair<hsa_loaded_code_object_t, llvm::SmallVector<char, 0>>, 1>
      Relocatables;

  LUTHIER_RETURN_ON_ERROR(printLiftedRepresentation(
      **InstrumentedLR, Relocatables, llvm::CodeGenFileType::ObjectFile));

  // Link the object files into executables
  llvm::SmallVector<
      std::pair<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>, 1>
      Executables;
  for (const auto &[LCO, Relocatable] : Relocatables) {
    llvm::SmallVector<uint8_t> Executable;
    auto ISA = hsa::LoadedCodeObject(LCO).getISA();
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());
    LUTHIER_RETURN_ON_ERROR(CodeGenerator::linkRelocatableToExecutable(
        Relocatable, *ISA, Executable));
    Executables.emplace_back(LCO, Executable);
  }
  // Create a set of extern variables used in the instrumented code
  hsa::Executable ExecWrapper(Exec);

  llvm::SmallVector<std::tuple<hsa::GpuAgent, llvm::StringRef, void *>>
      ExternVariables;
  // set of static variables used in the original kernel itself
  for (const auto &[Symbol, GV] : LR.globals()) {
    auto SymbolWrapper = hsa::ExecutableSymbol::fromHandle(Symbol);
    LUTHIER_RETURN_ON_ERROR(SymbolWrapper.takeError());
    ExternVariables.emplace_back(llvm::cantFail(SymbolWrapper->getAgent()),
                                 llvm::cantFail(SymbolWrapper->getName()),
                                 reinterpret_cast<void *>(llvm::cantFail(
                                     SymbolWrapper->getVariableAddress())));
  }
  // Get all the Agents involved in this executable
  llvm::SmallDenseSet<hsa::GpuAgent, 4> Agents;
  for (const auto &[LCO, ELF] : Executables) {
    Agents.insert(llvm::cantFail(LCO.getAgent()));
  }
  auto &TEM = ToolExecutableManager::instance();
  const auto &SIM = TEM.getStaticInstrumentationModule();
  // Set of static variables used in the instrumentation module
  for (const auto &GVName : SIM.getGlobalVariableNames()) {
    for (const auto &Agent : Agents) {
      auto VarAddress = SIM.getGlobalVariablesLoadedOnAgent(GVName, Agent);
      LUTHIER_RETURN_ON_ERROR(VarAddress.takeError());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(VarAddress->has_value()));
      ExternVariables.emplace_back(Agent, GVName,
                                   reinterpret_cast<void *>(**VarAddress));
    }
  }
  return TEM.loadInstrumentedExecutable(Executables, Preset, ExternVariables);
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
rocprofiler_dim3_t convertToRocprofilerDim3(const dim3 &d) {
  return rocprofiler_dim3_t{d.x, d.y, d.z};
}

} // namespace luthier
