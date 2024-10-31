//===-- luthier.cpp - Implementation of the Luthier API -------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

#include "common/Error.hpp"
#include "hip/HipCompilerApiInterceptor.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "luthier/InstrumentationTask.h"
#include "tooling_common/CodeGenerator.hpp"
#include "tooling_common/CodeLifter.hpp"
#include "tooling_common/ToolExecutableLoader.hpp"
#include <luthier/hsa/Instr.h>

namespace luthier {

namespace hip {

const HipCompilerDispatchTable &getSavedCompilerTable() {
  return hip::HipCompilerApiInterceptor::instance().getSavedApiTableContainer();
}

} // namespace hip

namespace hsa {

void setAtHsaApiEvtCallback(
    const std::function<void(ApiEvtArgs *, ApiEvtPhase, ApiEvtID)> &Callback) {
  hsa::HsaRuntimeInterceptor::instance().setUserCallback(Callback);
}

const HsaApiTable &getHsaApiTable() {
  return hsa::HsaRuntimeInterceptor::instance()
      .getSavedApiTableContainer()
      .root;
}

const hsa_ven_amd_loader_1_03_pfn_s &getHsaVenAmdLoaderTable() {
  return hsa::HsaRuntimeInterceptor::instance().getHsaVenAmdLoaderTable();
}

llvm::Error enableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID) {
  return hsa::HsaRuntimeInterceptor::instance().enableUserCallback(ApiID);
}

llvm::Error disableHsaApiEvtIDCallback(hsa::ApiEvtID ApiID) {
  return hsa::HsaRuntimeInterceptor::instance().disableUserCallback(ApiID);
}

} // namespace hsa

llvm::Expected<llvm::ArrayRef<hsa::Instr>>
disassemble(const hsa::LoadedCodeObjectKernel &Kernel) {
  return luthier::CodeLifter::instance().disassemble(Kernel);
}

llvm::Expected<llvm::ArrayRef<hsa::Instr>>
disassemble(const hsa::LoadedCodeObjectDeviceFunction &Func) {
  return luthier::CodeLifter::instance().disassemble(Func);
}

llvm::Expected<const luthier::LiftedRepresentation &>
lift(const hsa::LoadedCodeObjectKernel &Kernel) {
  return CodeLifter::instance().lift(Kernel);
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
  auto Lock = LR.getLock();
  CompiledObjectFiles.reserve(LR.size());
  for (auto &[LCO, ModuleAndMMIWP] : LR.modules()) {
    auto &[Module, MMIWP] = ModuleAndMMIWP;
    llvm::SmallVector<char, 0> ObjectFile;
    LUTHIER_RETURN_ON_ERROR(CodeGenerator::printAssembly(
        *Module, *LR.getTM(LCO), MMIWP, ObjectFile, FileType));
    CompiledObjectFiles.emplace_back(LCO, ObjectFile);
  }
  return llvm::Error::success();
}

llvm::Error
instrumentAndLoad(const hsa::LoadedCodeObjectKernel &Kernel,
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

  llvm::StringMap<const void *> ExternVariables;
  // set of static variables used in the original kernel itself
  for (const auto &[Symbol, GV] : LR.globals()) {
    ExternVariables.insert({llvm::cantFail(Symbol->getName()),
                            reinterpret_cast<const void *>(llvm::cantFail(
                                Symbol->getLoadedSymbolAddress()))});
  }
  auto &TEM = ToolExecutableLoader::instance();
  const auto &SIM = TEM.getStaticInstrumentationModule();
  auto Agent = llvm::cantFail(Kernel.getAgent());
  // Set of static variables used in the instrumentation module
  for (const auto &GVName : SIM.gv_names()) {
    auto VarAddress =
        SIM.getGlobalVariablesLoadedOnAgent(GVName, hsa::GpuAgent(Agent));
    LUTHIER_RETURN_ON_ERROR(VarAddress.takeError());
    ExternVariables.insert({GVName, reinterpret_cast<void *>(**VarAddress)});
  }
  return TEM.loadInstrumentedKernel(Executables, Kernel, Preset,
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

  llvm::SmallVector<std::tuple<hsa::GpuAgent, llvm::StringRef, const void *>>
      ExternVariables;
  // set of static variables used in the original kernel itself
  for (const auto &[Symbol, GV] : LR.globals()) {
    ExternVariables.emplace_back(
        hsa::GpuAgent(llvm::cantFail(Symbol->getAgent())),
        llvm::cantFail(Symbol->getName()),
        reinterpret_cast<const void *>(
            llvm::cantFail(Symbol->getLoadedSymbolAddress())));
  }
  // Get all the Agents involved in this executable
  llvm::SmallDenseSet<hsa::GpuAgent, 4> Agents;
  for (const auto &[LCO, ELF] : Executables) {
    Agents.insert(llvm::cantFail(LCO.getAgent()));
  }
  auto &TEM = ToolExecutableLoader::instance();
  const auto &SIM = TEM.getStaticInstrumentationModule();
  // Set of static variables used in the instrumentation module
  for (const auto &GVName : SIM.gv_names()) {
    for (const auto &Agent : Agents) {
      auto VarAddress = SIM.getGlobalVariablesLoadedOnAgent(GVName, Agent);
      LUTHIER_RETURN_ON_ERROR(VarAddress.takeError());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          VarAddress->has_value(),
          "Failed to get the global variable associated with {0} "
          "on agent {1:x}.",
          GVName, Agent.hsaHandle()));
      ExternVariables.emplace_back(Agent, GVName,
                                   reinterpret_cast<void *>(**VarAddress));
    }
  }
  return TEM.loadInstrumentedExecutable(Executables, Preset, ExternVariables);
}

llvm::Expected<bool>
isKernelInstrumented(const hsa::LoadedCodeObjectKernel &Kernel,
                     llvm::StringRef Preset) {
  return ToolExecutableLoader::instance().isKernelInstrumented(Kernel, Preset);
}

llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                                     llvm::StringRef Preset) {
  auto *Symbol = llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(
      luthier::hsa::LoadedCodeObjectSymbol::fromLoadedAddress(
          Packet.kernel_object));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      Symbol != nullptr, "Failed to locate the kernel symbol of the dispatch "
                         "packet from its kernel_object field."));

  auto InstrumentedKernel =
      luthier::ToolExecutableLoader::instance().getInstrumentedKernel(*Symbol,
                                                                      Preset);
  LUTHIER_RETURN_ON_ERROR(InstrumentedKernel.takeError());
  auto InstrumentedKD = InstrumentedKernel->getKernelDescriptor();

  LUTHIER_RETURN_ON_ERROR(InstrumentedKD.takeError());

  Packet.kernel_object = reinterpret_cast<uint64_t>(*InstrumentedKD);

  auto InstrumentedKernelMD = InstrumentedKernel->getKernelMetadata();

  Packet.private_segment_size = InstrumentedKernelMD.PrivateSegmentFixedSize;

  return llvm::Error::success();
}

} // namespace luthier
