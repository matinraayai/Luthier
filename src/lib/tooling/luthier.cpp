//===-- luthier.cpp - Implementation of the Public Luthier API ------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// <tt>luthier.h</tt>.
/// For the controller logic of Luthier, see <tt>luthier::Controller</tt>.
//===----------------------------------------------------------------------===//
#include "luthier/luthier.h"
#include "hip/HipCompilerApiInterceptor.hpp"
#include "hip/HipRuntimeApiInterceptor.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "luthier/hsa/Instr.h"
#include "luthier/tooling/InstrumentationTask.h"
#include "tooling_common/CodeGenerator.hpp"
#include "tooling_common/CodeLifter.hpp"
#include "tooling_common/ToolExecutableLoader.hpp"
#include <llvm/ADT/StringRef.h>
#include <optional>

namespace luthier {

namespace hip {

const HipCompilerDispatchTable &getSavedCompilerTable() {
  return HipCompilerApiInterceptor::instance().getSavedApiTableContainer();
}

const HipDispatchTable &getSavedDispatchTable() {
  return HipRuntimeApiInterceptor::instance().getSavedApiTableContainer();
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

llvm::Expected<std::unique_ptr<LiftedRepresentation>>
instrument(const LiftedRepresentation &LR,
           llvm::function_ref<llvm::Error(InstrumentationTask &,
                                          LiftedRepresentation &)>
               Mutator) {
  return CodeGenerator::instance().instrument(LR, Mutator);
}

llvm::Error
printLiftedRepresentation(LiftedRepresentation &LR,
                          llvm::SmallVectorImpl<char> &CompiledObjectFile,
                          llvm::CodeGenFileType FileType) {
  auto Lock = LR.getLock();
  LUTHIER_RETURN_ON_ERROR(CodeGenerator::printAssembly(
      LR.getModule(), LR.getTM(), LR.getMMIWP(), CompiledObjectFile, FileType));
  return llvm::Error::success();
}

llvm::Error
instrumentAndLoad(const hsa::LoadedCodeObjectKernel &Kernel,
                  const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset) {
  auto Lock = LR.getLock();
  // Instrument the lifted representation
  auto InstrumentedLR = CodeGenerator::instance().instrument(LR, Mutator);
  LUTHIER_RETURN_ON_ERROR(InstrumentedLR.takeError());

  // Print the assembly file of the Instrumented LR
  llvm::SmallVector<char> Relocatable;

  LUTHIER_RETURN_ON_ERROR(printLiftedRepresentation(
      **InstrumentedLR, Relocatable, llvm::CodeGenFileType::ObjectFile));

  // Link the object file into executables
  llvm::SmallVector<uint8_t> Executable;
  auto ISA = hsa::LoadedCodeObject(LR.getLoadedCodeObject()).getISA();
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());
  LUTHIER_RETURN_ON_ERROR(CodeGenerator::linkRelocatableToExecutable(
      Relocatable, *ISA, Executable));
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
  return TEM.loadInstrumentedKernel(Executable, Kernel, Preset,
                                    ExternVariables);
}

llvm::Expected<bool>
isKernelInstrumented(const hsa::LoadedCodeObjectKernel &Kernel,
                     llvm::StringRef Preset) {
  return ToolExecutableLoader::instance().isKernelInstrumented(Kernel, Preset);
}

llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                                     llvm::StringRef Preset) {
  auto Symbol = luthier::hsa::LoadedCodeObjectSymbol::fromLoadedAddress(
      Packet.kernel_object);
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      *Symbol != nullptr, "Failed to locate the kernel symbol of the dispatch "
                          "packet from its kernel_object field."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      llvm::isa<hsa::LoadedCodeObjectKernel>(**Symbol),
      "The dispatch packet kernel object does not point to a kernel symbol."));

  auto InstrumentedKernel =
      luthier::ToolExecutableLoader::instance().getInstrumentedKernel(
          *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Symbol.get().get()),
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
