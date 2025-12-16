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
#include "luthier/Comgr/Comgr.h"
#include "luthier/HSA/Instr.h"
#include "luthier/Tooling/CodeGenerator.h"
#include "luthier/Tooling/CodeLifter.h"
#include "luthier/Tooling/Context.h"
#include "luthier/Tooling/InstrumentationTask.h"
#include "luthier/Tooling/ToolExecutableLoader.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <optional>

namespace luthier {

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
  const auto &LoaderApiTable = Context::instance().getHsaLoaderTable();
  auto Lock = LR.getLock();
  // Instrument the lifted representation
  auto InstrumentedLR = CodeGenerator::instance().instrument(LR, Mutator);
  LUTHIER_RETURN_ON_ERROR(InstrumentedLR.takeError());

  // Print the assembly file of the Instrumented LR
  llvm::SmallVector<char> Relocatable;

  LUTHIER_RETURN_ON_ERROR(printLiftedRepresentation(
      **InstrumentedLR, Relocatable, llvm::CodeGenFileType::ObjectFile));

  // Link the object file into executables
  llvm::SmallVector<char> Executable;
  LUTHIER_RETURN_ON_ERROR(
      comgr::linkRelocatableToExecutable(Relocatable, Executable));
  // Create a set of extern variables used in the instrumented code

  llvm::StringMap<const void *> ExternVariables;
  // set of static variables used in the original kernel itself
  for (const auto &[Symbol, GV] : LR.globals()) {
    ExternVariables.insert(
        {llvm::cantFail(Symbol->getName()),
         reinterpret_cast<const void *>(
             llvm::cantFail(Symbol->getLoadedSymbolAddress(LoaderApiTable)))});
  }
  auto &TEM = ToolExecutableLoader::instance();
  const auto &SIM = TEM.getStaticInstrumentationModule();
  auto Agent = llvm::cantFail(Kernel.getAgent(LoaderApiTable));
  // Set of static variables used in the instrumentation module
  for (const auto &GVName : SIM.gv_names()) {
    auto VarAddress = SIM.getGlobalVariablesLoadedOnAgent(GVName, Agent);
    LUTHIER_RETURN_ON_ERROR(VarAddress.takeError());
    ExternVariables.insert({GVName, reinterpret_cast<void *>(**VarAddress)});
  }
  return TEM.loadInstrumentedKernel(
      llvm::ArrayRef(reinterpret_cast<uint8_t *>(Executable.data()),
                     Executable.size()),
      Kernel, Preset, ExternVariables);
}

llvm::Expected<bool>
isKernelInstrumented(const hsa::LoadedCodeObjectKernel &Kernel,
                     llvm::StringRef Preset) {
  return ToolExecutableLoader::instance().isKernelInstrumented(Kernel, Preset);
}

llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                                     llvm::StringRef Preset) {
  luthier::Context &C = Context::instance();
  auto CoreApiTable = C.getHsaCoreTable();
  const auto &LoaderApiTable = C.getHsaLoaderTable();
  auto Symbol = luthier::hsa::LoadedCodeObjectSymbol::fromLoadedAddress(
      CoreApiTable, LoaderApiTable, Packet.kernel_object);
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      *Symbol != nullptr, "Failed to locate the kernel symbol of the dispatch "
                          "packet from its kernel_object field."));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      llvm::isa<hsa::LoadedCodeObjectKernel>(**Symbol),
      "The dispatch packet kernel object does not point to a kernel symbol."));

  auto InstrumentedKernel =
      luthier::ToolExecutableLoader::instance().getInstrumentedKernel(
          *(*Symbol)->getExecutableSymbol(), Preset);
  LUTHIER_RETURN_ON_ERROR(InstrumentedKernel.takeError());
  auto &[InstrumentedSymbol, InstrumentedMD] = *InstrumentedKernel;

  llvm::Expected<uint64_t> InstrumentedKD =
      hsa::executableSymbolGetAddress(CoreApiTable, InstrumentedSymbol);
  LUTHIER_RETURN_ON_ERROR(InstrumentedKD.takeError());

  Packet.kernel_object = *InstrumentedKD;

  Packet.private_segment_size = InstrumentedMD.PrivateSegmentFixedSize;

  return llvm::Error::success();
}

} // namespace luthier
