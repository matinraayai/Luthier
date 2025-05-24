//===-- InstrumentationModule.cpp - Luthier Instrumentation Module --------===//
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
/// This file implements Luthier's Instrumentation Module and its variants.
//===----------------------------------------------------------------------===//
#include "tooling_common/InstrumentationModule.hpp"
#include "hsa/CodeObjectReader.hpp"
#include "hsa/Executable.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include "luthier/consts.h"
#include "luthier/object/ELFObjectUtils.h"
#include "luthier/object/ObjectUtils.h"
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>

namespace {

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
getBCBufferInObjectFile(const llvm::object::ObjectFile &ObjFile) {
  // Find the ".llvmbc" section of the ELF
  for (const llvm::object::SectionRef &Section : ObjFile.sections()) {
    auto SectionName = Section.getName();
    LUTHIER_RETURN_ON_ERROR(SectionName.takeError());
    if (*SectionName == luthier::IModuleBCSectionName) {
      auto SectionContents = Section.getContents();
      LUTHIER_RETURN_ON_ERROR(SectionContents.takeError());
      return llvm::MemoryBuffer::getMemBuffer(*SectionContents, "", false);
    }
  }
  return nullptr;
}

} // namespace

namespace luthier {

llvm::Expected<std::unique_ptr<InstrumentationModule>>
InstrumentationModule::create(std::vector<uint8_t> CodeObject) {

  std::unique_ptr<llvm::object::ObjectFile> ObjFile;
  LUTHIER_RETURN_ON_ERROR(
      luthier::object::createObjectFile(llvm::toStringRef(CodeObject))
          .moveInto(ObjFile));

  // Confirm that the ELF contains a Luthier bitcode section
  std::unique_ptr<llvm::MemoryBuffer> BCBuffer;
  LUTHIER_RETURN_ON_ERROR(getBCBufferInObjectFile(*ObjFile).moveInto(BCBuffer));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      BCBuffer != nullptr, "Failed to find the Luthier bitcode section inside "
                           "the instrumentation module"));

  return std::make_unique<InstrumentationModule>(
      std::move(CodeObject), std::move(ObjFile), std::move(BCBuffer));
}

llvm::Error LoadedInstrumentationModule::readManifest(
    llvm::StringMap<uint64_t> &Manifest) const {
  hsa::LoadedCodeObject LCOWrapper(LCO);
  llvm::Expected<llvm::ArrayRef<uint8_t>> LCOLoadedMemOrErr =
      LCOWrapper.getLoadedMemory(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(LCOLoadedMemOrErr.takeError());
  auto LCOLoadBase = reinterpret_cast<uint64_t>(LCOLoadedMemOrErr->data());

  LUTHIER_RETURN_ON_ERROR(IModule->readManifest(Manifest));

  for (const llvm::object::SymbolRef &Symbol : IModule->getObject().symbols()) {
    llvm::Expected<llvm::StringRef> SymNamOrErr = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymNamOrErr.takeError());
    Manifest[*SymNamOrErr] += LCOLoadBase;
  }

  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<HipLoadedInstrumentationModule>>
HipLoadedInstrumentationModule::getIfHipLoadedIModule(
    hsa_loaded_code_object_t LCO,
    decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &LoadedCodeObjectGetInfoFun) {
  hsa::LoadedCodeObject LCOWrapper(LCO);
  // Get the LCO storage memory and parse its ELF
  llvm::ArrayRef<uint8_t> StorageMemory;
  LUTHIER_RETURN_ON_ERROR(
      LCOWrapper.getStorageMemory(LoadedCodeObjectGetInfoFun)
          .moveInto(StorageMemory));

  // Parse the ELF
  std::unique_ptr<llvm::object::ObjectFile> ObjFile;
  LUTHIER_RETURN_ON_ERROR(
      luthier::object::createObjectFile(llvm::toStringRef(StorageMemory))
          .moveInto(ObjFile));

  // Check for the Luthier reserved managed variable inside the object
  std::optional<llvm::object::ELFSymbolRef>
      LuthierReservedManagedVariableIfPresent{};
  LUTHIER_RETURN_ON_ERROR(
      luthier::object::lookupSymbolByName(
          llvm::cast<llvm::object::ELF64LEObjectFile>(*ObjFile),
          StaticIModuleReservedManagedVar, false)
          .moveInto(LuthierReservedManagedVariableIfPresent));

  if (LuthierReservedManagedVariableIfPresent.has_value()) {

    std::unique_ptr<InstrumentationModule> IModule;
    LUTHIER_RETURN_ON_ERROR(
        InstrumentationModule::create(StorageMemory).moveInto(IModule));

    return std::make_unique<HipLoadedInstrumentationModule>(
        LCO, LoadedCodeObjectGetInfoFun, std::move(IModule));
  }
  return nullptr;
}

llvm::Expected<std::unique_ptr<DynamicallyLoadedInstrumentationModule>>
DynamicallyLoadedInstrumentationModule::loadInstrumentationModule(
    std::vector<uint8_t> CodeObject, hsa_agent_t Agent,
    const decltype(hsa_executable_create_alt) &HsaExecutableCreateAltFn,
    const decltype(hsa_code_object_reader_create_from_memory)
        &HsaCodeObjectReaderCreateFromMemory,
    const decltype(hsa_executable_load_agent_code_object)
        &HsaExecutableLoadAgentCodeObjectFn,
    const decltype(hsa_executable_freeze) &HsaExecutableFreezeFn,
    const decltype(hsa_code_object_reader_destroy)
        &HsaCodeObjectReaderDestroyFn,
    const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn,
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
            &LoadedCodeObjectGetInfoFun) {
  std::unique_ptr<InstrumentationModule> IModule;
  LUTHIER_RETURN_ON_ERROR(
      InstrumentationModule::create(std::move(CodeObject)).moveInto(IModule));

  // Create the executable
  hsa::Executable Executable{hsa_executable_t{}};
  LUTHIER_RETURN_ON_ERROR(
      hsa::Executable::create(HsaExecutableCreateAltFn).moveInto(Executable));

  // Load the code objects into the executable
  hsa::CodeObjectReader Reader{hsa_code_object_reader_t{}};
  LUTHIER_RETURN_ON_ERROR(
      hsa::CodeObjectReader::createFromMemory(
          HsaCodeObjectReaderCreateFromMemory,
          IModule->getObject().getMemoryBufferRef().getBuffer())
          .moveInto(Reader));

  hsa::LoadedCodeObject LCO{hsa_loaded_code_object_t{}};
  LUTHIER_RETURN_ON_ERROR(
      Executable
          .loadAgentCodeObject(HsaExecutableLoadAgentCodeObjectFn, Reader,
                               hsa::GpuAgent(Agent))
          .moveInto(LCO));
  // Freeze the executable
  LUTHIER_RETURN_ON_ERROR(Executable.freeze(HsaExecutableFreezeFn));

  // Destroy the code object reader
  LUTHIER_RETURN_ON_ERROR(Reader.destroy(HsaCodeObjectReaderDestroyFn));

  return std::make_unique<DynamicallyLoadedInstrumentationModule>(
      Executable.asHsaType(), LCO.asHsaType(), LoadedCodeObjectGetInfoFun,
      std::move(IModule), HsaExecutableDestroyFn);
}

DynamicallyLoadedInstrumentationModule::
    ~DynamicallyLoadedInstrumentationModule() {
  LUTHIER_REPORT_FATAL_ON_ERROR(
      hsa::Executable(Exec).destroy(HsaExecutableDestroyFn));
}

} // namespace luthier