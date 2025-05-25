//===-- LoadedInstrumentationModule.cpp -----------------------------------===//
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
/// This file implements Luthier's Loaded Instrumentation Module and its
/// variants.
//===----------------------------------------------------------------------===//
#include "luthier/runtime/LoadedInstrumentationModule.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/consts.h"
#include "luthier/hsa/Executable.h"
#include "luthier/hsa/LoadedCodeObject.h"
#include "luthier/object/ELFObjectUtils.h"
#include "luthier/object/ObjectUtils.h"
#include "luthier/runtime/ToolExecutableLoader.h"

#include <llvm/Bitcode/BitcodeWriter.h>

namespace luthier {

llvm::Expected<bool> LoadedInstrumentationModule::isLoaded(
    const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn) const {
  llvm::Expected<hsa_executable_state_t> ExecStateOrErr =
      hsa::getExecState(Exec, HsaExecutableGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(ExecStateOrErr.takeError());
  return *ExecStateOrErr == HSA_EXECUTABLE_STATE_FROZEN;
}

llvm::Expected<llvm::StringMap<uint64_t>>
LoadedInstrumentationModule::getSymbolLoadAddressesMap(
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn,
    const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn) const {
  llvm::Expected<bool> IsLoadedOrErr = isLoaded(HsaExecutableGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(IsLoadedOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      *IsLoadedOrErr, "The instrumentation module has not been loaded yet."));
  llvm::Expected<llvm::ArrayRef<uint8_t>> LCOLoadedMemOrErr =
      hsa::getLCOLoadedMemory(LCO, HsaVenAmdLoaderLoadedCodeObjectGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(LCOLoadedMemOrErr.takeError());
  auto LCOLoadBase = reinterpret_cast<uint64_t>(LCOLoadedMemOrErr->data());

  llvm::StringMap<uint64_t> Out;
  LUTHIER_RETURN_ON_ERROR(IModule->getSymbolLoadOffsetsMap().moveInto(Out));

  for (auto &It : Out) {
    It.getValue() += LCOLoadBase;
  }

  return Out;
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