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
/// Implements Luthier's HSA Loaded Instrumentation Module and its variants.
//===----------------------------------------------------------------------===//
#include <llvm/Bitcode/BitcodeWriter.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/hsa/Executable.h>
#include <luthier/hsa/LoadedCodeObject.h>
#include <luthier/hsa/LoadedInstrumentationModule.h>
#include <luthier/hsa/ToolExecutableLoader.h>
#include <luthier/object/ELFObjectUtils.h>

namespace luthier::hsa {

llvm::Expected<bool> LoadedInstrumentationModule::isLoaded() const {
  llvm::Expected<hsa_executable_state_t> ExecStateOrErr =
      hsa::executableGetState(Exec, HsaExecutableGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(ExecStateOrErr.takeError());
  return *ExecStateOrErr == HSA_EXECUTABLE_STATE_FROZEN;
}

llvm::Expected<llvm::StringMap<uint64_t>>
LoadedInstrumentationModule::getSymbolLoadAddressesMap() const {
  llvm::Expected<bool> IsLoadedOrErr = isLoaded();
  LUTHIER_RETURN_ON_ERROR(IsLoadedOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
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

DynamicallyLoadedInstrumentationModule::
    ~DynamicallyLoadedInstrumentationModule() {
  LUTHIER_REPORT_FATAL_ON_ERROR(
      hsa::executableDestroy(Exec, HsaExecutableDestroyFn));
}

} // namespace luthier::hsa