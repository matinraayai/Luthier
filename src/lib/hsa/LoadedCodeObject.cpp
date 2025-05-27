//===-- LoadedCodeObject.cpp ----------------------------------------------===//
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
/// Implements commonly used functionality for the \c hsa_loaded_code_object
/// type in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/LoadedCodeObject.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Expected<hsa_executable_t>
getLCOExecutable(hsa_loaded_code_object_t LCO,
                 const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
                     &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE, &Exec)));
  return Exec;
}

llvm::Expected<hsa_agent_t>
getLCOAgent(hsa_loaded_code_object_t LCO,
            const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
                &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &Agent)));
  return Agent;
}

llvm::Expected<long>
getLCOLoadDelta(hsa_loaded_code_object_t LCO,
                const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
                    &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {
  long LoadDelta;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
          &LoadDelta)));
  return LoadDelta;
}

llvm::Expected<llvm::ArrayRef<uint8_t>> getLCOLoadedMemory(
    hsa_loaded_code_object_t LCO,
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {
  uint64_t LoadBase;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
          &LoadBase)));

  uint64_t LoadSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
          &LoadSize)));

  return llvm::ArrayRef{reinterpret_cast<uint8_t *>(LoadBase), LoadSize};
}

llvm::Expected<std::string>
getLCOUri(hsa_loaded_code_object_t LCO,
          const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
              &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {
  unsigned int UriLength;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH, &UriLength)));

  std::string URI;
  URI.resize(UriLength);
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
          URI.data())));

  return URI;
}

llvm::Expected<llvm::ArrayRef<uint8_t>> getLCOStorageMemory(
    hsa_loaded_code_object_t LCO,
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) {
  uint64_t StorageBase;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
          &StorageBase)));

  uint64_t StorageSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaVenAmdLoaderLoadedCodeObjectGetInfoFn(
          LCO,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
          &StorageSize)));

  return llvm::ArrayRef{reinterpret_cast<uint8_t *>(StorageBase), StorageSize};
}

} // namespace luthier::hsa
