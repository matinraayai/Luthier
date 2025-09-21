//===-- LoadedCodeObject.h --------------------------------------*- C++ -*-===//
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
/// Defines a set of commonly used functionality for the
/// \c hsa_loaded_code_object_t handle type in HSA with the following
/// assumptions:
/// 1. Even though the ROCr HSA vendor loader API
/// (under <tt><hsa/hsa_ven_amd_loader.h></tt>) acknowledge that both
/// file-backed and memory-backed Loaded Code Objects exists, only memory-backed
/// ones are actually implemented. Therefore, querying the storage type and
/// querying the FD of the storage is not included in the API. Luthier assumes
/// all Loaded Code Objects have a memory storage in order to return its
/// associated ELF. In the event that file-backed storage is implemented in the
/// loader, the code needs to be updated.
/// 2. Program Loaded Code Objects has been deprecated and are not used anywhere
/// in the ROCm stack. ROCr does not even allow using Loaded Code Objects with
/// program allocations. Therefore, it is safe to assume all Loaded Code Objects
/// are backed by a \c hsa_agent_t of type <tt>HSA_DEVICE_TYPE_GPU</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_H
#define LUTHIER_HSA_LOADED_CODE_OBJECT_H
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/ApiTable.h"
#include "luthier/hsa/HsaError.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

/// Queries the \c hsa_executable_t that contains the \p LCO
/// \param LoaderApiTable the HSA loader API table container used to dispatch
/// HSA loader methods
/// \param LCO the \c hsa_loaded_code_object_t being queried
/// \return Expects the \c hsa_executable_t of \p LCO on success
/// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE
template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
[[maybe_unused]] [[nodiscard]] llvm::Expected<hsa_executable_t>
loadedCodeObjectGetExecutable(LoaderTableType &LoaderApiTable,
                              hsa_loaded_code_object_t LCO) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE, &Exec),
      llvm::formatv(
          "Failed to obtain the executable of HSA loaded code object {0:x}.",
          LCO.handle)));
  return Exec;
}

/// Queries the \c hsa_agent_t associated with the \p LCO
/// \param LoaderApiTable the HSA loader API table container used to dispatch
/// HSA loader methods
/// \param LCO the \c hsa_loaded_code_object_t being queried
/// \note As Loaded Code Objects of program allocation are deprecated in ROCr,
/// it is safe to assume all Loaded Code Objects have agent allocation, and
/// therefore, are backed by an HSA GPU Agent
/// \return Expects the \c hsa_agent_t of the \p LCO on success
/// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT
template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
[[nodiscard]] llvm::Expected<hsa_agent_t>
loadedCodeObjectGetAgent(LoaderTableType &LoaderApiTable,
                         hsa_loaded_code_object_t LCO) {
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &Agent),
      llvm::formatv(
          "Failed to get the GPU agent of HSA loaded code object {0:x}",
          LCO.handle)));
  return Agent;
}

/// Queries the load delta of the \p LCO
/// \param LoaderApiTable the HSA loader API table container used to dispatch
/// HSA loader methods
/// \param LCO the \c hsa_loaded_code_object_t being queried
/// \return Expects the Load Delta of this Loaded Code Object on success
/// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA
template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
[[nodiscard]] llvm::Expected<long>
loadedCodeObjectGetLoadDelta(const LoaderTableType &LoaderApiTable,
                             hsa_loaded_code_object_t LCO) {
  long LoadDelta;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
          &LoadDelta),
      llvm::formatv("Failed to obtain the load delta of LCO {0:x}",
                    LCO.handle)));
  return LoadDelta;
}

/// Queries the entire loaded memory range of the \p LCO
/// \param LoaderApiTable the HSA loader API table container used to dispatch
/// HSA loader methods
/// \param LCO the \c hsa_loaded_code_object_t being queried
/// \return Expects the \c llvm::ArrayRef to the portion of GPU memory that
/// this code object has been loaded onto on success
/// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE
/// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE
template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
[[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
loadedCodeObjectGetLoadedMemory(LoaderTableType &LoaderApiTable,
                                hsa_loaded_code_object_t LCO) {
  uint64_t LoadBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &LoadBase),
      llvm::formatv(
          "Failed to get the load base address of loaded code object {0:x}",
          LCO.handle)));

  uint64_t LoadSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &LoadSize),
      llvm::formatv("Failed to get the load size of loaded code object {0:x}",
                    LCO.handle)));

  return llvm::ArrayRef{reinterpret_cast<uint8_t *>(LoadBase), LoadSize};
}

/// \param LCO the \c hsa_loaded_code_object_t being queried
/// \param LoaderApiTable the HSA loader API table container used to dispatch
/// HSA loader methods
/// \return Expects the URI describing the origins of the \p LCO
/// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH
/// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI
template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
[[nodiscard]] llvm::Expected<std::string>
loadedCodeObjectGetURI(LoaderTableType &LoaderApiTable,
                       hsa_loaded_code_object_t LCO) {
  unsigned int UriLength;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
          &UriLength),
      llvm::formatv("Failed to get the URI size of loaded code object {0:x}",
                    LCO.handle)));

  std::string URI;
  URI.resize(UriLength);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI, URI.data()),
      llvm::formatv("Failed to get the URI of loaded code object {0:x}",
                    LCO.handle)));

  return URI;
}

/// Queries where the host copy of the <tt>LCO</tt>'s ELF is
/// stored, and its size from HSA
/// \param LoaderApiTable the HSA loader API table container used to dispatch
/// HSA loader methods
/// \param LCO the \c hsa_loaded_code_object_t being queried
/// \return Expects the \c llvm::ArrayRef pointing to the beginning and the
/// end of the storage memory on success
template <typename LoaderTableType = hsa_ven_amd_loader_1_01_pfn_t>
[[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
loadedCodeObjectGetStorageMemory(const LoaderTableType &LoaderApiTable,
                                 hsa_loaded_code_object_t LCO) {
  uint64_t StorageBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
          &StorageBase),
      llvm::formatv(
          "Failed to get the storage memory base of loaded code object {0:x}",
          LCO.handle)));

  uint64_t StorageSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApiTable.hsa_ven_amd_loader_loaded_code_object_get_info(
          LCO,
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
          &StorageSize),
      llvm::formatv(
          "Failed to get the storage memory size loaded code object {0:x}",
          LCO.handle)));

  return llvm::ArrayRef{reinterpret_cast<uint8_t *>(StorageBase), StorageSize};
}

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_loaded_code_object_t> {
  static hsa_loaded_code_object_t getEmptyKey() {
    return hsa_loaded_code_object_t({DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getEmptyKey()});
  }

  static hsa_loaded_code_object_t getTombstoneKey() {
    return hsa_loaded_code_object_t({DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const hsa_loaded_code_object_t &LCO) {
    return DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getHashValue(LCO.handle);
  }

  static bool isEqual(const hsa_loaded_code_object_t &Lhs,
                      const hsa_loaded_code_object_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
}; // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_loaded_code_object_t> {
  size_t operator()(const hsa_loaded_code_object_t &Obj) const noexcept {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct less<hsa_loaded_code_object_t> {
  bool operator()(const hsa_loaded_code_object_t &Lhs,
                  const hsa_loaded_code_object_t &Rhs) const {
    return Lhs.handle < Rhs.handle;
  }
};

template <> struct less_equal<hsa_loaded_code_object_t> {
  bool operator()(const hsa_loaded_code_object_t &Lhs,
                  const hsa_loaded_code_object_t &Rhs) const {
    return Lhs.handle <= Rhs.handle;
  }
};

template <> struct equal_to<hsa_loaded_code_object_t> {
  bool operator()(const hsa_loaded_code_object_t &Lhs,
                  const hsa_loaded_code_object_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

template <> struct not_equal_to<hsa_loaded_code_object_t> {
  bool operator()(const hsa_loaded_code_object_t &Lhs,
                  const hsa_loaded_code_object_t &Rhs) const {
    return Lhs.handle != Rhs.handle;
  }
};

template <> struct greater<hsa_loaded_code_object_t> {
  bool operator()(const hsa_loaded_code_object_t &Lhs,
                  const hsa_loaded_code_object_t &Rhs) const {
    return Lhs.handle > Rhs.handle;
  }
};

template <> struct greater_equal<hsa_loaded_code_object_t> {
  bool operator()(const hsa_loaded_code_object_t &Lhs,
                  const hsa_loaded_code_object_t &Rhs) const {
    return Lhs.handle >= Rhs.handle;
  }
};

} // namespace std

#endif