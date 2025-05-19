//===-- LoadedCodeObject.hpp - HSA Loaded Code Object Wrapper -------------===//
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
/// Defines the \c LoadedCodeObject class, a wrapper around
/// the \c hsa_loaded_code_object_t HSA type.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_HPP
#define LUTHIER_HSA_LOADED_CODE_OBJECT_HPP
#include "HandleType.hpp"
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

class GpuAgent;

class Executable;

class ExecutableSymbol;

class ISA;

/// \brief Wraps functionality related to \c hsa_loaded_code_object_t in Luthier
///
/// \note This wrapper is implemented with the following assumptions based on
/// the current state of ROCr:
/// 1. Even though the ROCr HSA vendor loader API
/// (under <hsa/hsa_ven_amd_loader.h>) acknowledge that both file-backed and
/// memory-backed Loaded Code Objects exists, only memory-backed ones are
/// actually implemented. Therefore, querying the storage type and querying the
/// FD of the storage is not included in the API.
/// Luthier assumes all Loaded Code Objects have a memory storage in order to
/// return its associated ELF.
/// In the event that file-backed storage is implemented in the loader, the
/// code needs to be updated.
/// 2. Program Loaded Code Objects has been deprecated and are not used anywhere
/// in the ROCm stack. ROCr does not even allow using Loaded Code Objects with
/// program allocations. Therefore, it is safe to assume all Loaded Code Objects
/// are backed by a \c GpuAgent.
class LoadedCodeObject : public HandleType<hsa_loaded_code_object_t> {
public:
  /// Primary constructor
  /// \param LCO HSA handle of the \c hsa_loaded_code_object_t
  explicit LoadedCodeObject(hsa_loaded_code_object_t LCO);

  /// Queries the \c Executable associated with this \c LoadedCodeObject
  /// \param HsaVenAmdLoaderLoadedCodeObjectGetInfoFn the
  /// \c hsa_ven_amd_loader_loaded_code_object_get_info function
  /// used to complete the operation
  /// \return the \c Executable of this \c LoadedCodeObject on success
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE
  [[nodiscard]] llvm::Expected<Executable>
  getExecutable(const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
                    *HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) const;

  /// Queries the \c GpuAgent associated with this \c LoadedCodeObject
  /// \param HsaVenAmdLoaderLoadedCodeObjectGetInfoFn the
  /// \c hsa_ven_amd_loader_loaded_code_object_get_info function
  /// used to complete the operation
  /// \note As Loaded Code Objects of program allocation are deprecated in ROCr,
  /// it is safe to assume all Loaded Code Objects have agent allocation, and
  /// therefore, are backed by an HSA Agent
  /// \return the \c GpuAgent of this \c LoadedCodeObject on success
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT
  [[nodiscard]] llvm::Expected<GpuAgent>
  getAgent(const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
               *HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) const;

  /// \param HsaVenAmdLoaderLoadedCodeObjectGetInfoFn the
  /// \c hsa_ven_amd_loader_loaded_code_object_get_info function
  /// used to complete the operation
  /// \return the Load Delta of this Loaded Code Object on success
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA
  [[nodiscard]] llvm::Expected<long>
  getLoadDelta(const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
                   *HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) const;

  /// \param HsaVenAmdLoaderLoadedCodeObjectGetInfoFn the
  /// \c hsa_ven_amd_loader_loaded_code_object_get_info function
  /// used to complete the operation
  /// \return \c llvm::ArrayRef to the portion of GPU memory that
  /// this code object has been loaded onto on success
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getLoadedMemory(const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
                      *HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) const;

  /// \param HsaVenAmdLoaderLoadedCodeObjectGetInfoFn the
  /// \c hsa_ven_amd_loader_loaded_code_object_get_info function
  /// used to complete the operation
  /// \return The URI describing the origins of this \c LoadedCodeObject
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI
  [[nodiscard]] llvm::Expected<std::string>
  getUri(const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
             *HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) const;

  /// Queries where the host copy of this <tt>LoadedCodeObject</tt>'s ELF is
  /// stored, and its size from HSA
  /// \param HsaVenAmdLoaderLoadedCodeObjectGetInfoFn the
  /// \c hsa_ven_amd_loader_loaded_code_object_get_info function
  /// used to complete the operation
  /// \return \c llvm::ArrayRef pointing to the beginning and the end of the
  /// storage memory on success
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getStorageMemory(
      const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
          *HsaVenAmdLoaderLoadedCodeObjectGetInfoFn) const;
};

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

namespace llvm {

template <> struct DenseMapInfo<luthier::hsa::LoadedCodeObject> {
  static inline luthier::hsa::LoadedCodeObject getEmptyKey() {
    return luthier::hsa::LoadedCodeObject({DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getEmptyKey()});
  }

  static inline luthier::hsa::LoadedCodeObject getTombstoneKey() {
    return luthier::hsa::LoadedCodeObject({DenseMapInfo<
        decltype(hsa_loaded_code_object_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const luthier::hsa::LoadedCodeObject &LCO) {
    return DenseMapInfo<decltype(hsa_loaded_code_object_t::handle)>::
        getHashValue(LCO.hsaHandle());
  }

  static bool isEqual(const luthier::hsa::LoadedCodeObject &Lhs,
                      const luthier::hsa::LoadedCodeObject &Rhs) {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<luthier::hsa::LoadedCodeObject> {
  size_t operator()(const luthier::hsa::LoadedCodeObject &Obj) const {
    return hash<unsigned long>()(Obj.hsaHandle());
  }
};

template <> struct less<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &Lhs,
                  const luthier::hsa::LoadedCodeObject &Rhs) const {
    return Lhs.hsaHandle() < Rhs.hsaHandle();
  }
};

template <> struct less_equal<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &Lhs,
                  const luthier::hsa::LoadedCodeObject &Rhs) const {
    return Lhs.hsaHandle() <= Rhs.hsaHandle();
  }
};

template <> struct equal_to<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &Lhs,
                  const luthier::hsa::LoadedCodeObject &Rhs) const {
    return Lhs.hsaHandle() == Rhs.hsaHandle();
  }
};

template <> struct not_equal_to<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &Lhs,
                  const luthier::hsa::LoadedCodeObject &Rhs) const {
    return Lhs.hsaHandle() != Rhs.hsaHandle();
  }
};

template <> struct greater<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &Lhs,
                  const luthier::hsa::LoadedCodeObject &Rhs) const {
    return Lhs.hsaHandle() > Rhs.hsaHandle();
  }
};

template <> struct greater_equal<luthier::hsa::LoadedCodeObject> {
  bool operator()(const luthier::hsa::LoadedCodeObject &Lhs,
                  const luthier::hsa::LoadedCodeObject &Rhs) const {
    return Lhs.hsaHandle() >= Rhs.hsaHandle();
  }
};

} // namespace std

#endif