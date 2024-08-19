//===-- LoadedCodeObject.hpp - HSA Loaded Code Object Wrapper -------------===//
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
/// This file defines the \c LoadedCodeObject class under the \c luthier::hsa
/// namespace, representing a wrapper around the \c hsa_loaded_code_object_t HSA
/// type.
//===----------------------------------------------------------------------===//
#ifndef HSA_LOADED_CODE_OBJECT_HPP
#define HSA_LOADED_CODE_OBJECT_HPP
#include "HandleType.hpp"
#include "common/ObjectUtils.hpp"
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Error.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>

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
/// FD of the storage is not included in the API. L
/// Luthier assumes all Loaded Code Objects have a memory storage in order to
/// return its associated ELF.
/// In the event that file-backed storage is implemented in the loader, the
/// code needs to be updated.
/// 2. Program Loaded Code Objects has been deprecated and are not used anywhere
/// in the ROCm stack. ROCr does not even allow using Loaded Code Objects with
/// program allocations. Therefore, it is safe to assume all Loaded Code Objects
/// are backed by a \c GpuAgent.
/// 3. Internally, the vendor loader API keeps track of loaded segments, and
/// allows for querying these segments via the
/// \c hsa_ven_amd_loader_query_segment_descriptors function. As of right now,
/// Luthier does not use this information to locate the load address of the
/// symbols, and instead relies on the \c luthier::getSymbolLMA and
/// \c luthier::getSectionLMA functions to calculate the load address.
///
/// \note This wrapper relies on cached functionality as described by the
/// \c hsa::ExecutableBackedCachable interface and backed by the \c
/// hsa::Platform Singleton.
class LoadedCodeObject : public HandleType<hsa_loaded_code_object_t> {
public:
  /// Primary constructor
  /// \param LCO HSA handle of the \c hsa_loaded_code_object_t
  explicit LoadedCodeObject(hsa_loaded_code_object_t LCO);

  /// Queries the \c Executable associated with this \c LoadedCodeObject
  /// \return the \c Executable of this \c LoadedCodeObject, or an
  /// \c luthier::HsaError reporting any HSA errors occurred
  /// \note Performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE
  [[nodiscard]] llvm::Expected<Executable> getExecutable() const;

  /// Queries the \c GpuAgent associated with this \c LoadedCodeObject
  /// \note As Loaded Code Objects of program allocation are deprecated in ROCr,
  /// it is safe to assume all Loaded Code Objects have agent allocation, and
  /// therefore, are backed by an HSA Agent
  /// \return the \c GpuAgent of this \c LoadedCodeObject, or a
  /// \c Luthier::HsaError reporting any HSA errors occurred during this
  /// operation
  /// \note Performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT
  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  /// Returns a reference to the \c luthier::AMDGCNObjectFile of
  /// the ELF associated with this \c LoadedCodeObject
  /// The ELF is obtained by parsing the Loaded Code Object's Storage memory.
  /// \note This operation relies on cached information
  /// \return the \c luthier::AMDGCNObjectFile of this \c LoadedCodeObject, or
  /// an \c llvm::Error if this \c LoadedCodeObject has not been cached
  /// \sa
  /// HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE
  /// \sa
  /// HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE
  [[nodiscard]] llvm::Expected<luthier::AMDGCNObjectFile &>
  getStorageELF() const;

  /// \return the Load Delta of this Loaded Code Object, or a
  /// \c luthier::HsaError
  /// \note performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA
  [[nodiscard]] llvm::Expected<long> getLoadDelta() const;

  /// \return an \c llvm::ArrayRef to the portion of GPU memory that
  /// this code object has been loaded onto, or a \c luthier::HsaError
  /// \note performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getLoadedMemory() const;

  /// \return The URI describing the origins of this \c LoadedCodeObject
  /// \note performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI
  [[nodiscard]] llvm::Expected<std::string> getUri() const;

  /// \return the \c hsa::ISA of this loaded code object, or an \c llvm::Error
  /// indicating any issues encountered during the process
  /// \note this operation relies on cached information
  [[nodiscard]] llvm::Expected<ISA> getISA() const;

  /// \return the parsed Metadata document associated with this loaded code
  /// object, or an \c llvm::Error indicating any issues encountered during the
  /// process
  /// \note this operation relies on cached information
  [[nodiscard]] llvm::Expected<const hsa::md::Metadata &> getMetadata() const;

  /// Appends all the <tt>hsa::LoadedCodeObjectKernel</tt>s that belong to
  /// this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] llvm::Error getKernelSymbols(
      llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const;

  /// Appends all the <tt>hsa::LoadedCodeObjectVariable</tt>s that belong to
  /// this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] llvm::Error getVariableSymbols(
      llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const;

  /// Appends all the <tt>hsa::LoadedCodeObjectDeviceFunction</tt>s that belong
  /// to this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] llvm::Error getDeviceFunctionSymbols(
      llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const;

  /// Appends all the <tt>hsa::LoadedCodeObjectExternSymbol</tt>s that belong
  /// to this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] llvm::Error getExternalSymbols(
      llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const;

  /// Appends all <tt>LoadedCodeObjectSymbol</tt> of this loaded code object
  /// to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] llvm::Error getLoadedCodeObjectSymbols(
      llvm::SmallVectorImpl<const LoadedCodeObjectSymbol *> &Out) const;

  /// Looks up the associated \c LoadedCodeObjectSymbol with the given \p Name
  /// in this loaded code object and returns it if found
  /// \param Name name of the symbol
  /// \return on success, the \c LoadedCodeObjectSymbol associated with the
  /// \p Name if found, \c std::nullopt otherwise; On failure, an \c llvm::Error
  /// describing the issue encountered during the process
  [[nodiscard]] llvm::Expected<const LoadedCodeObjectSymbol *>
  getLoadedCodeObjectSymbolByName(llvm::StringRef Name) const;

  /// Queries where the host copy of this <tt>LoadedCodeObject</tt>'s ELF is
  /// stored, and its size from HSA
  /// \return An \b llvm::ArrayRef pointing to the beginning and the end of the
  /// storage memory on success, or an \c luthier::HsaError reporting any issues
  /// encountered during this operation
  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getStorageMemory() const;

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