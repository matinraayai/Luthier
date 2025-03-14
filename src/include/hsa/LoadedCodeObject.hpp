//===-- LoadedCodeObject.hpp ----------------------------------------------===//
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
/// This file defines the \c hsa::LoadedCodeObject interface, representing
/// a loaded code object inside the HSA standard.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_LOADED_CODE_OBJECT_HPP
#define LUTHIER_HSA_LOADED_CODE_OBJECT_HPP
#include <llvm/Support/Error.h>
#include <llvm/Support/ExtensibleRTTI.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>

namespace luthier::hsa {

class GpuAgent;

class Executable;

class ExecutableSymbol;

class ISA;

/// \brief Wraps functionality related to a loaded code object in HSA
class LoadedCodeObject
    : public llvm::RTTIExtends<LoadedCodeObject, llvm::RTTIRoot> {
public:
  static char ID;
  /// Queries the \c Executable associated with this \c LoadedCodeObject
  /// \return the \c Executable of this \c LoadedCodeObject, or an
  /// \c llvm::Error indicating any issues occurred in the process
  /// \note Performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<Executable>>
  getExecutable() const = 0;

  /// Queries the \c GpuAgent associated with this \c LoadedCodeObject
  /// \note As Loaded Code Objects of program allocation are deprecated in ROCr,
  /// it is safe to assume all Loaded Code Objects have agent allocation, and
  /// therefore, are backed by an HSA Agent
  /// \return the \c GpuAgent of this \c LoadedCodeObject, or a
  /// \c Luthier::HsaError reporting any HSA errors occurred during this
  /// operation
  /// \note Performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<GpuAgent>>
  getAgent() const = 0;

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
  [[nodiscard]] virtual llvm::Expected<llvm::object::ELF64LEObjectFile &>
  getStorageELF() const = 0;

  /// \return the Load Delta of this Loaded Code Object, or a
  /// \c luthier::HsaError
  /// \note performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA
  [[nodiscard]] virtual llvm::Expected<long> getLoadDelta() const = 0;

  /// \return an \c llvm::ArrayRef to the portion of GPU memory that
  /// this code object has been loaded onto, or a \c luthier::HsaError
  /// \note performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE
  [[nodiscard]] virtual llvm::Expected<llvm::ArrayRef<uint8_t>>
  getLoadedMemory() const = 0;

  /// \return The URI describing the origins of this \c LoadedCodeObject
  /// \note performs an HSA call to complete this operation
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH
  /// \sa HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI
  [[nodiscard]] virtual llvm::Expected<std::string> getUri() const = 0;

  /// \return the \c hsa::ISA of this loaded code object, or an \c llvm::Error
  /// indicating any issues encountered during the process
  /// \note this operation relies on cached information
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<ISA>> getISA() const = 0;

  /// \return the parsed Metadata document associated with this loaded code
  /// object, or an \c llvm::Error indicating any issues encountered during the
  /// process
  /// \note this operation relies on cached information
  [[nodiscard]] virtual llvm::Expected<const hsa::md::Metadata &>
  getMetadata() const = 0;

  /// Appends all the <tt>hsa::LoadedCodeObjectKernel</tt>s that belong to
  /// this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] virtual llvm::Error getKernelSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const = 0;

  /// Appends all the <tt>hsa::LoadedCodeObjectVariable</tt>s that belong to
  /// this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] virtual llvm::Error getVariableSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const = 0;

  /// Appends all the <tt>hsa::LoadedCodeObjectDeviceFunction</tt>s that belong
  /// to this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] virtual llvm::Error getDeviceFunctionSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const = 0;

  /// Appends all the <tt>hsa::LoadedCodeObjectExternSymbol</tt>s that belong
  /// to this loaded code object to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] virtual llvm::Error getExternalSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
      const = 0;

  /// Appends all <tt>LoadedCodeObjectSymbol</tt> of this loaded code object
  /// to the \p Out vector
  /// \param [out] Out an \c llvm::SmallVectorImpl where the symbols
  /// will be appended to
  /// \note this operation relies on cached information
  /// \return \c llvm::Error describing whether the operation has succeeded or
  /// not
  [[nodiscard]] virtual llvm::Error getLoadedCodeObjectSymbols(
      llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObjectSymbol>> &Out)
      const = 0;

  /// Looks up the associated \c LoadedCodeObjectSymbol with the given \p Name
  /// in this loaded code object and returns it if found
  /// \param Name name of the symbol
  /// \return on success, the \c LoadedCodeObjectSymbol associated with the
  /// \p Name if found, \c std::nullopt otherwise; On failure, an \c llvm::Error
  /// describing the issue encountered during the process
  [[nodiscard]] virtual llvm::Expected<std::unique_ptr<LoadedCodeObjectSymbol>>
  getLoadedCodeObjectSymbolByName(llvm::StringRef Name) const = 0;

  /// Queries where the host copy of this <tt>LoadedCodeObject</tt>'s ELF is
  /// stored, and its size from HSA
  /// \return An \b llvm::ArrayRef pointing to the beginning and the end of the
  /// storage memory on success, or an \c luthier::HsaError reporting any issues
  /// encountered during this operation
  [[nodiscard]] virtual llvm::Expected<llvm::ArrayRef<uint8_t>>
  getStorageMemory() const = 0;
};

} // namespace luthier::hsa

#endif