//===-- CodeObjectReader.hpp - HSA Code Object Reader Wrapper -------------===//
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
/// This file defines the \c CodeObjectReader class under the \c luthier::hsa
/// namespace, representing a wrapper around the \c hsa_code_object_reader_t
/// in charge of reading AMDGPU code objects into an \c Executable and
/// creating a <tt>LoadedCodeObject</tt>.
//===----------------------------------------------------------------------===//
#ifndef HSA_CODE_OBJECT_READER_HPP
#define HSA_CODE_OBJECT_READER_HPP
#include "hsa/hsa_handle_type.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// \brief Wrapper around the \c hsa_code_object_reader_t handle
/// \note
class CodeObjectReader : public HandleType<hsa_code_object_reader_t> {

public:
  /// Factory function, which creates a handle to a \c CodeObjectReader from
  /// an \p Elf in memory
  /// \param Elf the code object to be loaded
  /// \return on success, a \c CodeObjectReader ready to load the \p Elf into
  /// an <tt>hsa::Executable</tt>, on failure, an \c llvm::HsaError
  /// \sa hsa_code_object_reader_create_from_memory
  static llvm::Expected<CodeObjectReader> createFromMemory(llvm::StringRef Elf);

  /// Factory function, which creates a handle to a \c CodeObjectReader from
  /// an \p Elf in memory
  /// \param Elf the code object to be loaded
  /// \return on success, a \c CodeObjectReader ready to load the \p Elf into
  /// an <tt>hsa::Executable</tt>, on failure, an \c llvm::HsaError
  /// \sa hsa_code_object_reader_create_from_memory
  static llvm::Expected<CodeObjectReader>
  createFromMemory(llvm::ArrayRef<uint8_t> Elf);

  /// Destroys the code object reader instance
  /// \return an \c llvm::HsaError if any issues where encountered, or
  /// an \c llvm::ErrorSuccess if the operation was successful
  /// \sa hsa_code_object_reader_destroy
  llvm::Error destroy();

  /// Constructor from a \c hsa_code_object_reader_t handle
  /// \warning This should not be used to create a new
  /// <tt>hsa_code_object_reader_t</tt>, use \c createFromMemory instead.
  /// \param Reader a \c hsa_code_object_reader_t handle, which already has
  /// been created by HSA
  /// \sa createFromMemory
  explicit CodeObjectReader(hsa_code_object_reader_t Reader)
      : HandleType(Reader){};
};

} // namespace luthier::hsa


#endif