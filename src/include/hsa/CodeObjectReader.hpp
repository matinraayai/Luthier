//===-- CodeObjectReader.hpp - HSA Code Object Reader Wrapper -------------===//
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
/// This file defines the \c hsa::CodeObjectReader class, a
/// wrapper around \c hsa_code_object_reader_t in charge of
/// reading AMDGPU code objects into an \c Executable and creating a
/// <tt>LoadedCodeObject</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_CODE_OBJECT_READER_HPP
#define LUTHIER_HSA_CODE_OBJECT_READER_HPP
#include "hsa/HandleType.hpp"
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// \brief Wrapper around the \c hsa_code_object_reader_t handle
class CodeObjectReader : public HandleType<hsa_code_object_reader_t> {
public:
  /// Creates a new \c CodeObjectReader from a \c CodeObject in memory
  /// \param CodeObject the code object to be loaded
  /// \return on success, a \c CodeObjectReader for reading the
  /// \p CodeObjectReader into an <tt>hsa::Executable</tt>; On failure,
  /// an \c llvm::Error
  /// \sa hsa_code_object_reader_create_from_memory
  static llvm::Expected<CodeObjectReader>
  createFromMemory(llvm::StringRef CodeObject);


  /// Creates a new \c CodeObjectReader from a \c CodeObject in memory
  /// \param CodeObject the code object to be loaded
  /// \return on success, a \c CodeObjectReader for reading the
  /// \p CodeObjectReader into an <tt>hsa::Executable</tt>; On failure,
  /// an \c llvm::Error
  /// \sa hsa_code_object_reader_create_from_memory
  static llvm::Expected<CodeObjectReader>
  createFromMemory(llvm::ArrayRef<uint8_t> CodeObject);

  /// Destroys the code object reader handle
  /// \return an \c llvm::Error indicating the success or failure of the
  /// operation
  /// \sa hsa_code_object_reader_destroy
  llvm::Error destroy();

  /// Constructor from a \c hsa_code_object_reader_t handle
  /// \note This should not be used to create a new
  /// <tt>hsa_code_object_reader_t</tt>; For creating new handles
  /// use \c createFromMemory instead.
  /// \param Reader a valid \c hsa_code_object_reader_t handle
  /// \sa createFromMemory
  explicit CodeObjectReader(hsa_code_object_reader_t Reader)
      : HandleType(Reader) {};
};

} // namespace luthier::hsa

#endif