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
/// Defines the \c CodeObjectReader class, a wrapper around the
/// \c hsa_code_object_reader_t handle.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_CODE_OBJECT_READER_HPP
#define LUTHIER_HSA_HSA_CODE_OBJECT_READER_HPP
#include "hsa/HandleType.hpp"
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// \brief Wrapper around the \c hsa_code_object_reader_t handle
class CodeObjectReader : public HandleType<hsa_code_object_reader_t> {

public:
  /// Constructor from a \c hsa_code_object_reader_t handle
  /// \note This should not be used to create a new
  /// <tt>hsa_code_object_reader_t</tt>, use \c createFromMemory instead.
  /// \param Reader a valid \c hsa_code_object_reader_t handle
  /// \sa createFromMemory
  explicit CodeObjectReader(hsa_code_object_reader_t Reader)
      : HandleType(Reader) {};

  /// Factory function, which creates a handle to a \c CodeObjectReader from
  /// an \p Elf in memory
  /// \param HsaCodeObjectReaderCreateFromMemoryFn
  /// \c hsa_code_object_reader_create_from_memory Function used to
  /// perform the operation
  /// \param Elf the code object to be loaded
  /// \return Expects the newly created \c CodeObjectReader on success
  /// \sa hsa_code_object_reader_create_from_memory
  static llvm::Expected<CodeObjectReader>
  createFromMemory(const decltype(hsa_code_object_reader_create_from_memory)
                       *HsaCodeObjectReaderCreateFromMemoryFn,
                   llvm::StringRef Elf);

  /// Factory function, which creates a handle to a \c CodeObjectReader from
  /// an \p Elf in memory
  /// \param HsaCodeObjectReaderCreateFromMemoryFn
  /// \c hsa_code_object_reader_create_from_memory Function used to
  /// perform the operation
  /// \param Elf the code object to be loaded
  /// \return Expects the newly created \c CodeObjectReader on success
  /// \sa hsa_code_object_reader_create_from_memory
  static llvm::Expected<CodeObjectReader>
  createFromMemory(const decltype(hsa_code_object_reader_create_from_memory)
                       *HsaCodeObjectReaderCreateFromMemoryFn,
                   llvm::ArrayRef<uint8_t> Elf);

  /// Destroys the code object reader instance
  /// \param HsaCodeObjectReaderDestroyFn \c hsa_code_object_reader_destroy
  /// function used to perform the operation
  /// \return \c llvm::Error indicating the success or failure of the
  /// operation
  llvm::Error destroy(const decltype(hsa_code_object_reader_destroy)
                              *HsaCodeObjectReaderDestroyFn);
};

} // namespace luthier::hsa

#endif