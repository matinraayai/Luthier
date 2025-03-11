//===-- CodeObjectReader.hpp ----------------------------------------------===//
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
/// This file defines the \c hsa::CodeObjectReader interface, representing
/// a wrapper around <tt>hsa_code_object_reader_t</tt>.
/// It is in charge of reading AMDGPU code objects into an \c Executable and
/// creating a <tt>hsa::LoadedCodeObject</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_CODE_OBJECT_READER_HPP
#define LUTHIER_HSA_CODE_OBJECT_READER_HPP
#include "hsa/HandleType.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// \brief Wrapper around the \c hsa_code_object_reader_t handle
class CodeObjectReader : public HandleType<hsa_code_object_reader_t> {
public:
  /// Default constructor
  CodeObjectReader() : HandleType<hsa_code_object_reader_t>({0}) {};

  /// Constructor from a \c hsa_code_object_reader_t handle
  /// \warning This should not be used to create a new
  /// <tt>hsa_code_object_reader_t</tt>, use \c createFromMemory instead.
  /// \param Reader a \c hsa_code_object_reader_t handle, which already has
  /// been created by HSA
  /// \sa createFromMemory
  explicit CodeObjectReader(hsa_code_object_reader_t Reader)
      : HandleType(Reader) {};

  /// Creates a new \c hsa_code_object_reader_t handle
  /// for loading the \p Elf into an \c hsa_executable_t and assigns it to be
  /// managed by this \c CodeObjectReader
  /// \param Elf a code object in memory to be loaded by the
  /// \c hsa_code_object_reader_t
  /// \returns an error if the underlying handle is not zero
  /// \return an \c llvm::Error indicating the success or failure of the
  /// operation
  virtual llvm::Error createFromMemory(llvm::StringRef Elf) = 0;

  /// \see createFromMemory(llvm::StringRef Elf)
  llvm::Error createFromMemory(llvm::ArrayRef<uint8_t> Elf);

  /// Destroys the code object reader handle managed by this \c CodeObjectReader
  /// \return an \c llvm::Error indicating the success of failure of the
  /// operation
  /// \sa hsa_code_object_reader_destroy
  virtual llvm::Error destroy() = 0;

};

} // namespace luthier::hsa

#endif