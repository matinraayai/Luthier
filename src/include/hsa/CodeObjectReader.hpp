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
/// This file defines the \c hsa::CodeObjectReader interface, in charge of
/// reading AMDGPU code objects into an \c hsa::Executable and creating a
/// <tt>hsa::LoadedCodeObject</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_CODE_OBJECT_READER_HPP
#define LUTHIER_HSA_CODE_OBJECT_READER_HPP
#include "hsa/HandleType.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// \brief Represents a code object reader in the HSA standard. Reads
/// code objects into an \c hsa::Executable to create an instance
/// of a \c hsa::LoadedCodeObject
class CodeObjectReader {
public:
  /// \return a cloned instance of this \c CodeObjectReader
  [[nodiscard]] virtual std::unique_ptr<CodeObjectReader> clone() const = 0;

  /// \return hash of the \c CodeObjectReader
  [[nodiscard]] virtual size_t hash() const = 0;

  /// Creates a new code object reader for loading the \p Elf into an
  /// \c hsa::Executable and assigns it to be managed by this object
  /// \param Elf a code object in memory to be loaded by the
  /// \c CodeObjectReader
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