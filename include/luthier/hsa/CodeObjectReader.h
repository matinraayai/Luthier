//===-- CodeObjectReader.h --------------------------------------*- C++ -*-===//
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
/// \c hsa_code_object_reader_t handle in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_CODE_OBJECT_READER_H
#define LUTHIER_HSA_CODE_OBJECT_READER_H
#include "luthier/hsa/ApiTable.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// Creates a handle to a \c hsa_code_object_reader_t from an \p Elf in memory
/// \param CoreApi the HSA \c ::CoreApiTable container used to dispatch HSA
/// calls
/// \param Elf the code object to be loaded by the code object reader
/// \return Expects the newly created \c hsa_code_object_reader_t on success
/// \sa hsa_code_object_reader_create_from_memory
llvm::Expected<hsa_code_object_reader_t> codeObjectReaderCreateFromMemory(
    const ApiTableContainer<::CoreApiTable> &CoreApi, llvm::StringRef Elf);

/// Creates a handle to a \c hsa_code_object_reader_t from an \p Elf in memory
/// \param CoreApi the HSA \c ::CoreApiTable container used to dispatch HSA
/// calls
/// \param Elf the code object to be loaded
/// \return Expects the newly created \c hsa_code_object_reader_t on success
/// \sa hsa_code_object_reader_create_from_memory
inline llvm::Expected<hsa_code_object_reader_t>
codeObjectReaderCreateFromMemory(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    llvm::ArrayRef<uint8_t> Elf) {
  return codeObjectReaderCreateFromMemory(CoreApi, llvm::toStringRef(Elf));
}

/// Destroys a code object reader instance
/// \param COR the \c hsa_code_object_reader instance being destroyed
/// \param CoreApi the HSA \c ::CoreApiTable container used to dispatch HSA
/// calls
/// \return \c llvm::Error indicating the success or failure of the
/// operation
llvm::Error
codeObjectReaderDestroy(hsa_code_object_reader_t COR,
                        const ApiTableContainer<::CoreApiTable> &CoreApi);

} // namespace luthier::hsa

#endif