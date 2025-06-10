//===-- ObjectUtils.h - Luthier object file utilities  ----------*- C++ -*-===//
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
/// This file defines a set of utilities for the LLVM object file class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_OBJECT_OBJECT_UTILS_H
#define LUTHIER_OBJECT_OBJECT_UTILS_H
#include <llvm/Object/ObjectFile.h>

namespace luthier::object {

/// \param ObjFile a memory region pointed encapsulating an object file
/// \param InitContent indicates the argument of the same named passed to
/// \c llvm::object::ObjectFile when created
/// \return on success, creates and returns a \c llvm::object::ObjectFile
/// representation of the \p ObjFile or an \c llvm::Error on failure
llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>>
createObjectFile(llvm::StringRef ObjFile, bool InitContent = true);

/// \return the target string of the \p ObjFile if successful, an \c llvm::Error
/// on failure
llvm::Expected<std::string>
getObjectFileTarget(const llvm::object::ObjectFile &ObjFile);

/// Returns a mapping between the name of each symbol in the \p ObjFile and its
/// load offset
/// \param ObjFile the object file being inspected
/// \return Expects a \c llvm::StringMap containing the load offset of each
/// symbol inside \p ObjFile
llvm::Expected<llvm::StringMap<uint64_t>>
getSymbolLoadOffsetsMap(const llvm::object::ObjectFile &ObjFile);

} // namespace luthier::object

#endif