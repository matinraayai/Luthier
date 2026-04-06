//===-- Linker.h - LLD Linker Wrapper ---------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file Linker.h
/// Defines wrappers around LLD linking functionality used by Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LINKER_LINKER_H
#define LUTHIER_LINKER_LINKER_H
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>

namespace luthier::linker {

/// Links the relocatable object file \p Code to an executable shared object
/// using LLD, which can then be loaded into the HSA runtime.
/// \param [in] Code the relocatable ELF file bytes
/// \param [out] Out the linked executable bytes
/// \return an \c llvm::Error indicating the success or failure of the operation
llvm::Error linkRelocatableToExecutable(llvm::ArrayRef<char> Code,
                                        llvm::SmallVectorImpl<char> &Out);

} // namespace luthier::linker

#endif
