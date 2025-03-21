//===-- ObjectUtils.hpp - Luthier's Object File Utility  ------------------===//
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
/// This file contains all operations related to dealing with parsing and
/// processing AMDGPU code objects using LLVM object file and DWARF utilities.\n
/// Luthier uses LLVM's object library (under llvm/Object folder in LLVM)
/// to parse and inspect AMDGPU code objects, and if present, uses LLVM's
/// DebugInfo library (under llvm/DebugInfo) to parse and process DWARF
/// information from them.\n
/// <tt>ObjectUtils.hpp</tt> is meant to \b only include functionality that:\n
/// - concerns ELF object files and ELF file section parsing and processing,
/// including DWARF debug information.\n
/// - is specific to AMDGPU GCN code objects. Some examples include parsing
/// an AMDGCN object file and converting it to
/// \c llvm::object::ELF64LEObjectFile, or parsing the note section of an
/// AMDGPU code object into a \c luthier::hsa::md::Metadata.\n
/// - does not readily exist in LLVM's object library, and/or is implemented
/// in other LLVM-based tools or project. Some examples include retrieving
/// symbols by name, or getting the loaded address of a symbol\n
/// Although not strictly restricted for this specific purpose,
/// <tt>ObjectUtils.hpp</tt> is only used to supplement ROCr functionality,
/// by parsing the Storage memory ELF of an
/// <tt>luthier::hsa::LoadedCodeObject</tt>, which is exposed in hsa wrapper
/// primitives in the \c luthier::hsa namespace.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_OBJECT_UTILS_HPP
#define LUTHIER_COMMON_OBJECT_UTILS_HPP
#include "luthier/hsa/Metadata.h"
#include <luthier/common/AMDGCNObjectFile.h>

namespace luthier {

/// Parses the note section of \p Obj into a \c hsa::md::Metadata structure
/// for easier access to the document's metadata fields
/// \param Obj the \c luthier::AMDGCNObjectFile to be inspected
/// \return on success, the \c hsa::md::Metadata of the document, or an
/// \c llvm::Error describing the issue encountered during the process
llvm::Expected<std::unique_ptr<hsa::md::Metadata>>
parseNoteMetaData(const luthier::AMDGCNObjectFile &Obj);

} // namespace luthier

#endif