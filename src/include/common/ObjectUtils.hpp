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
#include <llvm/Object/ELFObjectFile.h>


#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DIContext.h"
using HandlerFn = std::function<bool(llvm::object::ObjectFile &, llvm::DWARFContext &DICtx,
  const llvm::Twine &, llvm::raw_ostream &OS)>;


namespace luthier {

struct KernelSymbolRef {
  llvm::object::ELFSymbolRef KernelDescriptorSymbol;
  llvm::object::ELFSymbolRef KernelFunctionSymbol;
};

/// As per <a href="https://llvm.org/docs/AMDGPUUsage.html#elf-code-object">
/// AMDGPU backend documentation</a>, AMDGCN object files are 64-bit LSB.
/// Luthier does not support the R600 target, hence it is safe to assume for now
/// all ELF object files encountered by Luthier are of this type.
typedef llvm::object::ELF64LEObjectFile AMDGCNObjectFile;

typedef llvm::object::ELFFile<llvm::object::ELF64LE> AMDGCNELFFile;

/// Parses the ELF file pointed to by \p Elf into a \c AMDGCNObjectFile.
/// \param ELF \c llvm::StringRef encompassing the ELF file in memory
/// \return a \c std::unique_ptr<llvm::object::ELF64LEObjectFile> on successful
/// parsing, an \c llvm::Error on failure
llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
parseAMDGCNObjectFile(llvm::StringRef ELF);

/// Parses the ELF file pointed to by \b Elf into a \b AMDGCNObjectFile.
/// \param ELF \p llvm::ArrayRef<uint8_t> encompassing the ELF file in memory
/// \return a \c std::unique_ptr<llvm::object::ELF64LEObjectFile> on successful
/// parsing, an \c llvm::Error on failure
llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
parseAMDGCNObjectFile(llvm::ArrayRef<uint8_t> ELF);

llvm::Error categorizeSymbols(
    const AMDGCNObjectFile &ObjectFile,
    llvm::SmallVectorImpl<KernelSymbolRef> *KernelSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *DeviceFunctionSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *VariableSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *ExternSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *MiscSymbols);

/// Looks up a symbol by its name in the given \p Elf from its symbol hash
/// table
/// \note Function was adapted from LLVM's OpenMP library
/// \param ObjectFile the ELF object being queried
/// \param SymbolName Name of the symbol being looked up
/// \return an \c llvm::object::ELFSymbolRef if the Symbol was found,
/// an \c std::nullopt if the symbol was not found, and \c llvm::Error if
/// any issue was encountered during the process
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
lookupSymbolByName(const luthier::AMDGCNObjectFile &ObjectFile,
                   llvm::StringRef SymbolName);

llvm::Expected<uint64_t>
getLoadedMemoryOffset(const luthier::AMDGCNELFFile &ELF,
                      const llvm::object::ELFSectionRef &Sec);

llvm::Expected<uint64_t>
getLoadedMemoryOffset(const luthier::AMDGCNELFFile &ELF,
                      const llvm::object::ELFSymbolRef &Sym);

/// Finds the ISA string components of the given \p Obj file
/// \param Obj AMD GCN object file being queried
/// \return an \c std::tuple with the first element being the target triple,
/// the second element being the CPU name, and the last element being the
/// feature string
llvm::Expected<
    std::tuple<llvm::Triple, llvm::StringRef, llvm::SubtargetFeatures>>
getELFObjectFileISA(const luthier::AMDGCNObjectFile &Obj);

/// Parses the note section of \p Obj into a \c hsa::md::Metadata structure
/// for easier access to the document's metadata fields
/// \param Obj the \c luthier::AMDGCNObjectFile to be inspected
/// \return on success, the \c hsa::md::Metadata of the document, or an
/// \c llvm::Error describing the issue encountered during the process
llvm::Expected<std::unique_ptr<hsa::md::Metadata>>
parseNoteMetaData(const luthier::AMDGCNObjectFile &Obj);

/// Dump DICtx



bool handleFile(llvm::StringRef Filename, HandlerFn HandleObj, llvm::raw_ostream &OS);

} // namespace luthier

#endif