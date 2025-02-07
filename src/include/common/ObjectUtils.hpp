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
/// This file defines operations related to dealing with parsing and
/// processing object files using LLVM object file and DWARF utilities.\n
/// Luthier uses LLVM's object library (under llvm/Object folder in LLVM)
/// to parse and inspect AMDGPU code objects, and if present, uses LLVM's
/// DebugInfo library (under llvm/DebugInfo) to parse and process DWARF
/// information from them.\n
/// <tt>ObjectUtils.hpp</tt> is meant to \b only include functionality that:\n
/// - concerns ELF object files and ELF file section parsing and processing,
/// including DWARF debug information.\n
/// - does not readily exist in LLVM's object library, and/or is implemented
/// in other LLVM-based tools or project. Some examples include retrieving
/// symbols by name, or getting the loaded address of a symbol.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_OBJECT_UTILS_HPP
#define LUTHIER_COMMON_OBJECT_UTILS_HPP
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/Metadata.h"
#include <llvm/Object/ELFObjectFile.h>

namespace luthier {

//===----------------------------------------------------------------------===//
// Object file parsing methods
//===----------------------------------------------------------------------===//

/// Parses the ELF file pointed to by \p Elf into a \c llvm::object::ObjectFile
/// \param ELF \c llvm::StringRef encompassing the ELF file in memory
/// \return a \c std::unique_ptr<llvm::object::ObjectFile> on successful
/// parsing, an \c llvm::Error on failure
llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>>
parseObjectFile(llvm::StringRef ObjectFile);

/// Parses the ELF file pointed to by \p Elf into a \c llvm::object::ObjectFile
/// \param ELF \c llvm::ArrayRef encompassing the ELF file in memory
/// \return a \c std::unique_ptr<llvm::object::ObjectFile> on successful
/// parsing, an \c llvm::Error on failure
llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>>
parseObjectFile(llvm::ArrayRef<uint8_t> ObjectFile);

template <typename ELT>
llvm::Expected<std::unique_ptr<llvm::object::ELFObjectFile<ELT>>>
parseObjectFile(llvm::StringRef Elf) {
  auto ObjectFileOrErr = parseObjectFile(Elf);
  LUTHIER_RETURN_ON_ERROR(ObjectFileOrErr.takeError());
  return unique_dyn_cast<llvm::object::ELFObjectFile<ELT>>(
      std::move(*ObjectFileOrErr));
}

//===----------------------------------------------------------------------===//
// ELF machine query methods
//===----------------------------------------------------------------------===//

/// \return \c true if the object file is intended for R600 targets, \c false
/// otherwise
bool isR600(const llvm::object::ELFObjectFileBase &ObjectFile);

/// \return \c true if the object file is intended for AMDGCN targets, \c false
/// otherwise
bool isAMDGCN(const llvm::object::ELFObjectFileBase &ObjectFile);

//===----------------------------------------------------------------------===//
// ELF ISA query method
//===----------------------------------------------------------------------===//

/// Finds the ISA string components of the given \p Obj file
/// \param Obj AMD GCN object file being queried
/// \return an \c std::tuple with the first element being the target triple,
/// the second element being the CPU name, and the last element being the
/// feature string
llvm::Expected<
    std::tuple<llvm::Triple, llvm::StringRef, llvm::SubtargetFeatures>>
getELFObjectFileISA(const llvm::object::ELFObjectFileBase &Obj);

//===----------------------------------------------------------------------===//
// ELF symbol query methods
//===----------------------------------------------------------------------===//

/// \return \c true if \p Symbol is a amdgpu kernel descriptor, \c false if not,
/// \c llvm::Error if any issues were encountered during the process
llvm::Expected<bool>
isAMDGPUKernelDescriptor(const llvm::object::ELFSymbolRef &Symbol);

/// \return \c true if \p Symbol is a amdgpu kernel function, \c false if not,
/// \c llvm::Error if any issues were encountered during the process
llvm::Expected<bool>
isAMDGPUKernelFunction(const llvm::object::ELFSymbolRef &Symbol);

/// \return \c true if \p Symbol is a amdgpu device function, \c false if not,
/// \c llvm::Error if any issues were encountered during the process
llvm::Expected<bool>
isAMDGPUDeviceFunction(const llvm::object::ELFSymbolRef &Symbol);

inline bool isVariable(const llvm::object::ELFSymbolRef &Symbol) {
  return Symbol.getELFType() == llvm::ELF::STT_OBJECT;
}

inline bool isExtern(const llvm::object::ELFSymbolRef &Symbol) {
  return Symbol.getELFType() == llvm::ELF::STT_NOTYPE &&
         Symbol.getBinding() == llvm::ELF::STB_GLOBAL;
}

///// Iterates over the symbols of \p ObjectFile and categorizes them into
///// different AMDGPU symbol types
///// \param ObjectFile the \c AMDGCNObjectFile being inspected
///// \param [out] KernelSymbols if not \c nullptr will return the kernel
/// symbols
///// inside the \p ObjectFile
///// \param [out] DeviceFunctionSymbols if not \c nullptr will return the
/// device
///// function symbols inside the \p ObjectFile
///// \param [out] VariableSymbols if not \c nullptr will
///// \param [out] ExternSymbols
///// \param [out] MiscSymbols
///// \return an \c llvm::Error describing the success of the operation or any
///// issue encountered during the process
// llvm::Error categorizeSymbols(
//     const AMDGCNObjectFile &ObjectFile,
//     llvm::SmallVectorImpl<KernelSymbolRef> *KernelSymbols,
//     llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *DeviceFunctionSymbols,
//     llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *VariableSymbols,
//     llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *ExternSymbols,
//     llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *MiscSymbols);

/// Looks up a symbol by its name in the given \p Elf from its symbol hash
/// table
/// \note Function was adapted from LLVM's OpenMP library
/// \param ObjectFile the ELF object being queried
/// \param SymbolName Name of the symbol being looked up
/// \return an \c llvm::object::ELFSymbolRef if the Symbol was found,
/// an \c std::nullopt if the symbol was not found, and \c llvm::Error if
/// any issue was encountered during the process
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
lookupSymbolByName(const llvm::object::ELFObjectFileBase &ObjectFile,
                   llvm::StringRef SymbolName);

//===----------------------------------------------------------------------===//
// ELF loading query functions
//===----------------------------------------------------------------------===//

/// Returns the <tt>Sym</tt>'s loaded memory offset from its object file's
/// loaded base
/// \note Function was adapted from LLVM's object dump utility
/// \param Sec the section being queried
/// \return on success, the loaded offset of the \c Sec with respect to the
/// ELF's load base; an \c llvm::Error on failure
llvm::Expected<uint64_t>
getLoadedMemoryOffset(const llvm::object::ELFSectionRef &Sec);

/// Returns the <tt>Sym</tt>'s loaded memory offset from its object file's
/// loaded base
/// \note Function was adapted from LLVM's object dump utility
/// \param Sym the symbol being queried
/// \return on success, the loaded offset of the \c Sym with respect to the
/// ELF's load base; an \c llvm::Error on failure
llvm::Expected<uint64_t>
getLoadedMemoryOffset(const llvm::object::ELFSymbolRef &Sym);

} // namespace luthier

#endif