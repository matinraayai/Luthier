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
/// processing object files using LLVM object and DWARF utilities.
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

/// Parses the ELF file pointed to by \p ObjectFile into a
/// \c llvm::object::ELFObjectFileBase
/// \param ELF \c llvm::StringRef encompassing the ELF file in memory
/// \return a \c std::unique_ptr<llvm::object::ELFObjectFileBase> on successful
/// parsing, an \c llvm::Error on failure
llvm::Expected<std::unique_ptr<llvm::object::ELFObjectFileBase>>
parseELFObjectFile(llvm::StringRef ObjectFile);

/// Parses the ELF file pointed to by \p ObjectFile into a
/// \c llvm::object::ELFObjectFileBase
/// \param ELF \c llvm::ArrayRef encompassing the ELF file in memory
/// \return a \c std::unique_ptr<llvm::object::ELFObjectFileBase> on successful
/// parsing, an \c llvm::Error on failure
llvm::Expected<std::unique_ptr<llvm::object::ELFObjectFileBase>>
parseELFObjectFile(llvm::ArrayRef<uint8_t> ObjectFile);


//===----------------------------------------------------------------------===//
// ELF ISA query method
//===----------------------------------------------------------------------===//

/// Finds the ISA string components of the \p Obj file
/// \return an \c std::tuple with the first element being the target triple,
/// the second element being the CPU name, and the last element being the
/// sub-target features
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

/// \return the kernel function symbol (i.e. the machine code
/// contents of the kernel) of the \p KernelDescriptorSymbol or an
/// \c llvm::Error if \p KernelDescriptorSymbol is not a kernel descriptor, or
/// if any other issues were encountered during the process
llvm::Expected<llvm::object::ELFSymbolRef>
getKernelFunctionForAMDGPUKernelDescriptor(
    const llvm::object::ELFSymbolRef &KernelDescriptorSymbol);

/// \return \c true if \p Symbol is a amdgpu kernel function (i.e. contains the
/// machine code contents of the kernel), \c false if the symbol is not an
/// amdgpu kernel function, or an \c llvm::Error if any issues were encountered
/// during the process
llvm::Expected<bool>
isAMDGPUKernelFunction(const llvm::object::ELFSymbolRef &Symbol);

/// \return the kernel descriptor symbol associated with the kernel function
/// \p Symbol or an \c llvm::Error if \p KernelFunctionSymbol is not a kernel
/// function or if an issue was encountered
llvm::Expected<llvm::object::ELFSymbolRef>
getKernelDescriptorForAMDGPUKernelFunction(
    const llvm::object::ELFSymbolRef &KernelFunctionSymbol);

/// \return \c true if \p Symbol is a amdgpu device function, \c false if not,
/// \c llvm::Error if any issues were encountered during the process
llvm::Expected<bool>
isAMDGPUDeviceFunction(const llvm::object::ELFSymbolRef &Symbol);

/// \return \c true if \p Symbol is a variable symbol, \c false if not, or
/// an \c llvm::Error if an issue was encountered
inline llvm::Expected<bool>
isVariable(const llvm::object::ELFSymbolRef &Symbol) {
  llvm::Expected<bool> IsKD = isAMDGPUKernelDescriptor(Symbol);
  LUTHIER_RETURN_ON_ERROR(IsKD.takeError());
  return Symbol.getELFType() == llvm::ELF::STT_OBJECT && !*IsKD;
}

/// \return \c true if \p Symbol is an external symbol, \c false otherwise
inline bool isExtern(const llvm::object::ELFSymbolRef &Symbol) {
  return Symbol.getELFType() == llvm::ELF::STT_NOTYPE &&
         Symbol.getBinding() == llvm::ELF::STB_GLOBAL;
}

/// Iterates over the symbols of \p AMDGPUObjectFile and categorizes them into
/// different AMDGPU symbol types
/// \param AMDGPUObjectFile the \c AMDGCNObjectFile being inspected
/// \param [out] KernelDescriptorSymbols if not \c nullptr will return the
/// kernel descriptor symbols inside the \p ObjectFile
/// \param [out] KernelFunctionSymbols if not \c nullptr will return the kernel
/// function symbols inside the \p ObjectFile
/// \param [out] DeviceFunctionSymbols if not \c nullptr will return the
/// device function symbols inside the \p ObjectFile
/// \param [out] VariableSymbols if not \c nullptr will return the variable
/// symbols defined inside the symbol
/// \param [out] ExternSymbols if not \c nullptr will return external variable
/// symbols not defined inside \p ObjectFile
/// \param [out] MiscSymbols if not \c nullptr will return any symbol that is
/// not a kernel descriptor, kernel function, device function, variable, or
/// external symbol
/// \return an \c llvm::Error describing the success of the operation or any
/// issue encountered during the process
llvm::Error categorizeAMDGPUSymbols(
    const llvm::object::ELFObjectFileBase &AMDGPUObjectFile,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *KernelDescriptorSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *KernelFunctionSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *DeviceFunctionSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *VariableSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *ExternSymbols,
    llvm::SmallVectorImpl<llvm::object::ELFSymbolRef> *MiscSymbols);

/// Looks up a symbol by its name in the given \p ObjectFile from its symbol
/// hash table
/// \note Function was adopted from LLVM's OpenMP library
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

/// Returns the <tt>Section</tt>'s loaded memory offset from its object file's
/// loaded base
/// \note Function was adapted from LLVM's object dump utility
/// \param Section the section being queried
/// \return on success, the loaded offset of the \c Sec with respect to the
/// ELF's load base; an \c llvm::Error on failure
llvm::Expected<uint64_t>
getLoadedMemoryOffset(const llvm::object::ELFSectionRef &Section);

/// Returns the <tt>Symbol</tt>'s loaded memory offset from its object file's
/// loaded base
/// \note Function was adapted from LLVM's object dump utility
/// \param Symbol the symbol being queried
/// \return on success, the loaded offset of the \c Sym with respect to the
/// ELF's load base; an \c llvm::Error on failure
llvm::Expected<uint64_t>
getLoadedMemoryOffset(const llvm::object::ELFSymbolRef &Symbol);


/// \return an \c llvm::ArrayRef<uint8_t> encapsulating the <tt>Symbol</tt>'s
/// contents inside its parent section, or an \c llvm::Error otherwise
llvm::Expected<llvm::ArrayRef<uint8_t>>
getELFSymbolRefContents(const llvm::object::ELFSymbolRef &Symbol);

} // namespace luthier

#endif