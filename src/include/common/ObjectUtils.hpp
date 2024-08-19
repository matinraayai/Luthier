//===-- ObjectUtils.hpp - Luthier's Object File Utility  ------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
#include <hip/hip_runtime.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/BinaryFormat/MsgPackDocument.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Object/ELFTypes.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/AMDGPUAddrSpace.h>

#include <common/Error.hpp>
#include <luthier/hsa/Metadata.h>
#include <luthier/types.h>
#include <map>
#include <optional>
#include <utility>

namespace luthier {

/// As per <a href="https://llvm.org/docs/AMDGPUUsage.html#elf-code-object">
/// AMDGPU backend documentation</a>, AMDGCN object files are 64-bit LSB.
/// Luthier does not support the R600 target, hence it is safe to assume for now
/// all ELF object files encountered by Luthier are of this type.
typedef llvm::object::ELF64LEObjectFile AMDGCNObjectFile;

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
getAMDGCNObjectFile(llvm::ArrayRef<uint8_t> ELF);

/// Looks up a symbol by its name in the given \p Elf from its symbol hash
/// table
/// \param ELF the ELF object being queried
/// \param SymbolName Name of the symbol being looked up
/// \return an \c llvm::object::ELFSymbolRef if the Symbol was found,
/// an \c std::nullopt if the symbol was not found, and \c llvm::Error if
/// any issue was encountered during the process
llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
lookupSymbolByName(const luthier::AMDGCNObjectFile &ELF,
                   llvm::StringRef SymbolName);

/// Returns the <tt>Sec</tt>'s loaded memory offset from the <tt>ELF</tt>'s
/// loaded base
/// \tparam ELFT type of ELF used
/// \param ELF the Object file being queried
/// \param Sec the ELF's section
/// \return on success, the loaded offset of the section with respect to the
/// ELF's load base; an \c llvm::Error on failure
template <class ELFT>
llvm::Expected<uint64_t> getSectionLMA(const llvm::object::ELFFile<ELFT> &ELF,
                                       const llvm::object::ELFSectionRef &Sec) {
  auto PhdrRange = ELF.program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRange.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRange)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(
            Phdr, *llvm::cast<const llvm::object::ELFObjectFile<ELFT>>(
                       Sec.getObject())
                       ->getSection(Sec.getRawDataRefImpl()))))
      return Sec.getAddress() - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return Sec.getAddress();
}

template <class ELFT>
llvm::Expected<uint64_t> getSymbolLMA(const llvm::object::ELFFile<ELFT> &Obj,
                                      const llvm::object::ELFSymbolRef &Sym) {
  auto PhdrRange = Obj.program_headers();
  LUTHIER_RETURN_ON_ERROR(PhdrRange.takeError());

  auto SymbolSection = Sym.getSection();
  LUTHIER_RETURN_ON_ERROR(SymbolSection.takeError());

  auto SymbolAddress = Sym.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolAddress.takeError());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFT::Phdr &Phdr : *PhdrRange)
    if ((Phdr.p_type == llvm::ELF::PT_LOAD) &&
        (llvm::object::isSectionInSegment<ELFT>(
            Phdr, *llvm::cast<const llvm::object::ELFObjectFile<ELFT>>(
                       Sym.getObject())
                       ->getSection(SymbolSection.get()->getRawDataRefImpl()))))
      return *SymbolAddress - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return *SymbolAddress;
}

template <typename ELFT>
llvm::Expected<
    std::tuple<llvm::Triple, llvm::StringRef, llvm::SubtargetFeatures>>
getELFObjectFileISA(const llvm::object::ELFObjectFile<ELFT> &Obj) {
  llvm::Triple TT = Obj.makeTriple();
  std::optional<llvm::StringRef> CPU = Obj.tryGetCPUName();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CPU.has_value()));
  llvm::SubtargetFeatures Features;
  LUTHIER_RETURN_ON_ERROR(Obj.getFeatures().moveInto(Features));
  return std::make_tuple(TT, *CPU, Features);
}

llvm::Expected<hsa::md::Metadata>
parseMetaDoc(llvm::msgpack::Document &KernelMetaNode);

template <class ELFT>
static bool
processNote(const typename ELFT::Note &Note, const std::string &NoteDescString,
            llvm::msgpack::Document &Doc, llvm::msgpack::DocNode &Root) {

  if (Note.getName() == "AMD" &&
      Note.getType() == llvm::ELF::NT_AMD_HSA_METADATA) {
    if (!Root.isEmpty()) {
      return false;
    }
    if (!Doc.fromYAML(NoteDescString)) {
      return false;
    }
    return true;
  }
  if (((Note.getName() == "AMD" || Note.getName() == "AMDGPU") &&
       Note.getType() == llvm::ELF::NT_AMD_PAL_METADATA) ||
      (Note.getName() == "AMDGPU" &&
       Note.getType() == llvm::ELF::NT_AMDGPU_METADATA)) {
    if (!Doc.readFromBlob(NoteDescString, false)) {
      return false;
    }
    return true;
  }
  return false;
}

template <typename ELFT>
llvm::Expected<hsa::md::Metadata>
parseNoteMetaData(const llvm::object::ELFObjectFile<ELFT> &Obj) {
  bool Found = false;
  llvm::msgpack::Document Doc;
  auto &Root = Doc.getRoot();
  const llvm::object::ELFFile<ELFT> &ELFFile = Obj.getELFFile();
  auto ProgramHeaders = ELFFile.program_headers();
  std::string DescString;
  LUTHIER_RETURN_ON_ERROR(ProgramHeaders.takeError());
  for (const auto &Phdr : *ProgramHeaders) {
    if (Phdr.p_type == llvm::ELF::PT_NOTE) {
      llvm::Error Err = llvm::Error::success();
      for (const auto &Note : ELFFile.notes(Phdr, Err)) {
        DescString = Note.getDescAsStringRef(4);
        if (processNote<ELFT>(Note, DescString, Doc, Root)) {
          Found = true;
        }
      }
      LUTHIER_RETURN_ON_ERROR(Err);
    }
  }
  if (Found) {
    return parseMetaDoc(Doc);
  }

  auto Sections = ELFFile.sections();
  LUTHIER_RETURN_ON_ERROR(Sections.takeError());

  for (const auto &Shdr : *Sections) {
    if (Shdr.sh_type != llvm::ELF::SHT_NOTE) {
      continue;
    }
    llvm::Error Err = llvm::Error::success();
    for (const auto &Note : ELFFile.notes(Shdr, Err)) {
      DescString = Note.getDescAsStringRef(4);
      if (processNote<ELFT>(Note, DescString, Doc, Root)) {
        Found = true;
      }
    }
    LUTHIER_RETURN_ON_ERROR(Err);
  }

  if (Found)
    return parseMetaDoc(Doc);
  else
    return LUTHIER_ASSERTION(Found);
}

} // namespace luthier

#endif