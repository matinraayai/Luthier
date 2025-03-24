//===-- Disassembler.cpp - Luthier's Disassembler Singleton  --------------===//
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
/// This file implements Luthier's Disassembler singleton.
//===----------------------------------------------------------------------===//
#include "tooling_common/Disassembler.hpp"
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCDisassembler/MCRelocationInfo.h>
#include <luthier/object/ELFObjectUtils.h>
#include <mutex>

namespace luthier {

static llvm::Error disassemble(llvm::ArrayRef<uint8_t> Code,
                               const llvm::object::ELFSymbolRef &Sym,
                               const llvm::MCDisassembler &DisAsm,
                               const llvm::MCAsmInfo &AsmInfo,
                               llvm::SmallVectorImpl<Instr> &Instructions) {
  size_t MaxReadSize = AsmInfo.getMaxInstLength();
  size_t Idx = 0;
  size_t CurrentOffset = 0;

  while (Idx < Code.size()) {
    size_t ReadSize =
        (Idx + MaxReadSize) < Code.size() ? MaxReadSize : Code.size() - Idx;
    size_t InstSize{};
    llvm::MCInst Inst;
    auto ReadBytes =
        arrayRefFromStringRef(toStringRef(Code).substr(Idx, ReadSize));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        DisAsm.getInstruction(Inst, InstSize, ReadBytes, CurrentOffset,
                              llvm::nulls()) == llvm::MCDisassembler::Success,
        "Failed to disassemble instruction at address {0:x}", CurrentOffset));
    CurrentOffset += InstSize;

    llvm::Expected<Instr> LuthierInst =
        Instr::create(Inst, Sym, CurrentOffset, InstSize);
    LUTHIER_RETURN_ON_ERROR(LuthierInst.takeError());

    Instructions.push_back(*LuthierInst);
    Idx += InstSize;
  }
  return llvm::Error::success();
}

llvm::Expected<Disassembler::DisassemblyInfo &>
Disassembler::getDisassemblyInfo(const llvm::object::ObjectFile &ObjFile) {
  llvm::Triple TT = ObjFile.makeTriple();

  std::optional<llvm::StringRef> CPUName = ObjFile.tryGetCPUName();

  llvm::SubtargetFeatures Features;
  LUTHIER_RETURN_ON_ERROR(ObjFile.getFeatures().moveInto(Features));

  return getDisassemblyInfo(TT, CPUName.has_value() ? *CPUName : "unknown",
                            Features);
}

llvm::Expected<Disassembler::DisassemblyInfo &>
Disassembler::getDisassemblyInfo(const llvm::Triple &TT,
                                 llvm::StringRef CPUName,
                                 const llvm::SubtargetFeatures &STF) {
  /// Construct the ISA string
  std::string ISAName = (TT.str() + "-" + CPUName + STF.getString()).str();

  /// Return the disassembly info if already created
  if (DisassemblyInfoMap.contains(ISAName))
    return *DisassemblyInfoMap[ISAName];

  /// Otherwise create a new disassembly info for the newly encountered
  /// ISA string

  auto DInfo = std::make_unique<DisassemblyInfo>();

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      !DInfo, "Failed to create a new disassembly info for ISA {0}.", ISAName));

  /// Find the LLVM target associated with TT
  std::string Error{};

  DInfo->Target = llvm::TargetRegistry::lookupTarget(TT.normalize(), Error);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->Target || !Error.empty(),
      "Failed to lookup target {0} in LLVM. Reason according to LLVM: {1}.",
      TT.normalize(), Error));

  /// Create a new MC register info object
  DInfo->MRI.reset(DInfo->Target->createMCRegInfo(TT.getTriple()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->MRI != nullptr, "Failed to create machine register info for {0}.",
      TT.getTriple()));

  /// Create a new Target options
  DInfo->TargetOptions = std::make_unique<llvm::TargetOptions>();

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->TargetOptions != nullptr,
      "Failed to create an llvm::TargetOptions for ISA {0}.", ISAName));

  /// Create an MC AssemblyInfo
  DInfo->MAI.reset(DInfo->Target->createMCAsmInfo(
      *DInfo->MRI, TT.getTriple(), DInfo->TargetOptions->MCOptions));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->MAI != nullptr,
      "Failed to create MCAsmInfo from target {0} for Target Triple {1}.",
      DInfo->Target->getName(), TT.getTriple()));

  /// Create a MC Instr Info
  DInfo->MII.reset(DInfo->Target->createMCInstrInfo());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->MII != nullptr, "Failed to create MCInstrInfo from target {0}",
      DInfo->Target->getName()));

  /// Create a new MC Instr Analysis
  DInfo->MIA.reset(DInfo->Target->createMCInstrAnalysis(DInfo->MII.get()));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->MIA != nullptr, "Failed to create MCInstrAnalysis for target {0}.",
      DInfo->Target->getName()));

  /// Create a new MC SubTargetInfo
  DInfo->STI.reset(DInfo->Target->createMCSubtargetInfo(TT.getTriple(), CPUName,
                                                        STF.getString()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->STI != nullptr,
      "Failed to create MCSubTargetInfo from target {0} "
      "for triple {1}, CPU {2}, with feature string {3}",
      DInfo->Target->getName(), TT.getTriple(), CPUName, STF.getString()));

  /// Create a new MC Instruction printer
  DInfo->IP.reset(DInfo->Target->createMCInstPrinter(
      TT, DInfo->MAI->getAssemblerDialect(), *DInfo->MAI, *DInfo->MII,
      *DInfo->MRI));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->IP != nullptr,
      "Failed to create MCInstPrinter from Target {0} for Triple {1}.",
      DInfo->Target->getName(), TT.getTriple()));

  /// Create a new MCContext
  DInfo->Ctx.reset(new (std::nothrow) llvm::MCContext(
      TT, DInfo->MAI.get(), DInfo->MRI.get(), DInfo->STI.get()));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      DInfo->Ctx != nullptr,
      "Failed to create MCContext for LLVM disassembly operation."));

  /// Create a new MCDisassembler
  DInfo->DisAsm.reset(
      DInfo->Target->createMCDisassembler(*DInfo->STI, *DInfo->Ctx));
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(DInfo->DisAsm != nullptr,
                          "Failed to create an MCDisassembler for the LLVM "
                          "disassembly operation."));

  /// Cache the newly constructed disassembly info and return it
  DisassemblyInfoMap.insert({ISAName, std::move(DInfo)});
  return *DisassemblyInfoMap[ISAName];
}

/// Disassembles the contents of the function-type \p Symbol and returns
/// a reference to its disassembled array of <tt>hsa::Instr</tt>s\n
/// Does not perform any symbolization or control flow analysis\n
/// The \c hsa::ISA of the backing \c hsa::LoadedCodeObject will be used to
/// disassemble the \p Symbol\n
/// The results of this operation gets cached on the first invocation
/// \tparam ST type of the loaded code object symbol; Must be of
/// type \p KERNEL or \p DEVICE_FUNCTION
/// \param Symbol the symbol to be disassembled
/// \return on success, a const reference to the cached disassembled
/// instructions; On failure, an \p llvm::Error
/// \sa hsa::Instr
llvm::Expected<llvm::ArrayRef<luthier::Instr>>
Disassembler::disassemble(const llvm::object::ELFSymbolRef &Symbol,
                          llvm::ArrayRef<uint8_t> LoadedContents) {
  /// Check if the object file is not a relocatable
  /// Check if the passed symbol is a function
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(Symbol.getELFType() == llvm::ELF::STT_FUNC,
                          "Requested disassembly of non-function symbol."));

  /// Get a shared lock; If the symbol is already disassembled, return the
  /// cached instructions
  {
    std::shared_lock Lock(Mutex);
    if (MCDisassembledSymbols.contains(Symbol))
      return *MCDisassembledSymbols.at(Symbol);
  }
  /// If the symbol is not disassembled, then get a unique lock, and
  /// start disassembling
  {
    std::unique_lock Lock(Mutex);
    /// Do another check after acquiring the unique lock to make sure the
    /// symbol was not already disassembled by another thread who had the
    /// unique lock before us
    if (MCDisassembledSymbols.contains(Symbol))
      return *MCDisassembledSymbols.at(Symbol);
    /// Start disassembling now that we're sure we have not disassembled ths
    /// symbol before
    // Get the ISA associated with the symbol's object

    // Obtain the symbol's start address that will be disassembled
    llvm::ArrayRef<uint8_t> SymbolContents;

    // If loaded contents is passed, get the loaded contents of the symbol;
    // Otherwise, use the contents of the ELF object file
    if (LoadedContents.data() != nullptr && LoadedContents.empty()) {
      llvm::Expected<std::optional<uint64_t>> SymbolLoadOffsetOrErr =
          getLoadOffset(Symbol);
      LUTHIER_RETURN_ON_ERROR(SymbolLoadOffsetOrErr.takeError());
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ERROR_CHECK(SymbolLoadOffsetOrErr->has_value(),
                              "Symbol does not have a load offset."));
      SymbolContents = {&LoadedContents[**SymbolLoadOffsetOrErr],
                        Symbol.getSize()};
    } else {
      LUTHIER_RETURN_ON_ERROR(getContents(Symbol).moveInto(SymbolContents));
    }

    // Create a new entry in the cache to hold the disassembled instructions
    MCDisassembledSymbols.insert(
        {Symbol, std::move(std::make_unique<llvm::SmallVector<Instr>>())});

    // Get the disassembly info
    const llvm::object::ObjectFile *ObjFile = Symbol.getObject();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        ObjFile != nullptr, "Symbol's object file is nullptr."));
    auto DisAsmInfoOrErr = getDisassemblyInfo(*ObjFile);
    LUTHIER_RETURN_ON_ERROR(DisAsmInfoOrErr.takeError());

    // Disassemble the instructions and return the results
    LUTHIER_RETURN_ON_ERROR(luthier::disassemble(
        SymbolContents, Symbol, *DisAsmInfoOrErr->DisAsm, *DisAsmInfoOrErr->MAI,
        *MCDisassembledSymbols[Symbol]));

    return *MCDisassembledSymbols[Symbol];
  }
}

} // namespace luthier