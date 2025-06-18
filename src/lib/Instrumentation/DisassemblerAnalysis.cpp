//===-- DisassemblerAnalysis.cpp ------------------------------------------===//
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
/// Implements the \c DisassemblerAnalysis pass.
//===----------------------------------------------------------------------===//
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/Instrumentation/DisassemblerAnalysis.h>
#include <luthier/Instrumentation/GlobalObjectSymbolsAnalysis.h>
#include <luthier/Instrumentation/LoadedContentsAnalysisPass.h>
#include <luthier/Object/ELFObjectUtils.h>

namespace luthier {

llvm::AnalysisKey DisassemblerAnalysis::Key;

static llvm::Expected<std::vector<Instr>>
disassemble(const llvm::object::ELFSymbolRef &Sym,
            const llvm::MCDisassembler &DisAsm, const llvm::MCAsmInfo &AsmInfo,
            std::optional<const uint8_t *> LoadedBase) {
  std::vector<Instr> Instructions;
  llvm::ArrayRef<uint8_t> Code;
  if (LoadedBase.has_value()) {
    auto LoadedOffsetOrErr = object::getLoadOffset(Sym);
    LUTHIER_RETURN_ON_ERROR(LoadedOffsetOrErr.takeError());
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_GENERIC_ERROR_CHECK(LoadedOffsetOrErr->has_value(),
                                    "Symbol does not have a loaded offset"));
    Code = llvm::ArrayRef{*LoadedBase, **LoadedOffsetOrErr};
  } else {
    LUTHIER_RETURN_ON_ERROR(object::getContents(Sym).moveInto(Code));
  }
  size_t MaxReadSize = AsmInfo.getMaxInstLength();
  size_t Idx = 0;
  size_t CurrentOffset = 0;

  while (Idx < Code.size()) {
    size_t ReadSize =
        (Idx + MaxReadSize) < Code.size() ? MaxReadSize : Code.size() - Idx;
    size_t InstSize{};
    llvm::MCInst Inst;
    auto ReadBytes = llvm::arrayRefFromStringRef(
        llvm::toStringRef(Code).substr(Idx, ReadSize));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
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
  return Instructions;
}

DisassemblerAnalysis::Result
DisassemblerAnalysis::run(llvm::MachineFunction &MF,
                          llvm::MachineFunctionAnalysisManager &MFAM) {
  /// Get the Module analysis
  const llvm::Module &M = *MF.getFunction().getParent();
  llvm::LLVMContext &Ctx = M.getContext();
  const auto &MAMProxy =
      MFAM.getResult<llvm::ModuleAnalysisManagerMachineFunctionProxy>(MF);
  /// Get the target machine and the MC Context
  const llvm::TargetMachine &TM = MF.getTarget();
  llvm::MCContext &MCCtx = MF.getContext();
  /// Get the symbol associated with the function
  auto GlobalValueSymbols =
      MAMProxy.getCachedResult<GlobalObjectSymbolsAnalysis>(M);
  if (!GlobalValueSymbols)
    Ctx.emitError("Failed to get the global value symbols analysis");
  std::optional<llvm::object::SymbolRef> SymRef =
      GlobalValueSymbols->getSymbolRef(MF.getFunction());
  if (!SymRef.has_value())
    Ctx.emitError(llvm::formatv(
        "Failed to obtain the object symbol ref associated with function {0}",
        MF.getName()));

  llvm::Expected<llvm::object::SymbolRef::Type> FuncSymTypeOrErr =
      SymRef->getType();
  if (llvm::Error Err =
          LUTHIER_LLVM_ERROR_CHECK(FuncSymTypeOrErr.takeError())) {
    Ctx.emitError(llvm::toString(std::move(Err)));
  }
  if (*FuncSymTypeOrErr != llvm::object::SymbolRef::ST_Function) {
    Ctx.emitError(
        llvm::formatv("Symbol associated with function {0} is not a function.",
                      MF.getName()));
  }

  /// Create a disassembler and disassemble the symbol's contents
  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TM.getTarget().createMCDisassembler(*TM.getMCSubtargetInfo(), MCCtx));

  const llvm::object::ObjectFile *ObjFile = SymRef->getObject();

  if (!ObjFile)
    Ctx.emitError(llvm::formatv(
        "Object file of the symbol ref for function {0} is nullptr",
        MF.getName()));

  std::optional<const uint8_t *> LoadBase{std::nullopt};

  if (auto LoadedContentsResult =
          MAMProxy.getCachedResult<LoadedContentsAnalysisPass>(M)) {
    LoadBase = LoadedContentsResult->getLoadedContents().data();
  }
  std::vector<Instr> Instructions;

  if (llvm::isa<llvm::object::ELFObjectFileBase>(ObjFile)) {
    LUTHIER_EMIT_ERROR_IN_CONTEXT(
        Ctx, disassemble(*SymRef, *DisAsm, *TM.getMCAsmInfo(), LoadBase)
                 .moveInto(Instructions));
  } else {
    Ctx.emitError(llvm::formatv(
        "Not yet implemented disassembly logic for object type {0}",
        ObjFile->getType()));
  }
  return Instructions;
}

} // namespace luthier