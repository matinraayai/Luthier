//===-- DisassembleAnalysisPass.cpp ---------------------------------------===//
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
/// Implements the \c DisassemblerAnalysisPass class.
//===----------------------------------------------------------------------===//
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/Instrumentation/DisassembleAnalysisPass.h>
#include <luthier/Instrumentation/GlobalValueSymbols.h>
#include <luthier/Instrumentation/LoadedContentsAnalysisPass.h>
#include <luthier/Object/ELFObjectUtils.h>

namespace luthier {

llvm::AnalysisKey DisassemblerAnalysisPass::Key;

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

DisassemblerAnalysisPass::Result
DisassemblerAnalysisPass::run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
  /// Get the Module analysis
  const llvm::Module &M = *F.getParent();
  llvm::LLVMContext &Ctx = M.getContext();
  const auto &MAMProxy =
      FAM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  /// Get the MMI, the target machine and the MC Context
  auto &MMI =
      MAMProxy.getCachedResult<llvm::MachineModuleAnalysis>(M)->getMMI();
  const llvm::TargetMachine &TM = MMI.getTarget();
  llvm::MCContext &MCCtx = MMI.getContext();
  /// Get the symbol associated with the function
  auto GlobalValueSymbols =
      MAMProxy.getCachedResult<GlobalValueSymbolsAnalysisPass>(M);
  if (!GlobalValueSymbols)
    Ctx.emitError("Failed to get the global value symbols analysis");
  std::optional<llvm::object::SymbolRef> SymRef =
      GlobalValueSymbols->getSymbolRef(F);
  if (!SymRef.has_value())
    Ctx.emitError(llvm::formatv(
        "Failed to obtain the object symbol ref associated with function {0}",
        F.getName()));

  llvm::Expected<llvm::object::SymbolRef::Type> FuncSymTypeOrErr =
      SymRef->getType();
  if (llvm::Error Err =
          LUTHIER_LLVM_ERROR_CHECK(FuncSymTypeOrErr.takeError())) {
    Ctx.emitError(llvm::toString(std::move(Err)));
  }
  if (*FuncSymTypeOrErr != llvm::object::SymbolRef::ST_Function) {
    Ctx.emitError(llvm::formatv(
        "Symbol associated with function {0} is not a function.", F.getName()));
  }

  /// Create a disassembler and disassemble the symbol's contents
  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TM.getTarget().createMCDisassembler(*TM.getMCSubtargetInfo(), MCCtx));

  const llvm::object::ObjectFile *ObjFile = SymRef->getObject();

  if (!ObjFile)
    Ctx.emitError(llvm::formatv(
        "Object file of the symbol ref for function {0} is nullptr",
        F.getName()));

  std::optional<const uint8_t *> LoadBase{std::nullopt};

  if (auto LoadedContentsResult =
          MAMProxy.getCachedResult<LoadedContentsAnalysisPass>(M)) {
    LoadBase = LoadedContentsResult->getLoadedContents().data();
  }

  llvm::Error Err = llvm::Error::success();
  std::vector<Instr> Instructions;

  switch (ObjFile->getType()) {
  case llvm::object::ID_ELF32L:
  case llvm::object::ID_ELF32B: // ELF 32-bit, big endian
  case llvm::object::ID_ELF64L: // ELF 64-bit, little endian
  case llvm::object::ID_ELF64B: // ELF 64-bit, big endian
    Err = disassemble(*SymRef, *DisAsm, *TM.getMCAsmInfo(), LoadBase)
              .moveInto(Instructions);
    break;
  case llvm::object::ID_COFF:
  case llvm::object::ID_XCOFF32:  // AIX XCOFF 32-bit
  case llvm::object::ID_XCOFF64:  // AIX XCOFF 64-bit
  case llvm::object::ID_MachO32L: // MachO 32-bit, little endian
  case llvm::object::ID_MachO32B: // MachO 32-bit, big endian
  case llvm::object::ID_MachO64L: // MachO 64-bit, little endian
  case llvm::object::ID_MachO64B: // MachO 64-bit, big endian
  case llvm::object::ID_GOFF:
  case llvm::object::ID_Wasm:
  default:
    Ctx.emitError(llvm::formatv(
        "Not yet implemented disassembly logic for object type {0}",
        ObjFile->getType()));
    ;
  }
  if (Err) {
    Ctx.emitError(llvm::toString(std::move(Err)));
  }
  return Instructions;
}

} // namespace luthier