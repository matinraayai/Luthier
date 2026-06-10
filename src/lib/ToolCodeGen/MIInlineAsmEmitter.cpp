//===-- MIInlineAsmEmitter.cpp --------------------------------------------===//
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
/// \file MIInlineAsmEmitter.cpp
/// Implements the \c MIInlineAsmEmitter class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MIInlineAsmEmitter.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/MIRToIRTranslator.h"
#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-inline-asm-emitter"

namespace luthier {

MIInlineAsmEmitter::MIInlineAsmEmitter(llvm::TargetMachine &TM)
    : MCCtx(std::make_unique<llvm::MCContext>(
          TM.getTargetTriple(), TM.getMCAsmInfo(), TM.getMCRegisterInfo(),
          TM.getMCSubtargetInfo(), nullptr, /*DoAutoReset=*/false)),
      TM(TM) {}

llvm::Expected<std::unique_ptr<MIInlineAsmEmitter>>
MIInlineAsmEmitter::get(llvm::TargetMachine &TM) {
  auto Emitter =
      std::unique_ptr<MIInlineAsmEmitter>(new MIInlineAsmEmitter(TM));
  Emitter->AsmStringOS =
      std::make_unique<llvm::raw_svector_ostream>(Emitter->AsmString);

  llvm::Expected<std::unique_ptr<llvm::MCStreamer>> MCSorErr =
      TM.createMCStreamer(*Emitter->AsmStringOS, nullptr,
                          llvm::CodeGenFileType::AssemblyFile, *Emitter->MCCtx);
  LUTHIER_RETURN_ON_ERROR(MCSorErr.takeError());

  llvm::AsmPrinter *AsmPrinter =
      TM.getTarget().createAsmPrinter(TM, std::move(*MCSorErr));
  if (!AsmPrinter)
    return LUTHIER_MAKE_GENERIC_ERROR("Failed to create assembly printer");

  Emitter->AP.reset(AsmPrinter);

  Emitter->IP.reset(TM.getTarget().createMCInstPrinter(
      TM.getTargetTriple(), TM.getMCAsmInfo().getAssemblerDialect(),
      TM.getMCAsmInfo(), *TM.getMCInstrInfo(), TM.getMCRegisterInfo()));

  return std::move(Emitter);
}

std::string MIInlineAsmEmitter::emitAsmString(const llvm::MachineInstr &MI) {
  AP->emitInstruction(&MI);
  std::string Out{AsmString};
  AsmString.clear();
  return Out;
}

std::string MIInlineAsmEmitter::getRegisterName(llvm::MCRegister Reg) {
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  IP->printRegName(OS, Reg);
  return Out;
}

void MIInlineAsmEmitter::emitInlineAsm(
    llvm::IRBuilderBase &Builder, const llvm::MachineInstr &MI,
    const std::function<llvm::Value &(llvm::MCRegister)> &InputRegValMap,
    const std::function<void(llvm::MCRegister, llvm::Value &)>
        &OutputRegValMap) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");
  const llvm::MachineFunction *MF = MBB->getParent();
  assert(MF && "MBB has no parent MF");
  AP->MF = const_cast<llvm::MachineFunction *>(MF);

  const llvm::SIRegisterInfo &TRI =
      *MF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();

  std::string AsmStr = emitAsmString(MI);

  LLVM_DEBUG(luthier::dbgs()
             << "[InlineAsmEmitter] Obtained instruction asm string: " << AsmStr
             << "\n");

  struct RegOperandInfo {
    const llvm::MachineOperand *Op;
    unsigned InlineAsmIdx;
    std::string Constraint;
  };

  auto GetRegConstraint = [&](llvm::MCRegister Reg) -> std::string {
    if (!TRI.isInAllocatableClass(Reg))
      return "r";
    const llvm::TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
    if (llvm::SIRegisterInfo::isAGPRClass(RC)) {
      return "a";
    } else if (llvm::SIRegisterInfo::isVGPRClass(RC)) {
      return "v";
    } else if (llvm::SIRegisterInfo::isSGPRClass(RC)) {
      return "s";
    } else
      return "r";
  };

  llvm::SmallVector<RegOperandInfo> Defs;
  llvm::SmallVector<RegOperandInfo> Uses;

  unsigned OpIdx = 0;

  for (const llvm::MachineOperand &Op : MI.all_defs()) {
    if (!TRI.isInAllocatableClass(Op.getReg()))
      continue;
    std::string RegConstraint = '=' + GetRegConstraint(Op.getReg().asMCReg());

    LLVM_DEBUG(llvm::StringRef RegName = TRI.getRegAsmName(Op.getReg());
               luthier::dbgs()
               << "[MIRToIRTranslator] Def Operand info: "
               << "Op idx: " << OpIdx << ", "
               << "Reg name: " << RegName << ", constraint: " << RegConstraint
               << ", is implicit: " << Op.isImplicit() << "\n");

    Defs.push_back({&Op, OpIdx, RegConstraint});
    OpIdx++;
  }

  for (const llvm::MachineOperand &Op : MI.all_uses()) {
    if (!TRI.isInAllocatableClass(Op.getReg()))
      continue;
    std::string RegConstraint = GetRegConstraint(Op.getReg().asMCReg());

    LLVM_DEBUG(llvm::StringRef RegName = TRI.getRegAsmName(Op.getReg());
               luthier::dbgs()
               << "[MIRToIRTranslator] Use Operand info: "
               << "Op idx: " << OpIdx << ", "
               << "Reg name: " << RegName << ", constraint: " << RegConstraint
               << ", is implicit: " << Op.isImplicit() << "\n");

    Uses.push_back({&Op, OpIdx, RegConstraint});
    OpIdx++;
  }

  /// Word-boundary-aware token check: a register-name "match" is only a real
  /// token when the surrounding characters are not identifier-continuation
  /// characters. Prevents e.g. "v0" from matching inside "v00" or "myv0_lo".
  auto IsIdentCont = [](char C) {
    return std::isalnum(static_cast<unsigned char>(C)) || C == '_';
  };

  for (const auto &RegOp : llvm::concat<RegOperandInfo>(Defs, Uses)) {
    size_t Pos = 0;
    llvm::StringRef RegAsmName = TRI.getRegAsmName(RegOp.Op->getReg());
    while ((Pos = AsmStr.find(RegAsmName, Pos)) != std::string::npos) {
      bool LeftOK = Pos == 0 || !IsIdentCont(AsmStr[Pos - 1]);
      size_t End = Pos + RegAsmName.size();
      bool RightOK = End == AsmStr.size() || !IsIdentCont(AsmStr[End]);
      if (!LeftOK || !RightOK) {
        Pos = End;
        continue;
      }
      std::string AsmOpIdxAsStr = llvm::formatv("${0}", RegOp.InlineAsmIdx);
      AsmStr.replace(Pos, RegAsmName.size(), AsmOpIdxAsStr);
      Pos += AsmOpIdxAsStr.size();
    }
  }

  std::string ConstraintStr;
  for (const auto &RegOp : llvm::concat<RegOperandInfo>(Defs, Uses)) {
    if (!ConstraintStr.empty())
      ConstraintStr += ",";
    ConstraintStr += RegOp.Constraint;
  }

  llvm::SmallVector<llvm::Value *, 8> Operands;
  llvm::SmallVector<llvm::Type *, 8> ArgTys;
  for (const auto &RegOp : Uses) {
    llvm::Value &Val = InputRegValMap(RegOp.Op->getReg());
    Operands.push_back(&Val);
    ArgTys.push_back(Val.getType());
  }

  llvm::Type *RetTy = [&]() -> llvm::Type * {
    unsigned NumDefs = Defs.size();
    if (NumDefs == 0)
      return Builder.getVoidTy();
    else if (NumDefs == 1) {
      return Builder.getIntNTy(TRI.getRegSizeInBits(
          *TRI.getMinimalPhysRegClass(Defs[0].Op->getReg())));
    } else {
      llvm::SmallVector<llvm::Type *, 2> OutTypes;
      for (const auto &Def : Defs) {
        OutTypes.push_back(Builder.getIntNTy(TRI.getRegSizeInBits(
            *TRI.getMinimalPhysRegClass(Def.Op->getReg()))));
      }
      return llvm::StructType::get(Builder.getContext(), OutTypes);
    }
  }();

  llvm::FunctionType *FTy = llvm::FunctionType::get(RetTy, ArgTys, false);
  llvm::InlineAsm *IA = llvm::InlineAsm::get(FTy, AsmStr, ConstraintStr, true);

  llvm::CallInst *CI = Builder.CreateCall(IA, Operands);
  CI->addAttributeAtIndex(llvm::AttributeList::FunctionIndex,
                          llvm::Attribute::NoUnwind);

  if (Defs.size() == 1) {
    OutputRegValMap(Defs[0].Op->getReg(), *CI);
  } else if (Defs.size() > 1) {
    for (const auto &[Idx, Def] : llvm::enumerate(Defs)) {
      llvm::Value *DefVal = Builder.CreateExtractValue(CI, Idx);
      OutputRegValMap(Def.Op->getReg(), *DefVal);
    }
  }
}

} // namespace luthier