//===- SIInstrSemanticsEmitter.cpp - SI Instruction Semantics Emitter
//------===//
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
/// Implementation of the TableGen backend that emits C++ code to translate
/// AMDGPU MachineInstrs into LLVM IR based on InstSISemantic and
/// InstSIIntrinsic records.
//===----------------------------------------------------------------------===//

#include "SIInstrSemanticsEmitter.hpp"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

using namespace llvm;

namespace luthier {

//===----------------------------------------------------------------------===//
// Utility: Map tablegen names to C++ expressions
//===----------------------------------------------------------------------===//

std::string SIInstrSemanticsEmitter::getTypeExpr(StringRef TypeName) {
  // Strip "llvm_" prefix and "_ty" suffix if present
  // e.g., "llvm_i32_ty" -> "i32", "llvm_float_ty" -> "float"
  return StringSwitch<std::string>(TypeName)
      .Case("llvm_i1_ty", "Builder.getInt1Ty()")
      .Case("llvm_i8_ty", "Builder.getInt8Ty()")
      .Case("llvm_i16_ty", "Builder.getInt16Ty()")
      .Case("llvm_i32_ty", "Builder.getInt32Ty()")
      .Case("llvm_i64_ty", "Builder.getInt64Ty()")
      .Case("llvm_half_ty", "Builder.getHalfTy()")
      .Case("llvm_float_ty", "Builder.getFloatTy()")
      .Case("llvm_double_ty", "Builder.getDoubleTy()")
      .Case("llvm_v2i16_ty",
            "llvm::FixedVectorType::get(Builder.getInt16Ty(), 2)")
      .Case("llvm_v2i32_ty",
            "llvm::FixedVectorType::get(Builder.getInt32Ty(), 2)")
      .Case("llvm_v3i32_ty",
            "llvm::FixedVectorType::get(Builder.getInt32Ty(), 3)")
      .Case("llvm_v4i32_ty",
            "llvm::FixedVectorType::get(Builder.getInt32Ty(), 4)")
      .Case("llvm_v2f16_ty",
            "llvm::FixedVectorType::get(Builder.getHalfTy(), 2)")
      .Case("llvm_v2f32_ty",
            "llvm::FixedVectorType::get(Builder.getFloatTy(), 2)")
      .Case("llvm_v4f32_ty",
            "llvm::FixedVectorType::get(Builder.getFloatTy(), 4)")
      .Default(formatv("/* unknown type: {0} */", TypeName));
}

std::string SIInstrSemanticsEmitter::getIRBuilderMethod(StringRef OpName) {
  return StringSwitch<std::string>(OpName)
      .Case("LLVMAdd", "CreateAdd")
      .Case("LLVMSub", "CreateSub")
      .Case("LLVMMul", "CreateMul")
      .Case("LLVMUDiv", "CreateUDiv")
      .Case("LLVMSDiv", "CreateSDiv")
      .Case("LLVMURem", "CreateURem")
      .Case("LLVMSRem", "CreateSRem")
      .Case("LLVMXor", "CreateXor")
      .Case("LLVMOr", "CreateOr")
      .Case("LLVMAnd", "CreateAnd")
      .Case("LLVMLShr", "CreateLShr")
      .Case("LLVMShl", "CreateShl")
      .Case("LLVMAShr", "CreateAShr")
      .Case("LLVMZExt", "CreateZExt")
      .Case("LLVMSExt", "CreateSExt")
      .Case("LLVMTrunc", "CreateTrunc")
      .Case("LLVMSelect", "CreateSelect")
      .Case("LLVMBitCast", "CreateBitCast")
      .Case("LLVMFAdd", "CreateFAdd")
      .Case("LLVMFSub", "CreateFSub")
      .Case("LLVMFMul", "CreateFMul")
      .Case("LLVMFDiv", "CreateFDiv")
      .Case("LLVMFRem", "CreateFRem")
      .Case("LLVMFMA", "CreateFMA")
      .Case("LLVMFNeg", "CreateFNeg")
      .Case("LLVMFPTrunc", "CreateFPTrunc")
      .Case("LLVMFPExt", "CreateFPExt")
      .Case("LLVMSIToFP", "CreateSIToFP")
      .Case("LLVMUIToFP", "CreateUIToFP")
      .Case("LLVMFPToSI", "CreateFPToSI")
      .Case("LLVMFPToUI", "CreateFPToUI")
      .Case("LLVMIntToPtr", "CreateIntToPtr")
      .Case("LLVMPtrAdd", "CreatePtrAdd")
      .Case("LLVMInsertElement", "CreateInsertElement")
      .Case("LLVMExtractElement", "CreateExtractElement")
      .Case("LLVMMaxNum", "CreateMaxNum")
      .Case("LLVMMinNum", "CreateMinNum")
      .Case("LLVMCtPop", "CreateCtPop")
      .Case("LLVMLdexp", "CreateLdexp")
      .Case("LLVMFCeil", "CreateFCeil")
      .Case("LLVMFFloor", "CreateFFloor")
      .Case("LLVMFTrunc", "CreateFTrunc")
      .Case("LLVMRoundEven", "CreateRoundEven")
      .Case("LLVMBitReverse", "CreateBitReverse")
      .Case("LLVMCtlz", "CreateCtlz")
      .Case("LLVMCttz", "CreateCttz")
      .Case("LLVMFSqrt", "CreateFSqrt")
      .Default("");
}

std::string SIInstrSemanticsEmitter::getCmpPredicate(StringRef PredName) {
  return StringSwitch<std::string>(PredName)
      // Integer predicates
      .Case("SETEQ", "llvm::CmpInst::ICMP_EQ")
      .Case("SETNE", "llvm::CmpInst::ICMP_NE")
      .Case("SETGT", "llvm::CmpInst::ICMP_SGT")
      .Case("SETGE", "llvm::CmpInst::ICMP_SGE")
      .Case("SETLT", "llvm::CmpInst::ICMP_SLT")
      .Case("SETLE", "llvm::CmpInst::ICMP_SLE")
      .Case("SETUGT", "llvm::CmpInst::ICMP_UGT")
      .Case("SETUGE", "llvm::CmpInst::ICMP_UGE")
      .Case("SETULT", "llvm::CmpInst::ICMP_ULT")
      .Case("SETULE", "llvm::CmpInst::ICMP_ULE")
      // Floating-point ordered predicates
      .Case("SETOEQ", "llvm::CmpInst::FCMP_OEQ")
      .Case("SETONE", "llvm::CmpInst::FCMP_ONE")
      .Case("SETOGT", "llvm::CmpInst::FCMP_OGT")
      .Case("SETOGE", "llvm::CmpInst::FCMP_OGE")
      .Case("SETOLT", "llvm::CmpInst::FCMP_OLT")
      .Case("SETOLE", "llvm::CmpInst::FCMP_OLE")
      .Case("SETO", "llvm::CmpInst::FCMP_ORD")
      // Floating-point unordered predicates
      .Case("SETUO", "llvm::CmpInst::FCMP_UNO")
      .Case("SETUEQ", "llvm::CmpInst::FCMP_UEQ")
      .Case("SETUNE", "llvm::CmpInst::FCMP_UNE")
      .Case("SETUGT", "llvm::CmpInst::FCMP_UGT")
      .Case("SETUGE", "llvm::CmpInst::FCMP_UGE")
      .Case("SETULT", "llvm::CmpInst::FCMP_ULT")
      .Case("SETULE", "llvm::CmpInst::FCMP_ULE")
      .Default(formatv("/* unknown predicate: {0} */", PredName));
}

std::string SIInstrSemanticsEmitter::getAtomicRMWOp(StringRef OpName) {
  return StringSwitch<std::string>(OpName)
      .Case("AtomicRMWAdd", "llvm::AtomicRMWInst::Add")
      .Case("AtomicRMWSub", "llvm::AtomicRMWInst::Sub")
      .Case("AtomicRMWAnd", "llvm::AtomicRMWInst::And")
      .Case("AtomicRMWOr", "llvm::AtomicRMWInst::Or")
      .Case("AtomicRMWXor", "llvm::AtomicRMWInst::Xor")
      .Case("AtomicRMWXchg", "llvm::AtomicRMWInst::Xchg")
      .Case("AtomicRMWMin", "llvm::AtomicRMWInst::Min")
      .Case("AtomicRMWUMin", "llvm::AtomicRMWInst::UMin")
      .Case("AtomicRMWMax", "llvm::AtomicRMWInst::Max")
      .Case("AtomicRMWUMax", "llvm::AtomicRMWInst::UMax")
      .Case("AtomicRMWNand", "llvm::AtomicRMWInst::Nand")
      .Default(formatv("/* unknown AtomicRMWOp: {0} */", OpName));
}

//===----------------------------------------------------------------------===//
// Per-function state management
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::resetFunctionState() {
  TmpCounter = 0;
  DefValMap.clear();
}

std::string SIInstrSemanticsEmitter::freshTmp() {
  return formatv("t{0}", TmpCounter++);
}

//===----------------------------------------------------------------------===//
// DAG expression code emitter
//===----------------------------------------------------------------------===//

std::string SIInstrSemanticsEmitter::emitDagExpr(raw_ostream &OS,
                                                 const DagInit *Dag,
                                                 StringRef Indent) {
  const Init *Op = Dag->getOperator();
  StringRef OpName;
  if (const auto *DI = dyn_cast<DefInit>(Op))
    OpName = DI->getDef()->getName();
  else
    PrintFatalError("DAG operator is not a def");

  // --- GetNamedOperand $name ---
  if (OpName == "GetNamedOperand") {
    StringRef ArgName = Dag->getArgNameStr(0);
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp << " = &Tracker.getOperandAsValue(\n"
       << Indent
       << "    *llvm::SIInstrInfo::getNamedOperand(MI, "
          "llvm::AMDGPU::OpName::"
       << ArgName << "));\n";
    return Tmp;
  }

  // --- GetVal $name ---
  if (OpName == "GetVal") {
    StringRef ArgName = Dag->getArgNameStr(0);
    auto It = DefValMap.find(ArgName);
    if (It == DefValMap.end())
      PrintFatalError("GetVal references undefined DefVal: " + ArgName);
    return It->second;
  }

  // --- ImplicitUse REG ---
  if (OpName == "ImplicitUse") {
    // First arg is a register def (e.g., VCC, EXEC, SCC)
    const auto *RegDef = dyn_cast<DefInit>(Dag->getArg(0));
    StringRef RegName = RegDef->getDef()->getName();
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp
       << " = &Tracker.getImplicitRegAsValue(MI, llvm::AMDGPU::" << RegName
       << ");\n";
    return Tmp;
  }

  // --- ConstantInt type, value ---
  if (OpName == "ConstantInt") {
    // (ConstantInt type, intval)
    const auto *TypeDef = dyn_cast<DefInit>(Dag->getArg(0));
    StringRef TypeName = TypeDef->getDef()->getName();
    std::string TypeExpr = getTypeExpr(TypeName);
    const auto *ValInit = Dag->getArg(1);
    std::string ValStr;
    if (const auto *II = dyn_cast<IntInit>(ValInit))
      ValStr = std::to_string(II->getValue());
    else
      ValStr = ValInit->getAsString();
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp << " = llvm::ConstantInt::get("
       << TypeExpr << ", " << ValStr << ");\n";
    return Tmp;
  }

  // --- ConstantTrue / ConstantFalse ---
  if (OpName == "ConstantTrue") {
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp
       << " = llvm::ConstantInt::getTrue(Ctx);\n";
    return Tmp;
  }
  if (OpName == "ConstantFalse") {
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp
       << " = llvm::ConstantInt::getFalse(Ctx);\n";
    return Tmp;
  }

  // --- LLVMCmp / LLVMFCmp pred, lhs, rhs ---
  if (OpName == "LLVMCmp" || OpName == "LLVMFCmp") {
    // First arg is predicate def, rest are value DAGs
    const auto *PredDef = dyn_cast<DefInit>(Dag->getArg(0));
    StringRef PredName = PredDef->getDef()->getName();
    std::string PredExpr = getCmpPredicate(PredName);
    std::string LHS =
        emitDagExpr(OS, dyn_cast<DagInit>(Dag->getArg(1)), Indent);
    std::string RHS =
        emitDagExpr(OS, dyn_cast<DagInit>(Dag->getArg(2)), Indent);
    std::string Tmp = freshTmp();
    std::string Method = (OpName == "LLVMFCmp") ? "CreateFCmp" : "CreateICmp";
    OS << Indent << "llvm::Value *" << Tmp << " = Builder." << Method << "("
       << PredExpr << ", " << LHS << ", " << RHS << ");\n";
    return Tmp;
  }

  // --- LLVMLoad type, addr ---
  if (OpName == "LLVMLoad") {
    const auto *TypeDef = dyn_cast<DefInit>(Dag->getArg(0));
    StringRef TypeName = TypeDef->getDef()->getName();
    std::string TypeExpr = getTypeExpr(TypeName);
    std::string Addr;
    // Second arg can be a DagInit or a name reference
    if (const auto *AddrDag = dyn_cast<DagInit>(Dag->getArg(1)))
      Addr = emitDagExpr(OS, AddrDag, Indent);
    else {
      StringRef AddrName = Dag->getArgNameStr(1);
      auto It = DefValMap.find(AddrName);
      if (It != DefValMap.end())
        Addr = It->second;
      else
        Addr = formatv("/* unresolved: {0} */", AddrName);
    }
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp << " = Builder.CreateLoad("
       << TypeExpr << ", " << Addr << ");\n";
    return Tmp;
  }

  // --- LLVMStore val, addr ---
  if (OpName == "LLVMStore") {
    std::string Val;
    if (const auto *ValDag = dyn_cast<DagInit>(Dag->getArg(0)))
      Val = emitDagExpr(OS, ValDag, Indent);
    else {
      StringRef ValName = Dag->getArgNameStr(0);
      auto It = DefValMap.find(ValName);
      if (It != DefValMap.end())
        Val = It->second;
      else
        Val = formatv("/* unresolved: {0} */", ValName);
    }
    std::string Addr;
    if (const auto *AddrDag = dyn_cast<DagInit>(Dag->getArg(1)))
      Addr = emitDagExpr(OS, AddrDag, Indent);
    else {
      StringRef AddrName = Dag->getArgNameStr(1);
      auto It = DefValMap.find(AddrName);
      if (It != DefValMap.end())
        Addr = It->second;
      else
        Addr = formatv("/* unresolved: {0} */", AddrName);
    }
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp << " = Builder.CreateStore(" << Val
       << ", " << Addr << ");\n";
    return Tmp;
  }

  // --- LLVMAtomicRMW op, addr, val ---
  if (OpName == "LLVMAtomicRMW") {
    const auto *RMWOpDef = dyn_cast<DefInit>(Dag->getArg(0));
    StringRef RMWOpName = RMWOpDef->getDef()->getName();
    std::string RMWOpExpr = getAtomicRMWOp(RMWOpName);
    std::string Addr;
    if (const auto *AddrDag = dyn_cast<DagInit>(Dag->getArg(1)))
      Addr = emitDagExpr(OS, AddrDag, Indent);
    else {
      StringRef AddrName = Dag->getArgNameStr(1);
      auto It = DefValMap.find(AddrName);
      Addr = (It != DefValMap.end()) ? It->second : "/* unresolved */";
    }
    std::string Val;
    if (const auto *ValDag = dyn_cast<DagInit>(Dag->getArg(2)))
      Val = emitDagExpr(OS, ValDag, Indent);
    else {
      StringRef ValName = Dag->getArgNameStr(2);
      auto It = DefValMap.find(ValName);
      Val = (It != DefValMap.end()) ? It->second : "/* unresolved */";
    }
    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp << " = Builder.CreateAtomicRMW("
       << RMWOpExpr << ", " << Addr << ", " << Val << ", llvm::MaybeAlign(), "
       << "llvm::AtomicOrdering::SequentiallyConsistent);\n";
    return Tmp;
  }

  // --- Standard binary/unary LLVMOps (CreateAdd, CreateFAdd, etc.) ---
  std::string Method = getIRBuilderMethod(OpName);
  if (!Method.empty()) {
    unsigned NumArgs = Dag->getNumArgs();

    // Collect all arguments
    SmallVector<std::string, 4> Args;
    for (unsigned i = 0; i < NumArgs; ++i) {
      const Init *Arg = Dag->getArg(i);
      if (const auto *SubDag = dyn_cast<DagInit>(Arg)) {
        Args.push_back(emitDagExpr(OS, SubDag, Indent));
      } else if (const auto *DefArg = dyn_cast<DefInit>(Arg)) {
        // This is a type argument (for casts like CreateBitCast(val, type))
        StringRef DefName = DefArg->getDef()->getName();
        // Check if it's a type
        if (DefName.starts_with("llvm_") && DefName.ends_with("_ty")) {
          Args.push_back(getTypeExpr(DefName));
        } else if (DefName.starts_with("flat_ptr_type") ||
                   DefName.starts_with("global_ptr_type") ||
                   DefName.starts_with("scratch_ptr_type") ||
                   DefName.starts_with("buffer_ptr_type") ||
                   DefName.starts_with("buffer_fat_ptr_type") ||
                   DefName.starts_with("region_ptr_type")) {
          // Pointer types defined in SISemOps.td
          // Map to llvm::PointerType::get(Ctx, addrspace)
          std::string AS = StringSwitch<std::string>(DefName)
                               .Case("flat_ptr_type", "0")
                               .Case("global_ptr_type", "1")
                               .Case("region_ptr_type", "2")
                               .Case("scratch_ptr_type", "5")
                               .Case("buffer_ptr_type", "8")
                               .Case("buffer_fat_ptr_type", "9")
                               .Default("0");
          Args.push_back(formatv("llvm::PointerType::get(Ctx, {0})", AS));
        } else {
          // Could be a predicate or other def — try as-is
          Args.push_back(DefName.str());
        }
      } else if (const auto *IntArg = dyn_cast<IntInit>(Arg)) {
        Args.push_back(std::to_string(IntArg->getValue()));
      } else {
        // Named argument reference (DefVal)
        StringRef ArgName = Dag->getArgNameStr(i);
        if (!ArgName.empty()) {
          auto It = DefValMap.find(ArgName);
          if (It != DefValMap.end())
            Args.push_back(It->second);
          else
            Args.push_back(formatv("/* unresolved: ${0} */", ArgName));
        } else {
          Args.push_back("/* unknown arg */");
        }
      }
    }

    std::string Tmp = freshTmp();
    OS << Indent << "llvm::Value *" << Tmp << " = Builder." << Method << "(";
    for (unsigned i = 0; i < Args.size(); ++i) {
      if (i > 0)
        OS << ", ";
      OS << Args[i];
    }
    OS << ");\n";
    return Tmp;
  }

  // --- Fallback: unknown operator ---
  std::string Tmp = freshTmp();
  OS << Indent << "llvm::Value *" << Tmp << " = nullptr; // TODO: unhandled op "
     << OpName << "\n";
  return Tmp;
}

//===----------------------------------------------------------------------===//
// Top-level semantic statement emitter
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::emitSemanticStatement(raw_ostream &OS,
                                                    const DagInit *Dag,
                                                    StringRef Indent) {
  const Init *Op = Dag->getOperator();
  StringRef OpName;
  if (const auto *DI = dyn_cast<DefInit>(Op))
    OpName = DI->getDef()->getName();
  else
    PrintFatalError("DAG operator is not a def");

  // --- SetNamedOperand $name, value_dag ---
  if (OpName == "SetNamedOperand") {
    StringRef OperandName = Dag->getArgNameStr(0);
    const auto *ValDag = dyn_cast<DagInit>(Dag->getArg(1));
    if (!ValDag)
      PrintFatalError("SetNamedOperand second arg is not a DAG");
    std::string ValVar = emitDagExpr(OS, ValDag, Indent);
    OS << Indent << "Tracker.setRegOperandValue(\n"
       << Indent
       << "    *llvm::SIInstrInfo::getNamedOperand(MI, "
          "llvm::AMDGPU::OpName::"
       << OperandName << "), *" << ValVar << ");\n";
    return;
  }

  // --- ImplicitDef REG, value_dag ---
  if (OpName == "ImplicitDef") {
    const auto *RegDef = dyn_cast<DefInit>(Dag->getArg(0));
    StringRef RegName = RegDef->getDef()->getName();
    const auto *ValDag = dyn_cast<DagInit>(Dag->getArg(1));
    if (!ValDag)
      PrintFatalError("ImplicitDef second arg is not a DAG");
    std::string ValVar = emitDagExpr(OS, ValDag, Indent);
    OS << Indent << "Tracker.setImplicitRegValue(MI, llvm::AMDGPU::" << RegName
       << ", *" << ValVar << ");\n";
    return;
  }

  // --- DefVal $name, value_dag ---
  if (OpName == "DefVal") {
    StringRef ValName = Dag->getArgNameStr(0);
    const auto *ValDag = dyn_cast<DagInit>(Dag->getArg(1));
    if (!ValDag)
      PrintFatalError("DefVal second arg is not a DAG");
    std::string ValVar = emitDagExpr(OS, ValDag, Indent);
    DefValMap[ValName] = ValVar;
    return;
  }

  // --- LLVMStore (bare top-level store, not inside SetNamedOperand) ---
  if (OpName == "LLVMStore") {
    emitDagExpr(OS, Dag, Indent);
    return;
  }

  // --- Fallback: emit as expression ---
  emitDagExpr(OS, Dag, Indent);
}

//===----------------------------------------------------------------------===//
// Function emitters
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::emitSemanticFunction(raw_ostream &OS,
                                                   const Record *Rec) {
  resetFunctionState();

  const Record *InstRec = Rec->getValueAsDef("Instruction");
  StringRef InstName = InstRec->getName();

  const auto *SemList = Rec->getValueAsListInit("Semantic");
  if (!SemList || SemList->empty())
    return; // Skip empty semantics

  OS << "template <>\n";
  OS << "void raiseMachineInstr<llvm::AMDGPU::" << InstName << ">(\n";
  OS << "    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,\n";
  OS << "    MBBOperandTracker &Tracker) {\n";
  OS << "  llvm::LLVMContext &Ctx = Builder.getContext();\n";
  OS << "  (void)Ctx; // May be unused for simple instructions\n";

  for (unsigned i = 0; i < SemList->size(); ++i) {
    const auto *StmtDag = dyn_cast<DagInit>(SemList->getElement(i));
    if (!StmtDag)
      PrintFatalError(Rec->getLoc(), "Semantic element is not a DAG");
    emitSemanticStatement(OS, StmtDag, "  ");
  }

  OS << "}\n\n";
}

void SIInstrSemanticsEmitter::emitIntrinsicFunction(raw_ostream &OS,
                                                    const Record *Rec) {
  const Record *InstRec = Rec->getValueAsDef("Instruction");
  StringRef InstName = InstRec->getName();

  const Record *IntrRec = Rec->getValueAsDef("Intrinsic");
  StringRef IntrName = IntrRec->getName();

  // Convert tablegen intrinsic name to LLVM enum
  // e.g., "int_amdgcn_interp_p1" -> "llvm::Intrinsic::amdgcn_interp_p1"
  std::string IntrEnum = IntrName.str();
  if (IntrEnum.starts_with("int_"))
    IntrEnum = "llvm::Intrinsic::" + IntrEnum.substr(4);

  OS << "template <>\n";
  OS << "void raiseMachineInstr<llvm::AMDGPU::" << InstName << ">(\n";
  OS << "    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,\n";
  OS << "    MBBOperandTracker &Tracker) {\n";
  OS << "  // Mapped to intrinsic: " << IntrName << "\n";
  OS << "  llvm::SmallVector<llvm::Value *, 8> Args;\n";
  OS << "  llvm::SmallVector<llvm::Type *, 4> OverloadTys;\n";
  OS << "  // Collect operand values from the MachineInstr\n";
  OS << "  for (const llvm::MachineOperand &MO : MI.explicit_uses()) {\n";
  OS << "    Args.push_back(&Tracker.getOperandAsValue(MO));\n";
  OS << "  }\n";
  OS << "  llvm::Value *Result = Builder.CreateIntrinsic(\n";
  OS << "      " << IntrEnum << ", OverloadTys, Args);\n";
  OS << "  // Set the output operand if the instruction has a def\n";
  OS << "  if (MI.getNumExplicitDefs() > 0) {\n";
  OS << "    Tracker.setRegOperandValue(MI.getOperand(0), *Result);\n";
  OS << "  }\n";
  OS << "}\n\n";
}

//===----------------------------------------------------------------------===//
// Dispatch function emitter
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::emitDispatchFunction(
    llvm::raw_ostream &OS, llvm::ArrayRef<const llvm::Record *> Semantics,
    llvm::ArrayRef<const llvm::Record *> Intrinsics) {
  OS << "/// Dispatch function: routes MI opcode to the appropriate\n";
  OS << "/// raiseMachineInstr<> specialization.\n";
  OS << "/// \\returns true if the instruction was successfully raised,\n";
  OS << "///          false if the opcode has no semantic model.\n";
  OS << "inline bool dispatchRaiseMachineInstr(\n";
  OS << "    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,\n";
  OS << "    MBBOperandTracker &Tracker) {\n";
  OS << "  switch (MI.getOpcode()) {\n";

  // Emit cases for InstSISemantic records (non-empty only)
  for (const Record *Rec : Semantics) {
    const auto *SemList = Rec->getValueAsListInit("Semantic");
    if (!SemList || SemList->empty())
      continue;
    const Record *InstRec = Rec->getValueAsDef("Instruction");
    StringRef InstName = InstRec->getName();
    OS << "  case llvm::AMDGPU::" << InstName << ":\n";
    OS << "    raiseMachineInstr<llvm::AMDGPU::" << InstName
       << ">(MI, Builder, Tracker);\n";
    OS << "    return true;\n";
  }

  // Emit cases for InstSIIntrinsic records
  for (const Record *Rec : Intrinsics) {
    const Record *InstRec = Rec->getValueAsDef("Instruction");
    StringRef InstName = InstRec->getName();
    OS << "  case llvm::AMDGPU::" << InstName << ":\n";
    OS << "    raiseMachineInstr<llvm::AMDGPU::" << InstName
       << ">(MI, Builder, Tracker);\n";
    OS << "    return true;\n";
  }

  OS << "  default:\n";
  OS << "    return false;\n";
  OS << "  }\n";
  OS << "}\n\n";
}

//===----------------------------------------------------------------------===//
// Header / Footer
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::emitHeader(raw_ostream &OS) {
  OS << "//===-- SIInstrSemantics.inc - Generated SI Instruction Semantics "
        "--------===//\n";
  OS << "// Auto-generated by luthier-tblgen. DO NOT EDIT.\n";
  OS << "//===---------------------------------------------------"
        "-------------------===//\n\n";
  OS << "#ifdef GET_SI_INSTR_SEMANTIC_FUNCTIONS\n\n";
}

void SIInstrSemanticsEmitter::emitFooter(raw_ostream &OS) {
  OS << "#endif // GET_SI_INSTR_SEMANTIC_FUNCTIONS\n\n";
  OS << "#ifdef GET_SI_INSTR_SEMANTIC_DISPATCH\n\n";
}

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::run(raw_ostream &OS) {
  // Collect all InstSISemantic records
  std::vector<const Record *> Semantics;
  for (const auto *Rec : Records.getAllDerivedDefinitions("InstSISemantic")) {
    Semantics.push_back(Rec);
  }

  // Collect all InstSIIntrinsic records
  std::vector<const Record *> Intrinsics;
  for (const auto *Rec : Records.getAllDerivedDefinitions("InstSIIntrinsic")) {
    Intrinsics.push_back(Rec);
  }

  // --- Emit template specializations ---
  emitHeader(OS);

  // Forward declare the template
  OS << "template <uint16_t Opcode>\n";
  OS << "void raiseMachineInstr(const llvm::MachineInstr &MI,\n";
  OS << "                       llvm::IRBuilderBase &Builder,\n";
  OS << "                       MBBOperandTracker &Tracker);\n\n";

  // Emit specializations for InstSISemantic (non-empty only)
  for (const Record *Rec : Semantics) {
    const auto *SemList = Rec->getValueAsListInit("Semantic");
    if (SemList && !SemList->empty())
      emitSemanticFunction(OS, Rec);
  }

  // Emit specializations for InstSIIntrinsic
  for (const Record *Rec : Intrinsics) {
    emitIntrinsicFunction(OS, Rec);
  }

  OS << "#endif // GET_SI_INSTR_SEMANTIC_FUNCTIONS\n\n";

  // --- Emit dispatch function ---
  OS << "#ifdef GET_SI_INSTR_SEMANTIC_DISPATCH\n\n";
  emitDispatchFunction(OS, Semantics, Intrinsics);
  OS << "#endif // GET_SI_INSTR_SEMANTIC_DISPATCH\n";
}

//===----------------------------------------------------------------------===//
// Global entry point
//===----------------------------------------------------------------------===//

void emitSIInstrSemantics(const RecordKeeper &Records, raw_ostream &OS) {
  SIInstrSemanticsEmitter Emitter(Records);
  Emitter.run(OS);
}

} // namespace luthier
