//===- SIInstrSemanticsEmitter.cpp - SI Instruction Semantics Emitter -----===//
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
/// \file SIInstrSemanticsEmitter.cpp
/// TableGen backend that emits C++ code to translate AMDGPU \c MachineInstr
/// instances into LLVM IR based on their \c InstSISemantic records.
//===----------------------------------------------------------------------===//
#include "SIInstrSemanticsEmitter.hpp"
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

namespace luthier {

/// Converts a record of type \c LLVMType into its appropriate LLVM constructor
static std::string getTypeExpr(llvm::ArrayRef<llvm::SMLoc> Loc,
                               const llvm::Record *Type) {
  const llvm::Record *ValueType = Type->getValueAsDef("VT");
  bool IsFP = ValueType->getValueAsBit("isFP");
  bool IsInt = ValueType->getValueAsBit("isInteger");
  llvm::StringRef TypeName = Type->getName();

  std::string Builder{};

  if (IsInt) {
    int64_t Size = ValueType->getValueAsInt("Size");
    Builder += llvm::StringSwitch<std::string>(TypeName)
                   .Case("llvm_i1_ty", "Builder.getInt1Ty()")
                   .Case("llvm_i8_ty", "Builder.getInt8Ty()")
                   .Case("llvm_i16_ty", "Builder.getInt16Ty()")
                   .Case("llvm_i32_ty", "Builder.getInt32Ty()")
                   .Case("llvm_i64_ty", "Builder.getInt64Ty()")
                   .Case("llvm_i128_ty", "Builder.getInt128Ty()")
                   .Default(llvm::formatv("Builder.getIntNTy({0})", Size));
  } else if (IsFP) {
    Builder += llvm::StringSwitch<std::string>(TypeName)
                   .Case("llvm_half_ty", "Builder.getHalfTy()")
                   .Case("llvm_float_ty", "Builder.getFloatTy()")
                   .Case("llvm_double_ty", "Builder.getDoubleTy()")
                   .Case("llvm_bfloat_ty", "Builder.getBFloatTy()");
  } else if (TypeName == "llvm_void_ty") {
    return "Builder.getVoidTy()";
  } else {
    llvm::PrintFatalError(Loc, "Unhandled type " + Type->getName() + ".");
  }

  if (ValueType->getValueAsBit("isVector")) {
    int64_t NumEls = ValueType->getValueAsInt("nElem");
    Builder =
        llvm::formatv("llvm::FixedVectorType::get({0}, {1})", Builder, NumEls);
  }

  return Builder;
}

void SIInstrSemanticsEmitter::emitSemanticStatement(
    llvm::raw_ostream &OS, const llvm::Init *Stmt,
    llvm::ArrayRef<llvm::SMLoc> Loc) {
  if (const auto *Dag = llvm::dyn_cast<llvm::DagInit>(Stmt)) {
    Dag->print(llvm::errs());
    llvm::errs() << "\n";
    const llvm::Record *Op = Dag->getOperatorAsDef(Loc);
    llvm::StringRef OpName = Op->getName();
    const llvm::RecordRecTy *OpClass = Op->getType();

    // --- SetNamedOperand $name, value_dag ---
    if (OpName == "SetNamedOperand") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 2)
        llvm::PrintFatalError(
            Loc, "Expected `SetNamedOperand` to have 2 arguments, got " +
                     llvm::Twine(NumArgs) + " instead");
      llvm::StringRef OperandName = Dag->getArgNameStr(0);
      const auto *AssignmentDag = llvm::dyn_cast<llvm::DagInit>(Dag->getArg(1));
      if (!AssignmentDag)
        llvm::PrintFatalError(Loc, "`SetNamedOperand` second arg is not a DAG");
      OS << "Tracker.setRegOperandValue(";
      OS << "*Tracker.getTII().getNamedOperand(MI, "
            "llvm::AMDGPU::OpName::"
         << OperandName << "), ";
      emitSemanticStatement(OS, AssignmentDag, Loc);
      OS << ")";
    }
    // --- ImplicitDef REG, value_dag ---
    else if (OpName == "ImplicitDef") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 2)
        llvm::PrintFatalError(
            Loc, "Expected `ImplicitDef` to have 2 arguments, got " +
                     llvm::Twine(NumArgs) + " instead");

      const auto *RegDef = llvm::dyn_cast<llvm::DefInit>(Dag->getArg(0));
      if (!RegDef)
        llvm::PrintFatalError(
            Loc, "First argument of `ImplicitDef` is not a record definition");

      const llvm::Record *RegisterClass =
          RegDef->getRecordKeeper().getClass("Register");
      if (!RegisterClass) {
        llvm::PrintFatalError(Loc, "Failed to get SIReg");
      }
      if (!RegDef->getDef()->isSubClassOf(RegisterClass)) {
        llvm::PrintFatalError(
            Loc, "First argument of `ImplicitDef` is not an Register");
      }
      llvm::StringRef RegName = RegDef->getDef()->getName();

      const auto *ValDag = llvm::dyn_cast<llvm::DagInit>(Dag->getArg(1));
      if (!ValDag)
        llvm::PrintFatalError("ImplicitDef second arg is not a DAG");
      OS << "Tracker.setRegOperandValue(MI, llvm::AMDGPU::" << RegName << ", ";
      emitSemanticStatement(OS, ValDag, Loc);
      OS << ")";
    }
    // --- DefVal $name, value_dag ---
    else if (OpName == "DefVal") {
      llvm::StringRef ValName = Dag->getArgNameStr(0);
      const auto *ValDag = llvm::dyn_cast<llvm::DagInit>(Dag->getArg(1));
      if (!ValDag)
        llvm::PrintFatalError("DefVal second arg is not a DAG");
      OS << "llvm::Value *" << ValName << " = ";
      emitSemanticStatement(OS, ValDag, Loc);
    }

    // --- GetNamedOperand $name ---
    else if (OpName == "GetNamedOperand") {
      llvm::StringRef ArgName = Dag->getArgNameStr(0);
      OS << "&Tracker.getOperandAsValue("
         << "*Tracker.getTII().getNamedOperand(MI, "
            "llvm::AMDGPU::OpName::"
         << ArgName << "))";
    }

    // --- GetVal $name ---
    else if (OpName == "GetVal") {
      llvm::errs() << "Getval: " << Dag->getArgNameStr(0) << "\n";
      OS << Dag->getArgNameStr(0);
    }

    // --- ImplicitUse REG ---
    else if (OpName == "ImplicitUse") {
      // First arg is a register def (e.g., VCC, EXEC, SCC)
      const auto *RegDef = llvm::dyn_cast<llvm::DefInit>(Dag->getArg(0));
      llvm::StringRef RegName = RegDef->getDef()->getName();
      OS << "&Tracker.getRegisterOperand(MI, llvm::AMDGPU::" << RegName << ")";
    }

    // --- LLVM Operands ---
    else if (OpClass->isSubClassOf(Records.getClass("LLVMOp"))) {
      OS << "Builder." << Op->getValueAsString("IRBuilderFunc") << "(";
      llvm::interleave(
          Dag->getArgs(),
          [&](const llvm::Init *Arg) { emitSemanticStatement(OS, Arg, Loc); },
          [&] { OS << ", "; });
      OS << ")";
    }

    // --- Constant Values ---
    else if (OpClass->isSubClassOf(Records.getClass("LLVMConstant"))) {
      OS << Op->getValueAsString("BuilderFunc") << "(";
      llvm::interleave(
          Dag->getArgs(),
          [&](const llvm::Init *Arg) { emitSemanticStatement(OS, Arg, Loc); },
          [&] { OS << ", "; });
      OS << ")";
    } else {
      llvm::PrintFatalError(Loc, "Unhandled operand: " + OpName + ".");
    }
  } else if (const auto *DefNode = llvm::dyn_cast<llvm::DefInit>(Stmt)) {
    const llvm::RecordRecTy *DefType = DefNode->getDef()->getType();

    /// LLVM Comparison predicates
    if (DefType->isSubClassOf(
            Stmt->getRecordKeeper().getClass("LLVMCmpPredicate"))) {
      OS << DefNode->getDef()->getValueAsString("Code");
    }

    /// LLVM Atomic ops
    else if (DefType->isSubClassOf(
                 Stmt->getRecordKeeper().getClass("AtomicRMWOp"))) {
      OS << DefNode->getDef()->getValueAsString("Op");
    }
    /// Pointer type operands
    else if (DefType->isSubClassOf(
                 Stmt->getRecordKeeper().getClass("LLVMQualPointerType"))) {
      const llvm::Record *PtrRecord = DefNode->getDef();
      const llvm::ListInit *PtrSig = PtrRecord->getValueAsListInit("Sig");
      OS << "Builder.getPtrTy("
         << (PtrSig->size() == 2 ? PtrSig->getElement(1)->getAsString() : "0")
         << ")";
    } else if (DefType->isSubClassOf(
                   Stmt->getRecordKeeper().getClass("LLVMType"))) {
      llvm::errs() << "Leaf type: " << DefNode->getDef()->getName() << "\n";
      OS << getTypeExpr(Loc, DefNode->getDef());
    } else if (DefType->isSubClassOf(
                   Stmt->getRecordKeeper().getClass("Intrinsic"))) {
      llvm::errs() << "Leaf type: " << DefNode->getDef()->getName() << "\n";
      llvm::StringRef IntrinsicName = DefNode->getDef()->getName();
      if (!IntrinsicName.consume_front("int_")) {
        llvm::PrintFatalError(Loc, "Intrinsic record's name " + IntrinsicName +
                                       "does not start with 'int_'");
      }
      llvm::errs() << "Final name of the intrinsic: " << IntrinsicName << "\n";
      OS << "llvm::Intrinsic::" << IntrinsicName;
    } else {
      llvm::PrintFatalError(
          Loc, "Unhandled def node: " + DefNode->getAsString() + ".");
    }
  } else if (const auto *ListNode = llvm::dyn_cast<llvm::ListInit>(Stmt)) {
    OS << "llvm::ArrayRef<llvm::Value *>({";
    llvm::interleave(
        ListNode->getElements(),
        [&](const llvm::Init *Elem) { emitSemanticStatement(OS, Elem, Loc); },
        [&] { OS << ", "; });
    OS << "})";
  } else if (auto *IntNode = llvm::dyn_cast<llvm::IntInit>(Stmt)) {
    llvm::outs() << "Int node: " << IntNode->getValue() << "\n";
    OS << IntNode->getValue();
  } else if (auto *StrNode = llvm::dyn_cast<llvm::StringInit>(Stmt)) {
    llvm::outs() << "String node: " << StrNode->getValue() << "\n";
    OS << StrNode->getValue();
  } else {
    llvm::PrintFatalError(Loc, "Unhandled node: " + Stmt->getAsString() + ".");
  }
}

void SIInstrSemanticsEmitter::emitSemanticFunction(llvm::raw_ostream &OS,
                                                   const llvm::Record *Rec) {

  const llvm::Record *InstRec = Rec->getValueAsDef("Instruction");
  llvm::StringRef InstName = InstRec->getName();

  const llvm::ListInit *SemList = Rec->getValueAsListInit("Semantic");

  llvm::SMLoc SemanticLoc = Rec->getFieldLoc("Semantic");

  /// Skip instructions with empty semantic
  if (!SemList || SemList->empty())
    return; // Skip empty semantics

  OS << "template <>\n";
  OS << "void inline raiseMachineInstr<llvm::AMDGPU::" << InstName << ">(\n";
  OS << "    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,\n";
  OS << "    MBBOperandTracker &Tracker) {\n";

  llvm::errs() << "Record: " << Rec;
  for (const llvm::Init *El : SemList->getElements()) {
    const auto *StmtDag = llvm::dyn_cast<llvm::DagInit>(El);
    if (!StmtDag)
      PrintFatalError(Rec->getLoc(), "Semantic element is not a DAG");
    emitSemanticStatement(OS.indent(2), StmtDag, SemanticLoc);
    OS << ";\n";
  }

  OS << "}\n\n";
}

void SIInstrSemanticsEmitter::emitIntrinsicFunction(llvm::raw_ostream &OS,
                                                    const llvm::Record *Rec) {
  const llvm::Record *InstRec = Rec->getValueAsDef("Instruction");
  llvm::StringRef InstName = InstRec->getName();

  const llvm::Record *IntrRec = Rec->getValueAsDef("Intrinsic");
  llvm::StringRef IntrName = IntrRec->getName();

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
  OS << "static void raiseMachineInstr(\n";
  OS << "    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,\n";
  OS << "    MBBOperandTracker &Tracker) {\n";
  OS << "  uint16_t Opcode = MI.getOpcode();\n";
  OS << "  switch (MI.getOpcode()) {\n";

  // Emit cases for InstSISemantic records (non-empty only)
  for (const llvm::Record *Rec : Semantics) {
    const auto *SemList = Rec->getValueAsListInit("Semantic");
    if (!SemList || SemList->empty())
      continue;
    const llvm::Record *InstRec = Rec->getValueAsDef("Instruction");
    llvm::StringRef InstName = InstRec->getName();
    OS << "  case llvm::AMDGPU::" << InstName << ":\n";
    OS << "    return raiseMachineInstr<llvm::AMDGPU::" << InstName
       << ">(MI, Builder, Tracker);\n";
  }

  // // Emit cases for InstSIIntrinsic records
  // for (const llvm::Record *Rec : Intrinsics) {
  //   const llvm::Record *InstRec = Rec->getValueAsDef("Instruction");
  //   llvm::StringRef InstName = InstRec->getName();
  //   OS << "  case llvm::AMDGPU::" << InstName << ":\n";
  //   OS << "    raiseMachineInstr<llvm::AMDGPU::" << InstName
  //      << ">(MI, Builder, Tracker);\n";
  //   OS << "    return true;\n";
  // }

  OS << "  default:\n";
  OS << "    return llvm_unreachable(llvm::formatv(\"Unmodeled Opcode: {0}.\", "
        "Opcode).str().c_str());\n";
  OS << "  }\n";
  OS << "}\n\n";
}

//===----------------------------------------------------------------------===//
// Header / Footer
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::emitHeader(llvm::raw_ostream &OS) {
  OS << "//===-- SIInstrSemantics.inc - Generated SI Instruction Semantics "
        "---------===//\n";
  OS << "// Auto-generated by luthier-tblgen. DO NOT EDIT.\n";
  OS << "//===---------------------------------------------------"
        "-------------------===//\n\n";
}

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::run(llvm::raw_ostream &OS) {
  llvm::ArrayRef<const llvm::Record *> Semantics =
      Records.getAllDerivedDefinitions("InstSISemantic");

  // Collect all InstSIIntrinsic records
  llvm::ArrayRef<const llvm::Record *> Intrinsics =
      Records.getAllDerivedDefinitions("InstSIIntrinsic");

  emitHeader(OS);

  /// --- Emit template specializations ---
  OS << "#ifdef GET_SI_INSTR_SEMANTIC_FUNCTIONS\n";
  OS << "#undef GET_SI_INSTR_SEMANTIC_FUNCTIONS\n";

  for (const llvm::Record *Rec : Semantics) {
    emitSemanticFunction(OS, Rec);
  }

  OS << "#endif // GET_SI_INSTR_SEMANTIC_FUNCTIONS\n\n";

  //
  // // Emit specializations for InstSIIntrinsic
  // for (const llvm::Record *Rec : Intrinsics) {
  //   emitIntrinsicFunction(OS, Rec);
  // }
  //
  // OS << "#endif // GET_SI_INSTR_SEMANTIC_FUNCTIONS\n\n";
  //
  // --- Emit dispatch function ---
  OS << "#ifdef GET_SI_INSTR_SEMANTIC_DISPATCH\n\n";
  emitDispatchFunction(OS, Semantics, Intrinsics);
  OS << "#undef GET_SI_INSTR_SEMANTIC_DISPATCH\n";
}

//===----------------------------------------------------------------------===//
// Global entry point
//===----------------------------------------------------------------------===//

void emitSIInstrSemantics(const llvm::RecordKeeper &Records,
                          llvm::raw_ostream &OS) {
  SIInstrSemanticsEmitter Emitter(Records);
  Emitter.run(OS);
}

} // namespace luthier
