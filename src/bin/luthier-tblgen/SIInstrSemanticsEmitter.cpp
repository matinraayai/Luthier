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

void SIInstrSemanticsEmitter::emitRegisterReference(
    llvm::raw_ostream &OS, const llvm::DefInit *RegDef,
    llvm::ArrayRef<llvm::SMLoc> Loc) {
  const llvm::Record *Def = RegDef->getDef();
  const auto &RK = RegDef->getRecordKeeper();

  if (const llvm::Record *RegisterClass = RK.getClass("Register");
      RegisterClass && Def->isSubClassOf(RegisterClass)) {
    OS << "llvm::AMDGPU::" << Def->getName();
    return;
  }

  if (const llvm::Record *SRFClass = RK.getClass("SuperRegFactory");
      SRFClass && Def->isSubClassOf(SRFClass)) {
    std::vector<const llvm::Record *> SubRegs =
        Def->getValueAsListOfDefs("Regs");
    OS << "llvm::AMDGPU::";
    llvm::interleave(
        SubRegs, [&](const llvm::Record *SubReg) { OS << SubReg->getName(); },
        [&] { OS << "_"; });
    return;
  }

  llvm::PrintFatalError(Loc, "Register operand " + Def->getName() +
                                 " is neither a `Register` nor a "
                                 "`SuperRegFactory<[...]>` record");
}

void SIInstrSemanticsEmitter::emitSemanticStatement(
    llvm::raw_ostream &OS, const llvm::Init *Stmt,
    llvm::ArrayRef<llvm::SMLoc> Loc) {
  if (const auto *Dag = llvm::dyn_cast<llvm::DagInit>(Stmt)) {
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
      OS << "Translator.setRegOperandValue(MI, llvm::AMDGPU::OpName::"
         << OperandName << ", ";
      emitSemanticStatement(OS, AssignmentDag, Loc);
      OS << ")";
    }
    // --- ImplicitDef REG, value_dag ---
    // REG is either a `Register` record (a direct physical register like
    // VCC/EXEC/SGPR0) or a `SuperRegFactory<[r0, r1, ...]>` record that
    // names a synthesized tuple register (e.g. SGPR0_SGPR1_SGPR2_SGPR3).
    // Tuple registers aren't materialized as tablegen `Register` records —
    // they're synthesized at codegen time by RegisterTuples expansion — but
    // their concatenated enum names (`llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3`)
    // are valid C++ symbols, so we emit them by string-joining the subreg
    // names with underscores.
    else if (OpName == "ImplicitDef") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 2)
        llvm::PrintFatalError(
            Loc, "Expected `ImplicitDef` to have 2 arguments, got " +
                     llvm::Twine(NumArgs) + " instead");

      const auto *RegDef = llvm::dyn_cast<llvm::DefInit>(Dag->getArg(0));
      if (!RegDef)
        llvm::PrintFatalError(
            Loc, "First argument of `ImplicitDef` is not a record definition");

      OS << "Translator.setRegOperandValue(MI, ";
      emitRegisterReference(OS, RegDef, Loc);
      OS << ", ";

      const auto *ValDag = llvm::dyn_cast<llvm::DagInit>(Dag->getArg(1));
      if (!ValDag)
        llvm::PrintFatalError("ImplicitDef second arg is not a DAG");
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
      OS << "&Translator.getOperandAsValue(MI, llvm::AMDGPU::OpName::" << ArgName;
      if (Dag->getNumArgs() > 1) {
        OS << ", ";
        emitSemanticStatement(OS, Dag->getArg(1), Loc);
      }
      OS << ")";
    }
    // --- GetNamedOperandAsBB $name ---
    else if (OpName == "GetNamedOperandAsBB") {
      llvm::StringRef ArgName = Dag->getArgNameStr(0);
      OS << "&Translator.getOperandAsBasicBlock(MI, llvm::AMDGPU::OpName::"
         << ArgName << ")";
    }

    // --- GetNamedOperandAsFunction $name ---
    // Retrieves the named operand as an llvm::Function* (set by
    // CodeDiscoveryPass after callgraph resolution for direct-call instructions).
    // Returns nullptr if the operand is not a Function-typed GlobalAddress.
    else if (OpName == "GetNamedOperandAsFunction") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 1)
        llvm::PrintFatalError(
            Loc, "Expected `GetNamedOperandAsFunction` to have 1 argument, got " +
                     llvm::Twine(NumArgs) + " instead");
      llvm::StringRef ArgName = Dag->getArgNameStr(0);
      OS << "Translator.getOperandAsFunction(MI, llvm::AMDGPU::OpName::"
         << ArgName << ")";
    }

    // --- GetSyncScope <value-dag> ---
    // Decode a cpol immediate's Value* into a SyncScope::ID via the
    // translator. The argument is a value-producing dag (typically
    // (GetNamedOperand $cpol)) — emits the dag to compute the Value*,
    // then wraps it in getSyncScope().
    else if (OpName == "GetSyncScope") {
      if (Dag->getNumArgs() != 1)
        llvm::PrintFatalError(
            Loc, "Expected `GetSyncScope` to have 1 argument");
      OS << "Translator.getSyncScope(";
      emitSemanticStatement(OS, Dag->getArg(0), Loc);
      OS << ")";
    }
    // --- GetOrdering <value-dag> ---
    // Decode a cpol immediate's Value* into an AtomicOrdering via the
    // translator. Currently always Monotonic.
    else if (OpName == "GetOrdering") {
      if (Dag->getNumArgs() != 1)
        llvm::PrintFatalError(
            Loc, "Expected `GetOrdering` to have 1 argument");
      OS << "Translator.getOrdering(";
      emitSemanticStatement(OS, Dag->getArg(0), Loc);
      OS << ")";
    }

    // --- DirectCall <inst-addr> <callee-expr> ---
    // Emits a direct call. The first argument is the instruction's PC-value
    // (used to compute the absolute target when the callee operand is still
    // a raw displacement); the second is the callee, which may be either an
    // llvm::Function* (post-callgraph fixup) or a ConstantInt displacement
    // (pre-fixup). Dispatches to TraceIRTranslator::emitDirectTailCall, which
    // picks the right lowering based on the callee's runtime type.
    else if (OpName == "DirectCall") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 2)
        llvm::PrintFatalError(
            Loc,
            "Expected `DirectCall` to have 2 arguments (instruction PC, "
            "callee Function* or displacement), got " +
                llvm::Twine(NumArgs) + " instead");
      OS << "Translator.emitDirectTailCall(MI, Builder, ";
      emitSemanticStatement(OS, Dag->getArg(0), Loc);
      OS << ", ";
      emitSemanticStatement(OS, Dag->getArg(1), Loc);
      OS << ")";
    }

    // --- GetVal $name ---
    else if (OpName == "GetVal") {
      OS << Dag->getArgNameStr(0);
    }

    // --- GetNextBB ---
    // Returns the fall-through BasicBlock (next block after current MI's block)
    else if (OpName == "GetNextBB") {
      OS << "Translator.getNextBB(MI)";
    }

    // --- GetRegisterFile $opname, <laneTy> ---
    // First arg is a named operand of `MI` (e.g. `$src0`); the register held
    // in that operand is the base, and the returned vector slice runs from
    // that base to the end of its register file.
    else if (OpName == "GetRegisterFile") {
      llvm::StringRef ArgName = Dag->getArgNameStr(0);
      OS << "Translator.getRegisterFile(MI, llvm::AMDGPU::OpName::" << ArgName
         << ", ";
      emitSemanticStatement(OS, Dag->getArg(1), Loc);
      OS << ")";
    }

    // --- SetRegisterFile $opname, <val> ---
    // Symmetric with GetRegisterFile; writes the vector slice starting at
    // the operand-named register.
    else if (OpName == "SetRegisterFile") {
      llvm::StringRef ArgName = Dag->getArgNameStr(0);
      OS << "Translator.setRegisterFile(MI, llvm::AMDGPU::OpName::" << ArgName
         << ", ";
      emitSemanticStatement(OS, Dag->getArg(1), Loc);
      OS << ")";
    }

    // --- emitIndirectTailCall <target> ---
    else if (OpName == "IndirectTailCall") {
      OS << "Translator.emitIndirectTailCall(MI, Builder, ";
      emitSemanticStatement(OS, Dag->getArg(0), Loc);
      OS << ")";
    }

    // --- ImplicitUse REG ---
    // REG is a `Register` record or a `SuperRegFactory<[...]>` (see the
    // comment on `ImplicitDef`).
    else if (OpName == "ImplicitUse") {
      const auto *RegDef = llvm::dyn_cast<llvm::DefInit>(Dag->getArg(0));
      if (!RegDef)
        llvm::PrintFatalError(
            Loc, "First argument of `ImplicitUse` is not a record definition");
      OS << "&Translator.getOperandAsValue(MI, ";
      emitRegisterReference(OS, RegDef, Loc);
      OS << ")";
    }

    // --- GetWavefrontSize: subtarget's wavefront size (32 or 64) ---
    else if (OpName == "GetWavefrontSize") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 0)
        llvm::PrintFatalError(
            Loc, "Expected `GetWavefrontSize` to have 0 arguments, got " +
                     llvm::Twine(NumArgs) + " instead");
      OS << "Translator.ST.getWavefrontSize()";
    }

    // --- SelectOperand <cond>, <then>, <else>: C++ ternary expression ---
    else if (OpName == "SelectOperand") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 3)
        llvm::PrintFatalError(
            Loc, "Expected `SelectOperand` to have 3 arguments (cond, then, "
                 "else), got " +
                     llvm::Twine(NumArgs) + " instead");
      OS << "(";
      emitSemanticStatement(OS, Dag->getArg(0), Loc);
      OS << " ? ";
      emitSemanticStatement(OS, Dag->getArg(1), Loc);
      OS << " : ";
      emitSemanticStatement(OS, Dag->getArg(2), Loc);
      OS << ")";
    }

    // --- hasFeature $<short-name>: check if subtarget has a feature ---
    // Lowers to MCSubtargetInfo::hasFeature on Translator.ST. The arg-name
    // placeholder is auto-prefixed with "Feature" to form the enum value
    // (e.g. $GFX9Insts -> llvm::AMDGPU::FeatureGFX9Insts).
    else if (OpName == "hasFeature") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 1)
        llvm::PrintFatalError(
            Loc, "Expected `hasFeature` to have 1 argument (feature name "
                 "placeholder), got " +
                     llvm::Twine(NumArgs) + " instead");
      llvm::StringRef ArgName = Dag->getArgNameStr(0);
      if (ArgName.empty())
        llvm::PrintFatalError(
            Loc, "`hasFeature` argument must be a named placeholder "
                 "(e.g. $GFX9Insts)");
      OS << "Translator.ST.hasFeature(llvm::AMDGPU::Feature" << ArgName << ")";
    }

    // --- hasModifierSet $<opname>: check if a modifier operand is set ---
    // Lowers to SIInstrInfo::hasModifiersSet, which returns true iff the
    // named operand exists on MI and has a non-zero immediate. The operand
    // is referenced by its dag-arg name (same convention as GetNamedOperand).
    else if (OpName == "hasModifierSet") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 1)
        llvm::PrintFatalError(
            Loc, "Expected `hasModifierSet` to have 1 argument (named operand "
                 "placeholder), got " +
                     llvm::Twine(NumArgs) + " instead");
      llvm::StringRef ArgName = Dag->getArgNameStr(0);
      if (ArgName.empty())
        llvm::PrintFatalError(
            Loc, "`hasModifierSet` argument must be a named placeholder "
                 "(e.g. $gds)");
      OS << "Translator.TII.hasModifiersSet(MI, llvm::AMDGPU::OpName::"
         << ArgName << ")";
    }

    // --- LLVMIntNTy <width-dag>: parameterized integer type ---
    else if (OpName == "LLVMIntNTy") {
      if (unsigned NumArgs = Dag->getNumArgs(); NumArgs != 1)
        llvm::PrintFatalError(
            Loc, "Expected `LLVMIntNTy` to have 1 argument, got " +
                     llvm::Twine(NumArgs) + " instead");
      OS << "Builder.getIntNTy(";
      emitSemanticStatement(OS, Dag->getArg(0), Loc);
      OS << ")";
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
    } else if (OpClass->isSubClassOf(
                   Stmt->getRecordKeeper().getClass("LLVMComplexType"))) {
      OS << Op->getValueAsString("Builder") << "(";
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
    /// Alignment
    else if (DefType->isSubClassOf(
                 Stmt->getRecordKeeper().getClass("AlignVal"))) {
      OS << DefNode->getDef()->getValueAsString("Expr");
    }
    /// Atomic ordering
    else if (DefType->isSubClassOf(
                 Stmt->getRecordKeeper().getClass("AtomicOrderingVal"))) {
      OS << DefNode->getDef()->getValueAsString("Ordering");
    }
    /// Sync scope
    else if (DefType->isSubClassOf(
                 Stmt->getRecordKeeper().getClass("SyncScopeVal"))) {
      OS << DefNode->getDef()->getValueAsString("Expr");
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
                   Stmt->getRecordKeeper().getClass("SILLVMType"))) {
      OS << DefNode->getDef()->getValueAsString("Builder");
    } else if (DefType->isSubClassOf(
                   Stmt->getRecordKeeper().getClass("Intrinsic"))) {
      llvm::StringRef IntrinsicName = DefNode->getDef()->getName();
      if (!IntrinsicName.consume_front("int_")) {
        llvm::PrintFatalError(Loc, "Intrinsic record's name " + IntrinsicName +
                                       "does not start with 'int_'");
      }
      OS << "llvm::Intrinsic::" << IntrinsicName;
    } else if (DefType->isSubClassOf(
                   Stmt->getRecordKeeper().getClass("SuperRegFactory"))) {
      std::vector<const llvm::Record *> SubRegs =
          DefNode->getDef()->getValueAsListOfDefs("Regs");
      OS << "llvm::AMDGPU::";
      llvm::interleave(
          SubRegs, [&](const llvm::Record *SubReg) { OS << SubReg->getName(); },
          [&] { OS << "_"; });

    } else {
      llvm::PrintFatalError(
          Loc, "Unhandled def node: " + DefNode->getAsString() + ".");
    }
  } else if (const auto *ListNode = llvm::dyn_cast<llvm::ListInit>(Stmt)) {
    OS << "{";
    llvm::interleave(
        ListNode->getElements(),
        [&](const llvm::Init *Elem) { emitSemanticStatement(OS, Elem, Loc); },
        [&] { OS << ", "; });
    OS << "}";
  } else if (auto *IntNode = llvm::dyn_cast<llvm::IntInit>(Stmt)) {
    OS << IntNode->getValue();
  } else if (auto *StrNode = llvm::dyn_cast<llvm::StringInit>(Stmt)) {
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
  OS << "    TraceIRTranslator &Translator) {\n";

  for (const llvm::Init *El : SemList->getElements()) {
    const auto *StmtDag = llvm::dyn_cast<llvm::DagInit>(El);
    if (!StmtDag)
      PrintFatalError(Rec->getLoc(), "Semantic element is not a DAG");
    emitSemanticStatement(OS.indent(2), StmtDag, SemanticLoc);
    OS << ";\n";
  }

  OS << "}\n\n";
}

//===----------------------------------------------------------------------===//
// Dispatch function emitter
//===----------------------------------------------------------------------===//

void SIInstrSemanticsEmitter::emitMacro(
    llvm::raw_ostream &OS, llvm::ArrayRef<const llvm::Record *> Semantics) {
  OS << "#ifndef HANDLE_INST_SEMANTIC\n";
  OS << "#define HANDLE_INST_SEMANTIC(OPCODE) \n";
  OS << "#endif\n";
  for (const llvm::Record *Rec : Semantics) {
    const auto *SemList = Rec->getValueAsListInit("Semantic");
    if (!SemList || SemList->empty())
      continue;
    const llvm::Record *InstRec = Rec->getValueAsDef("Instruction");
    llvm::StringRef InstName = InstRec->getName();
    OS << "HANDLE_INST_SEMANTIC(" << InstName << ")\n";
  }
  OS << "#undef HANDLE_INST_SEMANTIC\n";
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

  emitHeader(OS);

  OS << "#ifdef GET_SI_INSTR_SEMANTIC_FUNCTIONS\n";
  OS << "#undef GET_SI_INSTR_SEMANTIC_FUNCTIONS\n";

  for (const llvm::Record *Rec : Semantics) {
    emitSemanticFunction(OS, Rec);
  }

  OS << "#endif // GET_SI_INSTR_SEMANTIC_FUNCTIONS\n\n";

  emitMacro(OS, Semantics);

  // --- Emit array of all opcodes that have semantic definitions ---
  OS << "#ifdef GET_SI_INSTR_SEMANTIC_OPCODE_LIST\n";
  OS << "static constexpr uint16_t SemanticallyModeledOpcodes[] = {\n";
  for (const llvm::Record *Rec : Semantics) {
    const auto *SemList = Rec->getValueAsListInit("Semantic");
    if (!SemList || SemList->empty())
      continue;
    const llvm::Record *InstRec = Rec->getValueAsDef("Instruction");
    OS << "  llvm::AMDGPU::" << InstRec->getName() << ",\n";
  }
  OS << "};\n";
  OS << "static constexpr size_t NumSemanticallyModeledOpcodes = "
        "sizeof(SemanticallyModeledOpcodes) / "
        "sizeof(SemanticallyModeledOpcodes[0]);\n";
  OS << "#undef GET_SI_INSTR_SEMANTIC_OPCODE_LIST\n";
  OS << "#endif // GET_SI_INSTR_SEMANTIC_OPCODE_LIST\n\n";
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
