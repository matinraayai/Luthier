//===-- SIInstrSemanticsEmitter.hpp - SI Instruction Semantics Emitter
//-----===//
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
///
/// \file SIInstrSemanticsEmitter.hpp
/// TableGen backend that parses InstSISemantic and InstSIIntrinsic records
/// and emits C++ template specializations of raiseMachineInstr<Opcode>()
/// that translate each AMDGPU MachineInstr into equivalent LLVM IR using
/// IRBuilder.
///
/// For InstSISemantic records, the Semantic DAG list is walked recursively
/// to emit IRBuilder calls (CreateAdd, CreateFAdd, CreateBitCast, etc.).
///
/// For InstSIIntrinsic records, a CreateIntrinsic call is emitted mapping
/// the instruction to its corresponding LLVM intrinsic.
///
/// A dispatch function is also emitted with a switch on MI opcode.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TBLGEN_SI_INSTR_SEMANTICS_EMITTER_HPP
#define LUTHIER_TBLGEN_SI_INSTR_SEMANTICS_EMITTER_HPP
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/SMLoc.h>
#include <string>

namespace llvm {
class RecordKeeper;
class Record;
class Init;
class DagInit;
class raw_ostream;
} // namespace llvm

namespace luthier {

/// \brief Emits raiseMachineInstr<> template specializations from
/// InstSISemantic and InstSIIntrinsic tablegen records.
class SIInstrSemanticsEmitter {
public:
  explicit SIInstrSemanticsEmitter(const llvm::RecordKeeper &Records)
      : Records(Records) {}

  /// Emit all template specializations and the dispatch switch function.
  void run(llvm::raw_ostream &OS);

private:
  const llvm::RecordKeeper &Records;

  /// Counter for generating unique temporary variable names.
  unsigned TmpCounter = 0;

  /// Map from DefVal names to their C++ variable names.
  llvm::StringMap<std::string> DefValMap;

  /// Generate a fresh temporary variable name.
  std::string freshTmp();

  /// Emit the file header (includes, namespace, forward decls).
  void emitHeader(llvm::raw_ostream &OS);

  /// Emit the file footer (close namespace).
  void emitFooter(llvm::raw_ostream &OS);

  /// Emit a single raiseMachineInstr<> specialization for an InstSISemantic.
  void emitSemanticFunction(llvm::raw_ostream &OS, const llvm::Record *Rec);

  /// Emit a single raiseMachineInstr<> specialization for an InstSIIntrinsic.
  void emitIntrinsicFunction(llvm::raw_ostream &OS, const llvm::Record *Rec);

  /// Emit the dispatch switch function that routes MI opcodes to the
  /// appropriate specialization.
  void emitDispatchFunction(llvm::raw_ostream &OS,
                            llvm::ArrayRef<const llvm::Record *> Semantics,
                            llvm::ArrayRef<const llvm::Record *> Intrinsics);

  /// Recursively emit code for a DAG expression, returning the C++ variable
  /// name holding the result.
  /// \param OS Output stream
  /// \param Dag The DAG init to emit code for
  /// \param Indent Indentation string
  /// \returns The C++ variable name holding the result value
  std::string emitDagExpr(llvm::raw_ostream &OS, const llvm::DagInit *Dag,
                          llvm::StringRef Indent);

  /// Emit code for a top-level semantic statement (SetNamedOperand,
  /// ImplicitDef, DefVal, or a bare store/call).
  void emitSemanticStatement(llvm::raw_ostream &OS, const llvm::Init *Stmt,
                             llvm::ArrayRef<llvm::SMLoc> Loc,
                             unsigned int Indent);

  /// Map an LLVMType tablegen name to its C++ IRBuilder type getter.
  static std::string getTypeExpr(llvm::StringRef TypeName);

  /// Map a CmpInst predicate name to its C++ enum value.
  static std::string getCmpPredicate(llvm::StringRef PredName);

  /// Map a ConstantXxx name to its C++ builder expression.
  static std::string getConstantExpr(llvm::StringRef ConstName);

  /// Map an AtomicRMWOp name to its C++ enum value.
  static std::string getAtomicRMWOp(llvm::StringRef OpName);
};

/// Entry point for the tablegen backend.
void emitSIInstrSemantics(const llvm::RecordKeeper &Records,
                          llvm::raw_ostream &OS);

} // namespace luthier

#endif
