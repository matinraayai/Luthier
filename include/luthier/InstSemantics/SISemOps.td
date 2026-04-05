//===- InstSISemantics.td -------------------------------------*- tblgen -*-==//
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
// Describes a set of LLVM IR operations, types, and utilities used to
// describe the semantics of the AMD GPU backend Pseudo-physical instructions.
//===----------------------------------------------------------------------===//
include "AMDGPU.td"


/// Indicates the LLVMContext of the module should be passed as an argument
/// to the IR value builder function
def LLVMContext;

//------------------------------------------------------------------------------
// LLVM Constant Value Definitions
//------------------------------------------------------------------------------

class LLVMConstant<string builderFunc> {
  string BuilderFunc = builderFunc;
}

def ConstantTrue : LLVMConstant<"llvm::ConstantInt::getTrue">;

def ConstantFalse : LLVMConstant<"llvm::ConstantInt::getFalse">;

def ConstantBool : LLVMConstant<"llvm::ConstantInt::getBool">;

def ConstantInt : LLVMConstant<"llvm::ConstantInt::get">;

def ConstantSignedInt : LLVMConstant<"llvm::ConstantInt::getSigned">;

def ConstantFP : LLVMConstant<"llvm::ConstantFP::get">;

def NaN : LLVMConstant<"llvm::ConstantFP::getNaN">;

def ZeroFP : LLVMConstant<"llvm::ConstantFP::getZero">;

def PoisionVal : LLVMConstant<"llvm::PoisonValue::get">;

//------------------------------------------------------------------------------
// LLVM IR Operation Definitions
//------------------------------------------------------------------------------

/// Maps to the llvm::FastMathFlags class constructor
//def FastMathFlags;
//
//def FastMathFlagEnums {
//  int AllowReassoc    = !shl(1, 0);
//  int NoNaNs          = !shl(1 , 1);
//  int NoInfs          = !shl(1, 2);
//  int NoSignedZeros   = !shl(1, 3);
//  int AllowReciprocal = !shl(1, 4);
//  int AllowContract   = !shl(1, 5);
//  int ApproxFunc      = !shl(1, 6);
//  int FlagEnd         = !shl(1, 7);
//}

/// Represents LLVM operations used to describe the SI instruction semantics
/// records
class LLVMOp<string iRBuilderFunc> {
  /// The llvm::IRBuilder function used to construct the operation
  string IRBuilderFunc = iRBuilderFunc;
}

def LLVMAdd : LLVMOp<"CreateAdd">;

def LLVMSub : LLVMOp<"CreateSub">;

def LLVMMul : LLVMOp<"CreateMul">;

def LLVMUDiv : LLVMOp<"CreateUDiv">;

def LLVMSDiv : LLVMOp<"CreateSDiv">;

def LLVMURem : LLVMOp<"CreateURem">;

def LLVMSRem : LLVMOp<"CreateSRem">;

def LLVMXor : LLVMOp<"CreateXor">;

def LLVMOr : LLVMOp<"CreateOr">;

def LLVMAnd : LLVMOp<"CreateAnd">;

def LLVMLShr : LLVMOp<"CreateLShr">;

def LLVMShl : LLVMOp<"CreateShl">;

def LLVMAshl : LLVMOp<"CreateAshl">;

def LLVMZExt : LLVMOp<"CreateZExt">;

def LLVMSExt : LLVMOp<"CreateSExt">;

def LLVMSelect : LLVMOp<"CreateSelect">;

def LLVMCmp : LLVMOp<"CreateCmp">;

def LLVMTrunc : LLVMOp<"CreateTrunc">;

def LLVMIntToPtr : LLVMOp<"CreateIntToPtr">;

def LLVMPtrAdd : LLVMOp<"CreatePtrAdd">;

def LLVMLoad : LLVMOp<"CreateLoad">;

def LLVMStore : LLVMOp<"CreateStore">;

def LLVMBitCast : LLVMOp<"CreateBitCast">;

def LLVMInsertElement : LLVMOp<"CreateInsertElement">;

def LLVMExtractElement : LLVMOp<"CreateExtractElement">;

def LLVMFAdd : LLVMOp<"CreateFAdd">;

def LLVMFSub : LLVMOp<"CreateFSub">;

def LLVMFMul : LLVMOp<"CreateFMul">;

def LLVMFDiv : LLVMOp<"CreateFDiv">;

def LLVMFRem : LLVMOp<"CreateFRem">;

def LLVMFMA : LLVMOp<"CreateFMA">;

def LLVMFNeg : LLVMOp<"CreateFNeg">;

def LLVMFPTrunc : LLVMOp<"CreateFPTrunc">;

def LLVMFPExt : LLVMOp<"CreateFPExt">;

def LLVMFCmp : LLVMOp<"CreateFCmp">;

def LLVMSIToFP : LLVMOp<"CreateSIToFP">;

def LLVMUIToFP : LLVMOp<"CreateUIToFP">;

def LLVMFPToSI : LLVMOp<"CreateFPToSI">;

def LLVMFPToUI : LLVMOp<"CreateFPToUI">;

def LLVMAShr : LLVMOp<"CreateAShr">;

def LLVMMaxNum : LLVMOp<"CreateMaxNum">;

def LLVMMinNum : LLVMOp<"CreateMinNum">;

def LLVMUnaryIntrinsic : LLVMOp<"CreateUnaryIntrinsic">;

def LLVMBinaryIntrinsic : LLVMOp<"CreateBinaryIntrinsic">;

def LLVMIntrinsic : LLVMOp<"CreateIntrinsic">;

def LLVMLdexp : LLVMOp<"CreateLdexp">;

def LLVMAtomicRMW : LLVMOp<"CreateAtomicRMW">;

//------------------------------------------------------------------------------
// AtomicRMW Operation Enum Values
//------------------------------------------------------------------------------

class AtomicRMWOp<string op> {
  string Op = op;
}

def AtomicRMWAdd : AtomicRMWOp<"llvm::AtomicRMWInst::Add">;

def AtomicRMWSub : AtomicRMWOp<"llvm::AtomicRMWInst::Sub">;

def AtomicRMWAnd : AtomicRMWOp<"llvm::AtomicRMWInst::And">;

def AtomicRMWOr : AtomicRMWOp<"llvm::AtomicRMWInst::Or">;

def AtomicRMWXor : AtomicRMWOp<"llvm::AtomicRMWInst::Xor">;

def AtomicRMWXchg : AtomicRMWOp<"llvm::AtomicRMWInst::Xchg">;

def AtomicRMWMin : AtomicRMWOp<"llvm::AtomicRMWInst::Min">;

def AtomicRMWUMin : AtomicRMWOp<"llvm::AtomicRMWInst::UMin">;

def AtomicRMWMax : AtomicRMWOp<"llvm::AtomicRMWInst::Max">;

def AtomicRMWUMax : AtomicRMWOp<"llvm::AtomicRMWInst::UMax">;

def AtomicRMWNand : AtomicRMWOp<"llvm::AtomicRMWInst::Nand">;

def AtomicRMWFAdd : AtomicRMWOp<"llvm::AtomicRMWInst::FAdd">;

def AtomicRMWFMax : AtomicRMWOp<"llvm::AtomicRMWInst::FMax">;

def AtomicRMWFMin : AtomicRMWOp<"llvm::AtomicRMWInst::FMin">;

def AtomicRMWUIncWrap : AtomicRMWOp<"llvm::AtomicRMWInst::UIncWrap">;

def AtomicRMWUDecWrap : AtomicRMWOp<"llvm::AtomicRMWInst::UDecWrap">;

def AtomicRMWUSubCond : AtomicRMWOp<"llvm::AtomicRMWInst::USubCond">;

def AtomicRMWUSubSat : AtomicRMWOp<"llvm::AtomicRMWInst::USubSat">;

//------------------------------------------------------------------------------
// Alignment Values
//------------------------------------------------------------------------------

class AlignVal<string expr> {
  string Expr = expr;
}

/// Pass std::nullopt to let CreateAtomicRMW infer alignment from the
/// value type via DataLayout::getTypeStoreSize.
def AlignNone : AlignVal<"std::nullopt">;

//------------------------------------------------------------------------------
// Atomic Ordering Enum Values
//------------------------------------------------------------------------------

class AtomicOrderingVal<string ord> {
  string Ordering = ord;
}

def AtomicOrderingUnordered : AtomicOrderingVal<"llvm::AtomicOrdering::Unordered">;
def AtomicOrderingMonotonic : AtomicOrderingVal<"llvm::AtomicOrdering::Monotonic">;
def AtomicOrderingAcquire : AtomicOrderingVal<"llvm::AtomicOrdering::Acquire">;
def AtomicOrderingRelease : AtomicOrderingVal<"llvm::AtomicOrdering::Release">;
def AtomicOrderingAcqRel : AtomicOrderingVal<"llvm::AtomicOrdering::AcquireRelease">;
def AtomicOrderingSeqCst : AtomicOrderingVal<"llvm::AtomicOrdering::SequentiallyConsistent">;

//------------------------------------------------------------------------------
// Sync Scope Values
//------------------------------------------------------------------------------
// SyncScope::ID is a runtime-allocated uint8_t in LLVM. The two built-in
// scopes (SingleThread=0, System=1) are constants; all others are obtained
// via LLVMContext::getOrInsertSyncScopeID at code-generation time.

class SyncScopeVal<string expr> {
  string Expr = expr;
}

// Built-in LLVM sync scopes
def SyncScopeSingleThread : SyncScopeVal<"llvm::SyncScope::SingleThread">;
def SyncScopeSystem       : SyncScopeVal<"llvm::SyncScope::System">;

// AMDGPU-specific sync scopes (looked up by string name at runtime)
def SyncScopeAgent     : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"agent\")">;
def SyncScopeWorkgroup : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"workgroup\")">;
def SyncScopeWavefront : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"wavefront\")">;
def SyncScopeCluster   : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"cluster\")">;

// AMDGPU one-address-space sync scopes
def SyncScopeOneAS              : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"one-as\")">;
def SyncScopeAgentOneAS         : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"agent-one-as\")">;
def SyncScopeWorkgroupOneAS     : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"workgroup-one-as\")">;
def SyncScopeWavefrontOneAS     : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"wavefront-one-as\")">;
def SyncScopeSingleThreadOneAS  : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"singlethread-one-as\")">;
def SyncScopeClusterOneAS       : SyncScopeVal<
  "Builder.getContext().getOrInsertSyncScopeID(\"cluster-one-as\")">;

//------------------------------------------------------------------------------
// Comparison Predicates
//------------------------------------------------------------------------------

class LLVMCmpPredicate<string cc> {
  string Code = cc;
}

def LLVMFCmpFALSE : LLVMCmpPredicate<"llvm::CmpInst::FCMP_FALSE">;

def LLVMFCmpOEQ : LLVMCmpPredicate<"llvm::CmpInst::FCMP_OEQ">;

def LLVMFCmpOGT : LLVMCmpPredicate<"llvm::CmpInst::FCMP_OGT">;

def LLVMFCmpOGE : LLVMCmpPredicate<"llvm::CmpInst::FCMP_OGE">;

def LLVMFCmpOLT : LLVMCmpPredicate<"llvm::CmpInst::FCMP_OLT">;

def LLVMFCmpOLE : LLVMCmpPredicate<"llvm::CmpInst::FCMP_OLE">;

def LLVMFCmpONE : LLVMCmpPredicate<"llvm::CmpInst::FCMP_ONE">;

def LLVMFCmpORD : LLVMCmpPredicate<"llvm::CmpInst::FCMP_ORD">;

def LLVMFCmpUNO : LLVMCmpPredicate<"llvm::CmpInst::FCMP_UNO">;

def LLVMFCmpUEQ : LLVMCmpPredicate<"llvm::CmpInst::FCMP_UEQ">;

def LLVMFCmpUGT : LLVMCmpPredicate<"llvm::CmpInst::FCMP_UGT">;

def LLVMFCmpUGE : LLVMCmpPredicate<"llvm::CmpInst::FCMP_UGE">;

def LLVMFCmpULT : LLVMCmpPredicate<"llvm::CmpInst::FCMP_ULT">;

def LLVMFCmpULE : LLVMCmpPredicate<"llvm::CmpInst::FCMP_ULE">;

def LLVMFCmpUNE : LLVMCmpPredicate<"llvm::CmpInst::FCMP_UNE">;

def LLVMFCmpTrue : LLVMCmpPredicate<"llvm::CmpInst::FCMP_TRUE">;

def LLVMICmpEQ : LLVMCmpPredicate<"llvm::CmpInst::ICMP_EQ">;

def LLVMICmpNE : LLVMCmpPredicate<"llvm::CmpInst::ICMP_NE">;

def LLVMICmpUGT : LLVMCmpPredicate<"llvm::CmpInst::ICMP_UGT">;

def LLVMICmpUGE : LLVMCmpPredicate<"llvm::CmpInst::ICMP_UGE">;

def LLVMICmpULT : LLVMCmpPredicate<"llvm::CmpInst::ICMP_ULT">;

def LLVMICmpULE : LLVMCmpPredicate<"llvm::CmpInst::ICMP_ULE">;

def LLVMICmpSGT : LLVMCmpPredicate<"llvm::CmpInst::ICMP_SGT">;

def LLVMICmpSGE : LLVMCmpPredicate<"llvm::CmpInst::ICMP_SGE">;

def LLVMICmpSLT : LLVMCmpPredicate<"llvm::CmpInst::ICMP_SLT">;

def LLVMICmpSLE : LLVMCmpPredicate<"llvm::CmpInst::ICMP_SLE">;

//------------------------------------------------------------------------------
// InstSI-specific semantic operations
//------------------------------------------------------------------------------

/// Used to get the named operand of the machine instruction (instead of hard
/// coding the operand number)
def GetNamedOperand;

/// Used to set the named operand of the machine instruction (instead of hard
/// coding the operand number)
def SetNamedOperand;

/// Used to set a register directly in case of an implicit def
def ImplicitDef;

/// Used to get a register directly in case of an implicit use
def ImplicitUse;

/// Used to select an operand based on a boolean predicate (different from
/// the LLVMSelect operation). Does not translate into an IR operation
def SelectOperand;

/// Used to indicate definition of an intermediate value to be used by
/// Later operations inside the semantic
/// The second operand should be a name (e.g., (DefVal $tmp (...)))
def DefVal;

def GetVal;

/// Used to check if the MI has a modifier set (checked via
/// SIInstrInfo::hasModifierSet). Does not translate into an IR
/// operation
def hasModifierSet;

/// Defined here because AMDGPU.td records don't define it
def region_ptr_ty : LLVMQualPointerType<2>;

/// Scratch address space
def scratch_ptr_ty : LLVMQualPointerType<5>;

/// Buffer address space (resource pointer)
def buffer_ptr_ty : LLVMQualPointerType<8>;

/// Buffer fat pointer address space
def buffer_fat_ptr_ty : LLVMQualPointerType<9>;

/// 3 x f16 not defined in LLVM
def llvm_v3f16_ty      : LLVMType<v3f16>;

//------------------------------------------------------------------------------
// Complex LLVM IR Operation Definitions (i.e. low-level binary operations that
// require multiple LLVM IR operations)
//------------------------------------------------------------------------------

/// Performs a bit-wise inversion operation on any integer value
class LLVMNot<dag val, LLVMType type> {
  dag ret = (LLVMXor val, (LLVMSExt (ConstantTrue llvm_i1_ty), type));
}

/// Breaks down an input value of type inputType into a set of
/// sub values
//class LLVMBreakdownValue<dag input, LLVMType inputType, int numEls> {
//  assert !eq(inputType.VT.nElem, 1), "inputType must be a scalar value";
//  assert !or(inputType.VT.isFP, inputType.VT.isInteger),
//         "inputType must either be a floating point or an integer";
//  dag BitcastedVal = (LLVMBitCast:$bitcast input,
//                       !cast<LLVMType>(
//                        llvm_#!if(!head(types).VT.isFP, "f", "i")
//                           #numEls
//                           #_ty
//                        )
//                      );
//  list<dag> BrokenDownVals = !foreach(i, !range(numEls),
//    (LLVMExtractElement $bitcast, i));
//}

/// Merges a set of dag operands together into a single value
/// Input values must be the same type (scalar integer or
/// floating point)
/// For merging two types that are not the same, they must be
/// broken down into a common type first manually
//class LLVMMergeValue<list<dag> ins, LLVMType type> {
//  assert !not(!empty(ins)), "argument list shouldn't be empty";
//  assert !eq(type.VT.nElem, 1), "type must be a scalar";
//  assert !or(type.VT.isFP, type.VT.isInteger),
//         "type must be a floating point or scalar";
//
//
//  /// The output type is a floating point or an integer, with its
//  /// size being the aggregate of all input types
//  LLVMType OutputType =
//    !cast<LLVMType>(llvm_#!if(type.VT.isFP, "f", "i")#
//                          !mul(!size(ins), type.VT.Size)#_ty);
//  /// Type of vector the values will be inserted to before being
//  /// casted to the final scalar
//  LLVMType MergeVectorType =
//    !cast<LLVMType>(llvm_v#!size(ins)
//                          #_#
//                          !if(!head(types).VT.isFP, "f", "i")#
//                          type.VT.Size
//                          #_ty
//                    );
//
//  list<dag> Op = !listconcat(
//    [(LLVMInsertElement:$mergevalTmp
//      GreatestCommonValType<types>.ret, !head(ins)), 0],
//    !foreach(idx, !range(1, !size(types)),
//      (LLVMInsertElement:$mergevalTmp $mergevalTmp, ins[idx], idx)),
//    [(LLVMBitCast $mergevalTmp, OutputType)]
//  );
//}

class LLVMExtractValueConstIdx<dag val, LLVMType subValType, int idx> {
  assert !gt(subValType.VT.nElem, 1), "subValType must be a vector type";
  assert !gt(subValType.VT.nElem, idx),
         "Index cannot be greater than number of elements";
  assert !gt(idx, 0), "Index must be greater than zero";
  dag Op =
    (LLVMExtractElement (LLVMBitCast val, subValType), idx);
}

class LLVMExtractValueVarIdx<dag val, LLVMType subValType, dag idx> {
  assert !gt(subValType.VT.nElem, 1), "subValType must be a vector type";
  dag Op =
    (LLVMExtractElement (LLVMBitCast val, subValType), idx);
}

class LLVMExtractSignBit<dag val, LLVMType type> {
  dag Op = LLVMExtractValueConstIdx<val, type, !sub(type.VT.Size, 1)>.Op;
}

class LLVMSetSignBit<dag val, LLVMType type, dag sgn> {
  dag Op = (LLVMAnd val, (LLVMShl sgn, !sub(type.VT.Size, 1)));
}

class LLVMExtractBitsIntStartIntNumBits<dag val,
                                        LLVMType inputType,
                                        bit isSigned, int startIdx,
                                        int numBits = 1> {
  assert !lt(startIdx, inputType.VT.Size), "Start index must be less than the size of the element type";
  assert !eq(inputType.VT.nElem, 1), "type must be a scalar";
  assert !lt(!add(startIdx, numBits), inputType.VT.Size), "Total number of bits requested is out of range";
  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);

  dag Op = !if(!eq(numBits, 1),
             (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx)),
                      (ConstantInt inputType, !sub(!shl(1, numBits), 1))),
             (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx))));
}

class LLVMExtractBitsDagStartDagNumBits<dag val, LLVMType inputType, bit isSigned, dag startIdx, dag numBits> {
  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);
  dag Op = (LLVMAnd (rightShiftOp val, startIdx),
                    (LLVMSub (LLVMShl (ConstantInt inputType, 1), numBits), (ConstantInt inputType, 1)));
}

class LLVMExtractBitsIntStartDagNumBits<dag val, LLVMType inputType, bit isSigned, int startIdx, dag numBits> {
  assert !eq(inputType.VT.nElem, 1), "type must be a scalar";
  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);
  dag Op = (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx)),
                    (LLVMSub (LLVMShl (ConstantInt inputType, 1), numBits),
                                      (ConstantInt inputType, 1)));
}

//class LLVMExtractBitsDagStartIntNumBits<dag val, LLVMType inputType, dag startIdx, int numBits = 1> {
//  assert !eq(inputType.VT.nElem, 1), "type must be a scalar";
//  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);
//  dag Op = !if(!eq(numBits, 1),
//             (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx)),
//                                         (ConstantInt inputType, !sub(!shl(1, numBits), 1))),
//             (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx))));
//}
