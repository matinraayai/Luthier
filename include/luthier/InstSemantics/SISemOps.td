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

/// Extract a member from an aggregate (struct or array) value. Usage:
///   \c (LLVMExtractValue <aggregate>, [<idx0>, <idx1>, ...]).
/// The index list is forwarded as an \c ArrayRef<unsigned> to
/// \c IRBuilder::CreateExtractValue. Used for splitting struct-returning
/// intrinsic results (e.g. \c llvm.amdgcn.ds.bvh.stack.*).
def LLVMExtractValue : LLVMOp<"CreateExtractValue">;

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

/// Emit `IRBuilder::CreateAtomicCmpXchg(Ptr, Cmp, New, Align,
///   SuccessOrdering, FailureOrdering, SyncScope::ID)`.
/// Returns a {value, i1} struct — use `(LLVMExtractValue $rmw, 0)` to
/// read the loaded pre-swap value and `(LLVMExtractValue $rmw, 1)` to
/// read the success flag.
def LLVMAtomicCmpXchg : LLVMOp<"CreateAtomicCmpXchg">;

/// Emit an `IRBuilder::CreateFence(Ordering, SyncScope::ID)` call.
def LLVMFence : LLVMOp<"CreateFence">;

/// Decode a cpol immediate (Value*) into a SyncScope::ID per subtarget.
/// Use: (LLVMAtomicRMW ..., (GetSyncScope (GetNamedOperand $cpol)))
def GetSyncScope;

/// Decode a cpol immediate (Value*) into an AtomicOrdering per subtarget.
/// Currently always Monotonic; the cpol value is reserved for future use.
def GetOrdering;

def LLVMBranch : LLVMOp<"CreateBr">;

def LLVMCondBranch : LLVMOp<"CreateCondBr">;

def LLVMIndirectBranch : LLVMOp<"CreateIndirectBr">;

def LLVMCall : LLVMOp<"CreateCall">;

def LLVMUnreachable : LLVMOp<"CreateUnreachable">;

def LLVMZExtOrTrunc : LLVMOp<"CreateZExtOrTrunc">;

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

///
def GetNamedOperandAsBB;

/// Used to set the named operand of the machine instruction (instead of hard
/// coding the operand number)
def SetNamedOperand;

/// Used to set a register directly in case of an implicit def
def ImplicitDef;

/// Used to get a register directly in case of an implicit use
def ImplicitUse;

/// Selects between two operands based on a boolean predicate, distinct from
/// the LLVMSelect IR operation. Lowers to a C++ ternary expression at codegen
/// time, so it must appear in a value position (not as a top-level statement).
/// Usage: \c (SelectOperand <cond-dag>, <then-dag>, <else-dag>).
def SelectOperand;

/// Returns true at codegen time if the current subtarget has the named
/// feature
def hasFeature;

/// Read the register file of a register as a vector.
/// The first argument is the register. The second argument is the lane
/// scalar type (e.g. \c si_llvm_i32_ty for
/// \c <NumRegs x i32>, \c si_llvm_i64_ty of \c <NumRegs / 2 x i64).
def GetRegisterFile;

/// Overwrite the entire register file associated with the register with
/// a vector value. The first argument is the register indicating the
/// destination register file. The second argument is the value to
/// overwrite the entire file with
/// The value to write must match the lane formation of the target kernel,
/// which is why it generally should origiate from a \c GetRegisterFile.
def SetRegisterFile;

/// Indirect transfer of control to \c <target> (PC)'s function without
/// ever returning. Materializes the full register-file state, packages
/// them as arguments to a call against a device function.
def IndirectTailCall;

/// Obtains the fallthrough basic block of the current machine instruction
/// Useful when translating branch instructions
def GetNextBB;

/// Used to indicate definition of an intermediate value to be used by
/// Later operations inside the semantic
/// The second operand should be a name (e.g., (DefVal $tmp (...)))
def DefVal;

def GetVal;

/// Used to check if the MI has a modifier set (checked via
/// SIInstrInfo::hasModifierSet). Does not translate into an IR
/// operation
def hasModifierSet;

/// Resolves at IR-build time to the current MachineFunction's wavefront size
/// (32 or 64). Emits a C++ expression of type \c unsigned. Intended for use
/// as the size argument to \c LLVMIntNTy or as an immediate operand to ops
/// that need wave-size-dependent values.
/// Retrieves a named operand as an \c llvm::Function*. The operand must be
/// of kind \c MO_GlobalAddress with a Function-typed global value — typically
/// set by \c CodeDiscoveryPass after callgraph resolution for direct-call
/// instructions like \c S_CALL_B64.
def GetNamedOperandAsFunction;

/// Emits a direct call to the given \c llvm::Function*. Reuses the
/// translator's tail-call plumbing (constructs the device-function-typed
/// argument tuple and emits the call). Argument should resolve to an
/// \c llvm::Function* (typically via \c GetNamedOperandAsFunction).
def DirectCall;

def GetWavefrontSize;

/// Parameterized integer LLVM type. Takes a single dag argument that resolves
/// to a C++ \c unsigned expression giving the bit width. Emits
/// \c Builder.getIntNTy(<width>). Typical use: \c (LLVMIntNTy (GetWavefrontSize))
/// produces \c Builder.getIntNTy(Translator.ST.getWavefrontSize()) — the
/// wave-mask integer type (i32 on wave32, i64 on wave64).
def LLVMIntNTy;

/// Wraps an i1 dag in \c int_amdgcn_ballot to produce a wave-mask integer
/// value matching the current subtarget's wavefront size. Use when writing
/// per-lane i1 results to wave-mask registers (VCC, EXEC, sdst-pair).
class BallotWaveMask<dag predicate> {
  dag ret = (LLVMIntrinsic (LLVMIntNTy (GetWavefrontSize)),
                             int_amdgcn_ballot, [predicate]);
}

/// Defined here because AMDGPU.td records don't define it
def region_ptr_ty : LLVMQualPointerType<2>;

// Note: flat_ptr_ty (addrspace 0) and global_ptr_ty (addrspace 1) are already
// defined by LLVM's IntrinsicsAMDGPU.td and are referenced directly.

/// Scratch address space
def scratch_ptr_ty : LLVMQualPointerType<5>;

/// Buffer address space (resource pointer)
def buffer_ptr_ty : LLVMQualPointerType<8>;

/// Buffer fat pointer address space
def buffer_fat_ptr_ty : LLVMQualPointerType<9>;

//------------------------------------------------------------------------------
// LLVM Types
//------------------------------------------------------------------------------

/// We make a separate class for representing type arguments in the semantics
/// to give us more freedom in representing more types, especially those
/// used in LLVM but not in ISEL
class SILLVMType<string builder, int nElem, int elemSize, bit isFP> {
  string Builder = builder;
  int NElem = nElem;
  int ElemSize = elemSize;
  bit IsFP = isFP;
}

foreach size = [1, 2, 4, 8, 16, 32, 64, 128] in {
  def si_llvm_i#size#_ty : SILLVMType<"Builder.getInt"#size#"Ty()", 1, size, false>;
}

def si_llvm_half_ty : SILLVMType<"Builder.getHalfTy()", 1, 16, true>;
def si_llvm_bfloat_ty : SILLVMType<"Builder.getBFloatTy()", 1, 16, true>;
def si_llvm_float_ty : SILLVMType<"Builder.getFloatTy()", 1, 32, true>;
def si_llvm_double_ty : SILLVMType<"Builder.getDoubleTy()", 1, 64, true>;

def si_llvm_void_ty : SILLVMType<"Builder.getVoidTy()", 0, 0, false>;

foreach size = [1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] in {
  foreach int_size = [1, 2, 4, 8, 16, 32, 64, 128] in {
    def si_llvm_v#size#i#int_size#_ty :
      SILLVMType<"llvm::FixedVectorType::get("#!cast<SILLVMType>("si_llvm_i"#int_size#"_ty").Builder#", "#size#")", size, int_size, false>;
  }
}

foreach size = [1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] in {
  foreach idx = [0, 1, 2, 3] in {
    defvar float_ty = [si_llvm_half_ty, si_llvm_bfloat_ty, si_llvm_float_ty, si_llvm_double_ty][idx];
    defvar float_name = ["f16", "bf16", "f32", "f64"][idx];
      def si_llvm_v#size#float_name#_ty :
        SILLVMType<"llvm::FixedVectorType::get("#float_ty.Builder#", "#size#")", size, float_ty.ElemSize, true>;
  }
}

class LLVMComplexType<string builder> {
  string Builder = builder;
}

def LLVMFuncTy : LLVMComplexType<"llvm::FunctionType::get">;

def LLVMStructTy : LLVMComplexType<"llvm::StructType::get">;

/// Used to create super register operands. Super register records in LLVM backends don't get materialized
/// in the tablegen source code; They instead get automatically generated in C++ by the CodeGenTarget class
/// using the RegisterClass record descriptions. This means that we can't refer to the super register records
/// directly in the semantics record files. This class exists so that the SI semantics generator backend
/// can materialize the register itself instead of relying on the CodeGenTarget
class SuperRegFactory<list<Register> regs> {
  list<Register> Regs = regs;
}

//------------------------------------------------------------------------------
// Complex LLVM IR Operation Definitions (i.e. low-level binary operations that
// require multiple LLVM IR operations)
//------------------------------------------------------------------------------

/// Performs a bit-wise inversion operation on any integer value
class LLVMNot<dag val, SILLVMType type> {
  dag ret = (LLVMXor val, (LLVMSExt (ConstantTrue si_llvm_i1_ty), type));
}

/// Breaks down an input value of type inputType into a set of
/// sub values
//class LLVMBreakdownValue<dag input, SILLVMType inputType, int numEls> {
//  assert !eq(inputType.VT.nElem, 1), "inputType must be a scalar value";
//  assert !or(inputType.VT.isFP, inputType.VT.isInteger),
//         "inputType must either be a floating point or an integer";
//  dag BitcastedVal = (LLVMBitCast:$bitcast input,
//                       !cast<SILLVMType>(
//                        si_llvm_#!if(!head(types).VT.isFP, "f", "i")
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
//class LLVMMergeValue<list<dag> ins, SILLVMType type> {
//  assert !not(!empty(ins)), "argument list shouldn't be empty";
//  assert !eq(type.VT.nElem, 1), "type must be a scalar";
//  assert !or(type.VT.isFP, type.VT.isInteger),
//         "type must be a floating point or scalar";
//
//
//  /// The output type is a floating point or an integer, with its
//  /// size being the aggregate of all input types
//  SILLVMType OutputType =
//    !cast<SILLVMType>(llvm_#!if(type.VT.isFP, "f", "i")#
//                          !mul(!size(ins), type.VT.Size)#_ty);
//  /// Type of vector the values will be inserted to before being
//  /// casted to the final scalar
//  SILLVMType MergeVectorType =
//    !cast<SILLVMType>(llvm_v#!size(ins)
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

class LLVMExtractValueConstIdx<dag val, SILLVMType subValType, int idx> {
  assert !gt(subValType.NElem, 1), "subValType must be a vector type";
  assert !gt(subValType.NElem, idx),
         "Index cannot be greater than number of elements";
  assert !gt(idx, 0), "Index must be greater than zero";
  dag Op =
    (LLVMExtractElement (LLVMBitCast val, subValType), idx);
}

class LLVMExtractValueVarIdx<dag val, SILLVMType subValType, dag idx> {
  assert !gt(subValType.NElem, 1), "subValType must be a vector type";
  dag Op =
    (LLVMExtractElement (LLVMBitCast val, subValType), idx);
}

class LLVMExtractSignBit<dag val, SILLVMType type> {
  dag Op = LLVMExtractValueConstIdx<val, type, !sub(type.ElemSize, 1)>.Op;
}

class LLVMSetSignBit<dag val, SILLVMType type, dag sgn> {
  dag Op = (LLVMAnd val, (LLVMShl sgn, !sub(type.ElemSize, 1)));
}

class LLVMExtractBitsIntStartIntNumBits<dag val,
                                        SILLVMType inputType,
                                        bit isSigned, int startIdx,
                                        int numBits = 1> {
  assert !lt(startIdx, inputType.ElemSize), "Start index must be less than the size of the element type";
  assert !eq(inputType.NElem, 1), "type must be a scalar";
  assert !lt(!add(startIdx, numBits), inputType.ElemSize), "Total number of bits requested is out of range";
  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);

  // Coerce the source to the requested width: the shift/mask constants below
  // are typed `inputType`, but `val` (e.g. a 32-bit src1 operand feeding a
  // 64-bit S_BFE_I64) may have a narrower natural width. Without this the
  // generated shift/sub mixes i32 and i64 operands and trips the constant
  // folder's "operand types must match" assertion.
  defvar cval = (LLVMZExtOrTrunc val, inputType);

  // mask = (1 << numBits) - 1, applied for every numBits (the prior numBits!=1
  // branch dropped the mask, leaving the extracted field unmasked).
  dag Op = (LLVMAnd (rightShiftOp cval, (ConstantInt inputType, startIdx)),
                    (ConstantInt inputType, !sub(!shl(1, numBits), 1)));
}

class LLVMExtractBitsDagStartDagNumBits<dag val, SILLVMType inputType, bit isSigned, dag startIdx, dag numBits> {
  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);
  dag Op = (LLVMAnd (rightShiftOp val, startIdx),
                    (LLVMSub (LLVMShl (ConstantInt inputType, 1), numBits), (ConstantInt inputType, 1)));
}

class LLVMExtractBitsIntStartDagNumBits<dag val, SILLVMType inputType, bit isSigned, int startIdx, dag numBits> {
  assert !eq(inputType.NElem, 1), "type must be a scalar";
  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);
  dag Op = (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx)),
                    (LLVMSub (LLVMShl (ConstantInt inputType, 1), numBits),
                                      (ConstantInt inputType, 1)));
}

class SetSCCImplicit<dag val> {
  dag ret = (ImplicitDef SCC, (LLVMZExtOrTrunc val, si_llvm_i32_ty));
}

class GetSCCImplicit {
  dag ret = (LLVMTrunc (ImplicitUse SCC), si_llvm_i1_ty);
}

/// Inverse of \c BallotWaveMask: extracts the current lane's bit from a
/// wave-mask SGPR value (i32 on wave32, i64 on wave64) as an \c i1.
/// Implemented as \c trunc(((mask >> laneId) & 1), i1) where the lane id
/// is produced by \c mbcnt.hi(-1, mbcnt.lo(-1, 0)) — that idiom resolves
/// to a per-lane 0..31 / 0..63 value on wave32 and wave64 respectively,
/// because \c mbcnt.hi on wave32 simply returns its base operand. Use
/// when consuming wave-mask values (VCC, EXEC, sdst-pair, arbitrary
/// SReg-pair masks) where the IR builder expects an \c i1, e.g. as the
/// condition of \c LLVMSelect.
class LaneBitFromWaveMask<dag mask> {
  dag ret = (LLVMTrunc
    (LLVMAnd
      (LLVMLShr mask,
        (LLVMZExtOrTrunc
          (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_hi,
            [(ConstantSignedInt si_llvm_i32_ty, -1),
             (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_lo,
               [(ConstantSignedInt si_llvm_i32_ty, -1),
                (ConstantInt si_llvm_i32_ty, 0)])]),
          (LLVMIntNTy (GetWavefrontSize)))),
      (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 1)),
    si_llvm_i1_ty);
}

//class LLVMExtractBitsDagStartIntNumBits<dag val, SILLVMType inputType, dag startIdx, int numBits = 1> {
//  assert !eq(inputType.VT.nElem, 1), "type must be a scalar";
//  defvar rightShiftOp = !if(isSigned, LLVMAShr, LLVMLShr);
//  dag Op = !if(!eq(numBits, 1),
//             (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx)),
//                                         (ConstantInt inputType, !sub(!shl(1, numBits), 1))),
//             (LLVMAnd (rightShiftOp val, (ConstantInt inputType, startIdx))));
//}
