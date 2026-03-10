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


/// Represents LLVM operations used to describe the SI instruction semantics
/// records
class LLVMOp<string iRBuilderFunc> {
  /// The llvm::IRBuilder function used to construct the operation
  string IRBuilderFunc = iRBuilderFunc;
}

def LLVMAdd : LLVMOp<"CreateAdd">;

def LLVMSub : LLVMOp<"CreateSub">;

def LLVMMul : LLVMOp<"CreateMul">;

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
/// the LLVMSelect operation)
def SelectOperand;

/// Used to check if the MI has a modifier set (checked via
/// SIInstrInfo::hasModifierSet). Does not translate into an IR
/// operation
def hasModifierSet;

/// Defined here because AMDGPU.td records don't define it
def region_ptr_type : LLVMQualPointerType<2>;

//------------------------------------------------------------------------------
// Complex LLVM IR Operation Definitions (i.e. low-level binary operations that
// require multiple LLVM IR operations)
//------------------------------------------------------------------------------

/// Breaks down an input value of type inputType into a set of
/// sub values
class LLVMBreakdownValue<dag input, LLVMType inputType, int numEls> {
  assert !eq(inputType.VT.nElem, 1), "inputType must be a scalar value";
  assert !eq(!or(inputType.VT.isFP, inputType.VT.isInteger)),
         "inputType must either be a floating point or an integer";
  dag BitcastedVal = (LLVMBitCast:$bitcast input,
                       !cast<LLVMType>(
                        llvm_#!if(!head(types).VT.isFP, "f", "i")
                           #numEls
                           #_ty
                        )
                      );
  list<dag> BrokenDownVals = !foreach(i, !range(numEls),
    (LLVMExtractElement $bitcast, i));
}

/// Merges a set of dag operands together into a single value
/// Input values must be the same type (scalar integer or
/// floating point)
/// For merging two types that are not the same, they must be
/// broken down into a common type first
class LLVMMergeValue<list<dag> ins, LLVMType type> {
  assert !not(!empty(ins)), "argument list shouldn't be empty";
  assert !eq(type.VT.nElem, 1), "type must be a scalar";
  assert !or(type.VT.isFP, type.VT.isInteger),
         "type must be a floating point or scalar";


  /// The output type is a floating point or an integer, with its
  /// size being the aggregate of all input types
  LLVMType OutputType =
    !cast<LLVMType>(llvm_#!if(type.VT.isFP, "f", "i")#
                          !mul(!size(ins), type.VT.Size)#_ty);
  /// Type of vector the values will be inserted to before being
  /// casted to the final scalar
  LLVMType MergeVectorType =
    !cast<LLVMType>(llvm_v#!size(ins)
                          #_#
                          !if(!head(types).VT.isFP, "f", "i")#
                          type.VT.Size
                          #_ty
                    );

  list<dag> Op = !listconcat(
    [(LLVMInsertElement:$mergevalTmp
      GreatestCommonValType<types>.ret, !head(ins)), 0],
    !foreach(idx, !range(1, !size(types)),
      (LLVMInsertElement:$mergevalTmp $mergevalTmp, ins[idx], idx)),
    [(LLVMBitCast $mergevalTmp, OutputType)]
  );
}

class LLVMExtractValueConstIdx<dag val, LLVMType subValType, int idx> {
  assert !gt(subValType.VT.nElem, 1), "subValType must be a vector type";
  assert !gt(subValType.VT.nElem, idx),
         "Index cannot be greater than number of elements";
  assert !gt(idx, 0), "Index must be greater than zero";
  dag Op =
    (LLVMExtractElement (LLVMBitCast:$tempVal val, subValType), idx);
}

class LLVMExtractValueVarIdx<dag val, LLVMType subValType, dag idx> {
  assert !gt(subValType.VT.nElem, 1), "subValType must be a vector type";
  dag Op =
    (LLVMExtractElement (LLVMBitCast:$tempVal val, subValType), idx);
}