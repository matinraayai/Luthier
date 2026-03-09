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

/// Superclass for all operations used in the semantics
class SISemOp;

//------------------------------------------------------------------------------
// Constant values
//------------------------------------------------------------------------------

class ConstantInt<VTInt valueType, int val> {
  LLVMType Type = LLVMType<valueType>;
  int Val = val;
}

class ConstantFP<VTFP valueType, int val> {
  LLVMType Type = LLVMType<valueType>;
  int Val = val;
}

//------------------------------------------------------------------------------
// LLVM IR Operation Definitions
//------------------------------------------------------------------------------


/// Represents LLVM operations used to describe the SI instruction semantics
/// records
class LLVMOp<string iRBuilderFunc = ""> : SISemOp {
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
// Complex LLVM IR Operation Definitions (i.e. low-level binary operations that
// require multiple LLVM IR operations)
//------------------------------------------------------------------------------

/// Used to calculate the greatest common value type between
/// two LLVM value types
class GCValTypeTwoInput<LLVMType a, LLVMType b> {
  assert !or(!and(!isa<VTInt>(a.VT), !isa<VInt>(b.VT)),
             !and(!isa<VTFP>(a.VT), !isa<VTFP>(b.VT))),
         "Both LLVM types must be integers or floats";
  /// This line calculates the greatest common divisor of the
  /// type sizes, and then finds the value type associated
  /// with that size
  LLVMType ret = LLVMType<!cast<ValueType>(
                  !if(a.VT.isFP, "f", "i")#
                    !cond(!eq(a, 0): b,
                          !eq(b, 0): a,
                          !eq(a, b): a,
                          !gt(a, b): GCValTypeTwoInput<!sub(a, b), b>.ret,
                          true: GCValTypeTwoInput<a, !sub(b, a)>.ret)
                          )>;
}

/// Used to calculate the greatest common value type between a
/// set of LLVM value types
class GreatestCommonValType<list<LLVMType> types> {
  LLVMType ret = !foldl(!head(types),
                        types,
                        acc,
                        var,
                        GCValTypeTwoInput<acc, var>.ret
                        );
}

class LLVM_MergeValues<list<dag> ins, list<LLVMType> types> {
  assert !eq(!size(ins), !size(types)),
         "the list of ins and types must be of same length";
  assert !not(!empty(ins)), "argument list shouldn't be empty";

  /// The output type is a floating point or an integer, with its
  /// size being the aggregate of all input types
  LLVMType OutputType =
    LLVMType<!cast<ValueType>(!if(!head(types).VT.isFP, "f", "i")#
                                  !foldl(0, types, acc, var,
                                         !add(acc.VT.acc, var.VT.Size)))>;

  list<dag> Op = !listconcat(
    [(LLVMInsertElement:$mergevalTmp
      GreatestCommonValType<types>.ret, !head(ins)), 0],
    !foreach(idx, !range(1, !size(types)),
      (LLVMInsertElement:$mergeval_tmp $mergeval_tmp, ins[idx], idx)),
    [(LLVMBitCast $mergeval_tmp, OutputType)]
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

//------------------------------------------------------------------------------
// InstSI-specific semantic operations
//------------------------------------------------------------------------------

/// Used to get the named operand of the machine instruction (instead of hard
/// coding the operand number)
def GetNamedOperand : SISemOp;

/// Used to set the named operand of the machine instruction (instead of hard
/// coding the operand number)
def SetNamedOperand : SISemOp;

/// Used to set a register directly in case of an implicit def
def ImplicitDef : SISemOp;

/// Used to get a register directly in case of an implicit use
def ImplicitUse : SISemOp;

/// Used to select an operand based on a boolean predicate (different from
/// the LLVMSelect operation)
def SelectOperand;

/// Used to check if the MI has a modifier set (checked via
/// SIInstrInfo::hasModifierSet). Does not translate into an IR
/// operation
def hasModifierSet;

/// Defined here because AMDGPU.td records don't define it
def region_ptr_type : LLVMQualPointerType<2>;