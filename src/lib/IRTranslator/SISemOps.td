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

//------------------------------------------------------------------------------
// Complex LLVM IR Operation Definitions (i.e. low-level binary operations that
// require multiple LLVM IR operations and have a native generic MIR opcode)
//------------------------------------------------------------------------------

class LLVM_ComplexOp;

class LLVMMergeValueArgs<dag inputVal, LLVMType type> {
  dag InputVal = inputVal;
  dag Type = type;
}

class LLVM_MergeValues<list<LLVMMergeValueArgs> ins> {
  dag ret =  zzzzz
}

class LLVMExtractValue : LLVM_ComplexOp;

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