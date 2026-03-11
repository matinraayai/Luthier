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
include "SISemOps.td"

//------------------------------------------------------------------------------
// Convenience Functions
//------------------------------------------------------------------------------

class CalculateSingleOffsetDSAddr<dag addr, dag offset,
                                  LLVMQualPointerType localOrRegionPtr> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr addr, localOrRegionPtr), offset);
}

class CalculateDoubleOffsetDSAddr<dag addr, dag offset0, dag offset1,
                                  LLVMQualPointerType localOrRegionPtr,
                                  LLVMType dataType> {
  int Adj = !cond(!eq(dataType.VT.Size, 64) : 8,
                  (!or(!eq(dataType.VT.Size, 4),
                       !eq(dataType.VT.Size, 8),
                       !eq(dataType.VT.Size, 16))
                       ) : 4);
  dag ret = (LLVMPtrAdd (LLVMIntToPtr addr, localOrRegionPtr),
                         (LLVMAdd (LLVMMul offset0, Adj),
                                   (LLVMMul offset1, Adj)
                                   )
                         );
}

class InstSISemantic<InstSI inst, list<dag> semantic = []> {
  InstSI Instruction = inst;
  list<dag> Semantic = semantic;
}

class InstSIIntrinsic<InstSI inst, Intrinsic intrinsic> {
  InstSI Instruction = inst;
  Intrinsic Intrinsic = intrinsic;
}

//------------------------------------------------------------------------------
// SOP2 Instructions
//------------------------------------------------------------------------------

def S_ADD_U32_sem : InstSISemantic<S_ADD_U32> {
   let Semantic = [
     (DefVal $tmp,
       (LLVMAdd (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty),
                (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty))),
     (ImplicitDef SCC, (LLVMCmp SETUGE, $tmp, (ConstantInt llvm_i64_ty, 0x100000000))),
     (SetNamedOperand $sdst, (LLVMTrunc $tmp, llvm_i32_ty))
   ];
}

def S_SUB_U32_sem : InstSISemantic<S_SUB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src0))),
    (ImplicitDef SCC, (LLVMCmp SETUGT, (GetNamedOperand $src1),
                                       (GetNamedOperand $src0))),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_ADD_I32_sem : InstSISemantic<S_ADD_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1))),
    (DefVal $src0_31, LLVMExtractValueConstIdx<(GetNamedOperand $src0), llvm_v32i1_ty, 31>.Op),
    (DefVal $src1_31, LLVMExtractValueConstIdx<(GetNamedOperand $src1), llvm_v32i1_ty, 31>.Op),
    (DefVal $sdst_31, LLVMExtractValueConstIdx<(GetNamedOperand $sdst), llvm_v32i1_ty, 31>.Op),
    (ImplicitDef SCC, (LLVMAnd (LLVMCmp SETEQ, $src0_31, $src1_31),
                               (LLVMCmp SETNE, $src0_31, $sdst_31))
    ),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_SUB_I32_sem : InstSISemantic<S_SUB_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src1))),
    (DefVal $src0_31, LLVMExtractValueConstIdx<(GetNamedOperand $src0), llvm_v32i1_ty, 31>.Op),
    (DefVal $src1_31, LLVMExtractValueConstIdx<(GetNamedOperand $src1), llvm_v32i1_ty, 31>.Op),
    (DefVal $sdst_31, LLVMExtractValueConstIdx<(GetNamedOperand $sdst), llvm_v32i1_ty, 31>.Op),
    (ImplicitDef SCC, (LLVMAnd (LLVMCmp SETNE, $src0_31, $src1_31),
                               (LLVMCmp SETNE, $src0_31, $sdst_31))),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_ADDC_U32_sem : InstSISemantic<S_ADDC_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMAdd (LLVMAdd (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty),
                                    (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty)),
                                    (LLVMZExt (ImplicitUse SCC), llvm_i64_ty))),
    (ImplicitDef SCC, (LLVMCmp SETUGE, $tmp,
                                       (ConstantInt llvm_i64_ty, 0x100000000))),
    (SetNamedOperand $sdst, (LLVMTrunc $tmp, llvm_i32_ty))
  ];
}

def S_SUBB_U32_sem : InstSISemantic<S_SUBB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMSub (LLVMSub (GetNamedOperand $src0),
                                    (GetNamedOperand $src1)),
                                    (LLVMZExt (ImplicitUse SCC), llvm_i32_ty))),
    (ImplicitDef SCC, (LLVMCmp SETUGT, (LLVMAdd (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty),
                                                (LLVMZExt (ImplicitUse SCC), llvm_i64_ty)),
                                       (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty))),
    (SetNamedOperand $sdst, $tmp)
  ];
}

foreach i = 0-3 in {
  defvar op = ["S_MIN_I32", "S_MIN_U32", "S_MAX_I32", "S_MAX_U32"][i];
  defvar cmpSgn = [SETLT, SETULT, SETGE, SETUGE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      (ImplicitDef SCC, (LLVMCmp cmpSgn, (GetNamedOperand $src0), (GetNamedOperand $src1))),
      (SetNamedOperand $sdst, (LLVMZExt (ImplicitUse SCC), llvm_i32_ty))
    ];
  }
}

foreach size = ["32", "64"] in {
  def S_CSELECT_B#size#_sem : InstSISemantic<S_CSELECT_B#size> {
    let Semantic = [(SetNamedOperand $sdst,
                    (LLVMSelect (ImplicitUse SCC), (GetNamedOperand $src0), (GetNamedOperand $src1)))];
  }
}

foreach op = ["And", "Or", "Xor"] in {
  foreach size = ["32", "64"] in {
    def S_#!toupper(op)#_B#size#_sem : InstSISemantic<S_#!toupper(op)#_B#size> {
      let Semantic =
        [(SetNamedOperand $sdst, (LLVM#op (GetNamedOperand $src0), (GetNamedOperand $src1))),
         (ImplicitDef SCC, (LLVMCmp SETNE, (GetNamedOperand $sdst), (ConstantInt llvm_i#size#_ty, 0)))
        ];
    }
  }
}

foreach op = ["And", "Or"] in {
  foreach size = ["32", "64"] in {
    def S_#!toupper(op)#N2_B#size#_sem : InstSISemantic<S_#!toupper(op)#N2_B#size> {
      let Semantic = [
        (SetNamedOperand $sdst, (LLVM#op (GetNamedOperand $src0), (LLVMNot<(GetNamedOperand $src1)>.ret))),
        (ImplicitDef SCC, (LLVMCmp SETNE, (GetNamedOperand $sdst), (ConstantInt llvm_i#size#_ty, 0)))
      ];
    }
  }
}

foreach op = ["And", "Or", "Xor"] in {
  foreach size = ["32", "64"] in {
    def S_N#!toupper(op)#_B#size#_sem : InstSISemantic<S_N#!toupper(op)#_B#size> {
      let Semantic = [
        (SetNamedOperand $sdst, (LLVM#op (GetNamedOperand $src0), (LLVMNot<(GetNamedOperand $src1)>.ret))),
        (ImplicitDef SCC, (LLVMCmp SETNE, (GetNamedOperand $sdst), (ConstantInt llvm_i#size#_ty, 0)))
      ];
    }
  }
}


foreach op = ["LShl", "LShr", "AShr"] in {
  foreach sizeIdx = 0-1 in {
    defvar size = ["32", "64"][sizeIdx];
    defvar bitSet = [0xF, 0x3F][sizeIdx];
    defvar shouldSignExtendSrc0 = !eq(op, "AShr");
    def S_#!toupper(op)#_B#size#_sem : InstSISemantic<S_N#!toupper(op)#_B#size> {
      let Semantic = [
        (SetNamedOperand $sdst, (LLVM#op
                                         !if(shouldSignExtendSrc0,
                                            (LLVMSExt (GetNamedOperand $src0), llvm_i32_ty),
                                            (GetNamedOperand $src0)),
                                         (LLVMAnd (GetNamedOperand $src1), (ConstantInt llvm_i#size_ty, bitSet)))),
        (ImplicitDef SCC, (LLVMCmp SETNE, (GetNamedOperand $sdst), (ConstantInt llvm_i#size#_ty, 0)))
      ];
    }
  }
}

foreach size = [32, 64] in {
  defvar bitSet = !sub(size, 1);
  def S_BFMB_#size#_sem : InstSISemantic<S_BFMB_#size> {
    let Semantic = [
      (DefVal $src0_bits_lo, (LLVMAnd (GetNamedOperand $src0), (ConstantInt llvm_i#size_ty, bitSet))),
      (DefVal $src1_bits_lo, (LLVMAnd (GetNamedOperand $src1), (ConstantInt llvm_i#size_ty, bitSet))),
      (SetNamedOperand $sdts, (LLVMLShr
                                (LLVMSub (LLVMLShr (ConstantInt llvm_i32_ty, 1), $src0_bits_lo),
                                $src1_bits_lo)))
    ];
  }
}


def DS_ADD_U32_sem : InstSISemantic<DS_ADD_U32> {
  let Semantic = [
     (LLVMLoad:$tmp llvm_i32_ty,
       CalculateSingleOffsetDSAddr<(GetNamedOperand $addr),
                                   (GetNamedOperand $offset),
                                   (SelectOperand (hasModifierSet $gds), region_ptr_type, local_ptr_ty)>.ret),
     (LLVMStore $tmp, (GetNamedOperand $data))
  ];
}
