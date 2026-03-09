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

def S_ADD_U32_sem : InstSISemantic<S_ADD_U32> {
   let Semantic = [
     (LLVMAdd:$tmp (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty),
                    (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty)
     ),
     (ImplicitDef SCC, (LLVMCmp SETUGE, $tmp,
                                        ConstantInt<i32, 0x100000000>)),
     (SetNamedOperand $sdst, (LLVMTrunc $tmp, llvm_i32_ty))
   ];
}

def S_SUB_U32_sem : InstSISemantic<S_SUB_U32> {
   let Semantic = [
     (SetNamedOperand $sdst, (LLVMSub (GetNamedOperand $src0),
                                       (GetNamedOperand $src0))
     ),
     (ImplicitDef SCC, (LLVMCmp SETGE, (GetNamedOperand $src1),
                                       (GetNamedOperand $src0)))
   ];
}

def S_ADD_I32_sem : InstSISemantic<S_ADD_I32> {
    let Semantic = [
        (set i32:$sdst, (add i32:$src0, i32:$src1)),
        (set v32i1:$src0_v32i1, (bitconvert i32:$src0, llvm_v32i1_ty)),
        (set i1:$src0_31, (extractelt v32i1:$src0_v32i1, 31)),
        (set v32i1:$src1_v32i1, (bitconvert i32:$src1, llvm_v32i1_ty)),
        (set i1:$src1_31, (extractelt v32i1:$src0_v32i1, 31)),
        (set v32i1:$sdst_v32i1, (bitconvert i32:$sdst, llvm_v32i1_ty)),
        (set i1:$sdst_31, (extractelt v32i1:$sdst_v32i1, 31)),
        (set SCC, (and (scmp SETEQ, i1:$src0_31, i1:$src1_31),
                  (scmp SETEQ, i1:$src0_31, i1:$dst_31)))
    ];
    let Doc = [
        "tmp = S0.i32 + S1.i32;",
        "SCC = ((S0.u32[31] == S1.u32[31]) && (S0.u32[31] != tmp.u32[31]));",
        "// signed overflow."
        "D0.i32 = tmp.i32"
    ];
}

def S_SUB_I32_sem : InstSISemantic<S_SUB_I32> {
    let Semantic = [
        (set i32:$sdst, (sub i32:$src0, i32:$src1)),
        (set v32i1:$src0_v32i1, (bitconvert i32:$src0, llvm_v32i1_ty)),
        (set i1:$src0_31, (extractelt v32i1:$src0_v32i1, 31)),
        (set v32i1:$src1_v32i1, (bitconvert i32:$src1, llvm_v32i1_ty)),
        (set i1:$src1_31, (extractelt v32i1:$src0_v32i1, 31)),
        (set v32i1:$sdst_v32i1, (bitconvert i32:$sdst, llvm_v32i1_ty)),
        (set i1:$sdst_31, (extractelt v32i1:$sdst_v32i1, 31)),
        (set SCC, (and (scmp SETEQ, i1:$src0_31, i1:$src1_31),
                  (scmp SETEQ, i1:$src0_31, i1:$dst_31)))
    ];
    let Doc = [
        "tmp = S0.i32 - S1.i32;"
        "SCC = ((S0.u32[31] != S1.u32[31]) && (S0.u32[31] != tmp.u32[31]));"
        "// signed overflow."
        "D0.i32 = tmp.i32"
    ];
}

def S_ADDC_U32_sem : InstSISemantic<S_ADDC_U32> {
    let Semantic = [
       (set i64:$tmp, (add (zext i32:$src0, llvm_i64_ty), (zext i32:$src1, llvm_i64_ty))),
       (set i64:$tmp1, (add i64:$tmp, (zext SCC, llvm_i64_ty))),
       (set SCC, (ucmp SETUGE, i64:$tmp1, 0x100000000)),
       (set i32:$sdst, (trunc i64:$tmp1, i32))
    ];

    let Doc = [
        "tmp = 64'U(S0.u32) + 64'U(S1.u32) + SCC.u64;"
        "SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;"
        "// unsigned overflow or carry-out for S_ADDC_U32."
        "D0.u32 = tmp.u32"
    ];
}

def S_SUBB_U32 : InstSISemantic<S_SUBB_U32> {
    let Semantic = [
        (set i32:$sdst, (sub i32:$src0, (sub i32:$src1, (zext SCC, llvm_i32_ty)))),
        (set SCC, (ucmp SETUGT,  ((zext )), 0x100000000)),
        (set i32:$sdst, (trunc i64:$tmp1, i32))
    ];
    let Doc = [
        "tmp = S0.u32 - S1.u32 - SCC.u32;"
        "SCC = 64'U(S1.u32) + SCC.u64 > 64'U(S0.u32) ? 1'1U : 1'0U;"
        "// unsigned overflow or carry-out for S_SUBB_U32."
        "D0.u32 = tmp.u32"
    ];
}


def S_MIN_I32_sem : InstSISemantic<S_MIN_I32> {
    let Semantic = [
        (set SCC, (scmp SETLT, i32:$src0, i32:$src1)),
        (set i32:$sdst, (select SCC, i32:$src0, i32:$src1))
    ];
    let Doc = [
        "SCC = S0.i32 < S1.i32;",
        "D0.i32 = SCC ? S0.i32 : S1.i32"
    ];
}


def DS_ADD_U32_sem : InstSISemantic<DS_ADD_U32> {
  let Semantic = [
     (LLVMLoad:$tmp llvm_i32_ty,
       CalculateSingleOffsetDSAddr<(GetNamedOperand $addr),
                                   (GetNamedOperand $offset),
                                   (SelectOperand (hasModifierSet $gds), region_ptr_type, local_ptr_ty)>.ret),
     (LLVMStore (LLVMAdd $tmp, (GetNamedOperand $data0)))
  ];
}
////  let Semantic = [(set addr, (.call CalcDsAddr:B.32, (.cast ADDR:U.32, B.32, B.32), (.cast OFFSET0, B.32, B.32), (.cast OFFSET1, B.32, B.32), U.32),
////z                  (set tmp, (.cast MEM:B.32[addr:U.32], U.32, U.32),
////                  (+= (.cast MEM:B.32[addr:U.32], U.32, U.32), (.cast DATA, U.32, U.32), U.32),
////                  (set (.cast RETURN_DATA, U.32, U.32), tmp:U.32,
////                  	U.32]
//}
