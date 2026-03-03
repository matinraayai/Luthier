include "AMDGPU.td"


class InstSISemantic<InstSI inst, list<dag> semantic = [], list<string> doc = []> {
  InstSI Instruction = inst;
  list<dag> Semantic = semantic;
  list<string> Doc = doc;
}

class InstSIIntrinsic<InstSI inst, Intrinsic intrinsic> {
  InstSI Instruction = inst;
  Intrinsic Intrinsic = intrinsic;
}

def S_ADD_U32_sem : InstSISemantic<S_ADD_U32> {
   let Semantic = [
       (set i64:$tmp, (add (zext i32:$src0, llvm_i64_ty), (zext i32:$src1, llvm_i64_ty))),
       (set SCC, (ucmp SETUGE, i64:$tmp, 0x100000000)),
       (set i32:$sdst, (trunc i64:$tmp, i32))
   ];
   let Doc = [
       "tmp = 64'U(S0.u32) + 64'U(S1.u32);",
       "SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U;",
       "// unsigned overflow or carry-out for S_ADDC_U32.",
       "D0.u32 = tmp.u32"
   ];
}

def S_SUB_U32_sem : InstSISemantic<S_SUB_U32> {
   let Semantic = [
       (set i32:$sdst, (sub i32:$src0, i32:$src1)),
       (set SCC, (ucmp SETGE, i32:$src1, i32:$src0))
   ];
   let Doc = [
       "tmp = S0.u32 - S1.u32;",
       "SCC = S1.u32 > S0.u32 ? 1'1U : 1'0U;",
       "// unsigned overflow or carry-out for S_SUBB_U32."
       "D0.u32 = tmp.u32"
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
        (set i64:$tmp, (add (zext i32:$src0, llvm_i64_ty), (zext i32:$src1, llvm_i64_ty))),
        (set i64:$tmp1, (add i64:$tmp, (zext SCC, llvm_i64_ty))),
        (set SCC, (ucmp SETUGE, i64:$tmp1, 0x100000000)),
        (set i32:$sdst, (trunc i64:$tmp1, i32))
    ];

    let Doc = [
        "tmp = S0.u32 - S1.u32 - SCC.u32;"
        "SCC = 64'U(S1.u32) + SCC.u64 > 64'U(S0.u32) ? 1'1U : 1'0U;"
        "// unsigned overflow or carry-out for S_SUBB_U32."
        "D0.u32 = tmp.u32"
    ];
}

