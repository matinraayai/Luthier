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
    (set i64:$tmp, (add i32:$src0, i32:$src1)),
    (set SCC, (ucmp i64:$tmp, 0x100000000)),
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
    (set SCC, (ucmp i32:$src1, i32:$src0))
   ];
   let Doc = [
    "tmp = S0.u32 - S1.u32;",
    "SCC = S1.u32 > S0.u32 ? 1'1U : 1'0U;",
    "// unsigned overflow or carry-out for S_SUBB_U32."
    "D0.u32 = tmp.u32"
   ];
}

//def S_ADD_I32_sem : InstSISemantic<S_ADD_I32, [(set i32:$sdst, (add i32:$src0, i32:$src1)),
//                                               (set SCC, (and ((), ()), (extractelt)))],
//                                                ["tmp = S0.u32 - S1.u32;",
//                                                 "SCC = S1.u32 > S0.u32 ? 1'1U : 1'0U;",
//                                                 "// unsigned overflow or carry-out for S_SUBB_U32."
//                                                 "D0.u32 = tmp.u32"]>;



//def S_CSELECT_B32_sem : InstSISemantic<S_CSELECT_B32, [(set i64:$dst)]>

