//===- SMEMSem.td -----------------------------------------------------*- tblgen -*-===//
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


//------------------------------------------------------------------------------
// S_BUFFER_LOAD Instructions
// S_BUFFER_LOAD_DWORD: Load dword(s) from buffer descriptor + offset into SGPRs.
// Sources:
//   southern-islands-instruction-set-architecture.pdf, pages 140-142.
//   sea-islands-instruction-set-architecture_0.pdf, pages 142-144.
//   gcn3-instruction-set-architecture.pdf, page 154.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 123.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 119.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 149.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 161.
//   rdna-shader-instruction-set-architecture.pdf, page 139.
//   rdna2-shader-instruction-set-architecture.pdf, page 138.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 256 (as S_BUFFER_LOAD_B32).
//   rdna4-instruction-set-architecture.pdf, page 292 (as S_BUFFER_LOAD_B32).
//------------------------------------------------------------------------------

def S_BUFFER_LOAD_DWORD_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORD_IMM> {
  let Semantic = [
    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i32_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORD_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_DWORD_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORD_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORD_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX2_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX2_IMM> {
  let Semantic = [
    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v2i32_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX2_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX2_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v2i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX2_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v2i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX3_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX3_IMM> {
  let Semantic = [
    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v3i32_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX3_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX3_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v3i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX3_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX3_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v3i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX4_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX4_IMM> {
  let Semantic = [
    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v4i32_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX4_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX4_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v4i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX4_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX4_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v4i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX8_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX8_IMM> {
  let Semantic = [
    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v8i32_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX8_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX8_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v8i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX8_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX8_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v8i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX16_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX16_IMM> {
  let Semantic = [
    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v16i32_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX16_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX16_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v16i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_LOAD_DWORDX16_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_DWORDX16_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_v16i32_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0)))
  ];
}

// S_BUFFER_LOAD_U8: Load 8 bits of unsigned data from a scalar buffer surface,
// zero extend to 32 bits and store the result into a scalar register.
// SDATA.u32 = 32'U({ 24'0U, MEM[ADDR].u8 })
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 295 (opcode 25).
def S_BUFFER_LOAD_U8_IMM_sem : InstSISemantic<S_BUFFER_LOAD_U8_IMM> {
  let Semantic = [
    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i8_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_U8_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_U8_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i8_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_U8_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_U8_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i8_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

// S_BUFFER_LOAD_I8: Load 8 bits of signed data from a scalar buffer surface,
// sign extend to 32 bits and store the result into a scalar register.
// SDATA.i32 = 32'I(signext(MEM[ADDR].i8))
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 295 (opcode 24).
def S_BUFFER_LOAD_I8_IMM_sem : InstSISemantic<S_BUFFER_LOAD_I8_IMM> {
  let Semantic = [
    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i8_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_I8_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_I8_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i8_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_I8_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_I8_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i8_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

// S_BUFFER_LOAD_U16: Load 16 bits of unsigned data from a scalar buffer surface,
// zero extend to 32 bits and store the result into a scalar register.
// SDATA.u32 = 32'U({ 16'0U, MEM[ADDR].u16 })
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 295 (opcode 27).
def S_BUFFER_LOAD_U16_IMM_sem : InstSISemantic<S_BUFFER_LOAD_U16_IMM> {
  let Semantic = [
    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i16_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_U16_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_U16_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i16_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_U16_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_U16_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i16_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

// S_BUFFER_LOAD_I16: Load 16 bits of signed data from a scalar buffer surface,
// sign extend to 32 bits and store the result into a scalar register.
// SDATA.i32 = 32'I(signext(MEM[ADDR].i16))
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 295 (opcode 26).
def S_BUFFER_LOAD_I16_IMM_sem : InstSISemantic<S_BUFFER_LOAD_I16_IMM> {
  let Semantic = [
    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i16_ty,

        (GetNamedOperand $sbase), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_I16_SGPR_sem : InstSISemantic<S_BUFFER_LOAD_I16_SGPR> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i16_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

def S_BUFFER_LOAD_I16_SGPR_IMM_sem : InstSISemantic<S_BUFFER_LOAD_I16_SGPR_IMM> {
  let Semantic = [
    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $tmp,

      (LLVMIntrinsic int_amdgcn_s_buffer_load, llvm_i16_ty,

        (GetNamedOperand $sbase), $total_offset,

        (ConstantInt llvm_i32_ty, 0))),

    (SetNamedOperand $sdst, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_STORE Instructions
//------------------------------------------------------------------------------

def S_BUFFER_STORE_DWORD_IMM_sem : InstSISemantic<S_BUFFER_STORE_DWORD_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORD_SGPR_sem : InstSISemantic<S_BUFFER_STORE_DWORD_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORD_SGPR_IMM_sem : InstSISemantic<S_BUFFER_STORE_DWORD_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORDX2_IMM_sem : InstSISemantic<S_BUFFER_STORE_DWORDX2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORDX2_SGPR_sem : InstSISemantic<S_BUFFER_STORE_DWORDX2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORDX2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_STORE_DWORDX2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORDX4_IMM_sem : InstSISemantic<S_BUFFER_STORE_DWORDX4_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORDX4_SGPR_sem : InstSISemantic<S_BUFFER_STORE_DWORDX4_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_STORE_DWORDX4_SGPR_IMM_sem : InstSISemantic<S_BUFFER_STORE_DWORDX4_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_store, llvm_void_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (ADD)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_ADD_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_ADD_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_ADD_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_ADD_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_ADD_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_ADD_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_ADD_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_ADD_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_ADD_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_ADD_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_ADD_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_ADD_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_ADD_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_add, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (AND)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_AND_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_AND_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_AND_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_AND_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_AND_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_AND_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_AND_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_AND_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_AND_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_AND_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_AND_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_AND_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_AND_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_and, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (OR)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_OR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_OR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_OR_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_OR_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_OR_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_OR_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_OR_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_OR_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_OR_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_OR_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_OR_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_OR_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_OR_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_or, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (XOR)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_XOR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_XOR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_XOR_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_XOR_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_XOR_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_XOR_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_XOR_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_XOR_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_XOR_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_XOR_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_XOR_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_XOR_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_XOR_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_xor, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (SUB)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_SUB_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SUB_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SUB_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SUB_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SUB_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SUB_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SUB_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SUB_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SUB_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SUB_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SUB_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SUB_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SUB_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_sub, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (SWAP)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_SWAP_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SWAP_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SWAP_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SWAP_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SWAP_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SWAP_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SWAP_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SWAP_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SWAP_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SWAP_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SWAP_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SWAP_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SWAP_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_swap, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (SMAX)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_SMAX_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMAX_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMAX_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMAX_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMAX_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMAX_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMAX_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMAX_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMAX_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMAX_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMAX_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMAX_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMAX_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (SMIN)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_SMIN_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMIN_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMIN_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMIN_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMIN_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMIN_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMIN_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMIN_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMIN_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMIN_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_SMIN_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_SMIN_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_SMIN_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_smin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (UMAX)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_UMAX_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMAX_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMAX_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMAX_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMAX_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMAX_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMAX_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMAX_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMAX_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMAX_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMAX_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMAX_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMAX_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umax, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (UMIN)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_UMIN_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMIN_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMIN_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMIN_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMIN_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMIN_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMIN_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMIN_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMIN_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMIN_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_UMIN_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

      (GetNamedOperand $sdata), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_UMIN_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_UMIN_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_umin, llvm_i32_ty,

        (GetNamedOperand $sdata), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (INC)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_INC_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i32_ty,

      (ConstantInt llvm_i32_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_INC_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i32_ty,

        (ConstantInt llvm_i32_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_INC_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i32_ty,

      (ConstantInt llvm_i32_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_INC_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i32_ty,

        (ConstantInt llvm_i32_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_INC_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i32_ty,

      (ConstantInt llvm_i32_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_INC_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i32_ty,

        (ConstantInt llvm_i32_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_INC_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i64_ty,

      (ConstantInt llvm_i64_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_INC_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i64_ty,

        (ConstantInt llvm_i64_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_INC_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i64_ty,

      (ConstantInt llvm_i64_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_INC_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i64_ty,

        (ConstantInt llvm_i64_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_INC_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i64_ty,

      (ConstantInt llvm_i64_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_INC_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_INC_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_inc, llvm_i64_ty,

        (ConstantInt llvm_i64_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (DEC)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_DEC_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i32_ty,

      (ConstantInt llvm_i32_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_DEC_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i32_ty,

        (ConstantInt llvm_i32_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_DEC_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i32_ty,

      (ConstantInt llvm_i32_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_DEC_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i32_ty,

        (ConstantInt llvm_i32_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_DEC_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i32_ty,

      (ConstantInt llvm_i32_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_DEC_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i32_ty,

        (ConstantInt llvm_i32_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_DEC_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i64_ty,

      (ConstantInt llvm_i64_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_DEC_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i64_ty,

        (ConstantInt llvm_i64_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_DEC_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i64_ty,

      (ConstantInt llvm_i64_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_DEC_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i64_ty,

        (ConstantInt llvm_i64_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

def S_BUFFER_ATOMIC_DEC_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i64_ty,

      (ConstantInt llvm_i64_ty, 1), $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_DEC_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_DEC_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (SetNamedOperand $sdst,

      (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_dec, llvm_i64_ty,

        (ConstantInt llvm_i64_ty, 1), $rsrc,

        (ConstantInt llvm_i32_ty, 0), $total_offset,

        (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)))
  ];
}

//------------------------------------------------------------------------------
// S_BUFFER_ATOMIC Instructions (CMPSWAP)
//------------------------------------------------------------------------------

def S_BUFFER_ATOMIC_CMPSWAP_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i32_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i32_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i32_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i32_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i32_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i32_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_X2_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_X2_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i64_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_X2_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i64_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $offset),

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i64_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i64_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR_IMM_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i64_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0))
  ];
}

def S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_BUFFER_ATOMIC_CMPSWAP_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $rsrc, (LLVMInsertElement (GetNamedOperand $sbase),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (ConstantInt llvm_i32_ty, 3))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset))),

    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),

    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),

    (LLVMIntrinsic int_amdgcn_struct_buffer_atomic_cmpswap, llvm_i64_ty,

      $src, $cmp, $rsrc,

      (ConstantInt llvm_i32_ty, 0), $total_offset,

      (ConstantInt llvm_i32_ty, 0), (ConstantInt llvm_i32_ty, 0)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

//------------------------------------------------------------------------------
// S_LOAD Instructions
//------------------------------------------------------------------------------

// S_LOAD Instructions
// S_LOAD_DWORD: Load dword(s) from flat address (sbase + offset) into SGPRs.
// Sources:
//   southern-islands-instruction-set-architecture.pdf, pages 138-139.
//   sea-islands-instruction-set-architecture_0.pdf, pages 145-146.
//   gcn3-instruction-set-architecture.pdf, page 159.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 122.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 118.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 146.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 158.
//   rdna-shader-instruction-set-architecture.pdf, page 139.
//   rdna2-shader-instruction-set-architecture.pdf, page 138.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 255 (as S_LOAD_B32).
//   rdna4-instruction-set-architecture.pdf, page 289 (as S_LOAD_B32).
//------------------------------------------------------------------------------

def S_LOAD_DWORD_IMM_sem : InstSISemantic<S_LOAD_DWORD_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def S_LOAD_DWORD_SGPR_sem : InstSISemantic<S_LOAD_DWORD_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def S_LOAD_DWORD_SGPR_IMM_sem : InstSISemantic<S_LOAD_DWORD_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def S_LOAD_DWORDX2_IMM_sem : InstSISemantic<S_LOAD_DWORDX2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX2_SGPR_sem : InstSISemantic<S_LOAD_DWORDX2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX2_SGPR_IMM_sem : InstSISemantic<S_LOAD_DWORDX2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX3_IMM_sem : InstSISemantic<S_LOAD_DWORDX3_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX3_SGPR_sem : InstSISemantic<S_LOAD_DWORDX3_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX3_SGPR_IMM_sem : InstSISemantic<S_LOAD_DWORDX3_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX4_IMM_sem : InstSISemantic<S_LOAD_DWORDX4_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX4_SGPR_sem : InstSISemantic<S_LOAD_DWORDX4_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX4_SGPR_IMM_sem : InstSISemantic<S_LOAD_DWORDX4_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX8_IMM_sem : InstSISemantic<S_LOAD_DWORDX8_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_v8i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX8_SGPR_sem : InstSISemantic<S_LOAD_DWORDX8_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp, (LLVMLoad llvm_v8i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX8_SGPR_IMM_sem : InstSISemantic<S_LOAD_DWORDX8_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp, (LLVMLoad llvm_v8i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX16_IMM_sem : InstSISemantic<S_LOAD_DWORDX16_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_v16i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX16_SGPR_sem : InstSISemantic<S_LOAD_DWORDX16_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp, (LLVMLoad llvm_v16i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_LOAD_DWORDX16_SGPR_IMM_sem : InstSISemantic<S_LOAD_DWORDX16_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp, (LLVMLoad llvm_v16i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

//------------------------------------------------------------------------------
// S_STORE Instructions
//------------------------------------------------------------------------------

def S_STORE_DWORD_IMM_sem : InstSISemantic<S_STORE_DWORD_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORD_SGPR_sem : InstSISemantic<S_STORE_DWORD_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORD_SGPR_IMM_sem : InstSISemantic<S_STORE_DWORD_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORDX2_IMM_sem : InstSISemantic<S_STORE_DWORDX2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORDX2_SGPR_sem : InstSISemantic<S_STORE_DWORDX2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORDX2_SGPR_IMM_sem : InstSISemantic<S_STORE_DWORDX2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORDX4_IMM_sem : InstSISemantic<S_STORE_DWORDX4_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORDX4_SGPR_sem : InstSISemantic<S_STORE_DWORDX4_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_STORE_DWORDX4_SGPR_IMM_sem : InstSISemantic<S_STORE_DWORDX4_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

//------------------------------------------------------------------------------
// S_SCRATCH_LOAD Instructions
//------------------------------------------------------------------------------

def S_SCRATCH_LOAD_DWORD_IMM_sem : InstSISemantic<S_SCRATCH_LOAD_DWORD_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def S_SCRATCH_LOAD_DWORD_SGPR_sem : InstSISemantic<S_SCRATCH_LOAD_DWORD_SGPR> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $soff_scaled)),
    (SetNamedOperand $sdst, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def S_SCRATCH_LOAD_DWORD_SGPR_IMM_sem : InstSISemantic<S_SCRATCH_LOAD_DWORD_SGPR_IMM> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), $soff_scaled)),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $total_offset)),
    (SetNamedOperand $sdst, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def S_SCRATCH_LOAD_DWORDX2_IMM_sem : InstSISemantic<S_SCRATCH_LOAD_DWORDX2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_SCRATCH_LOAD_DWORDX2_SGPR_sem : InstSISemantic<S_SCRATCH_LOAD_DWORDX2_SGPR> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $soff_scaled)),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_SCRATCH_LOAD_DWORDX2_SGPR_IMM_sem : InstSISemantic<S_SCRATCH_LOAD_DWORDX2_SGPR_IMM> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), $soff_scaled)),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $total_offset)),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_SCRATCH_LOAD_DWORDX4_IMM_sem : InstSISemantic<S_SCRATCH_LOAD_DWORDX4_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_SCRATCH_LOAD_DWORDX4_SGPR_sem : InstSISemantic<S_SCRATCH_LOAD_DWORDX4_SGPR> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $soff_scaled)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

def S_SCRATCH_LOAD_DWORDX4_SGPR_IMM_sem : InstSISemantic<S_SCRATCH_LOAD_DWORDX4_SGPR_IMM> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), $soff_scaled)),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $total_offset)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $sdst, $tmp)
  ];
}

//------------------------------------------------------------------------------
// S_SCRATCH_STORE Instructions
//------------------------------------------------------------------------------

def S_SCRATCH_STORE_DWORD_IMM_sem : InstSISemantic<S_SCRATCH_STORE_DWORD_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   (GetNamedOperand $offset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORD_SGPR_sem : InstSISemantic<S_SCRATCH_STORE_DWORD_SGPR> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $soff_scaled)),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORD_SGPR_IMM_sem : InstSISemantic<S_SCRATCH_STORE_DWORD_SGPR_IMM> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), $soff_scaled)),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $total_offset)),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORDX2_IMM_sem : InstSISemantic<S_SCRATCH_STORE_DWORDX2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   (GetNamedOperand $offset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORDX2_SGPR_sem : InstSISemantic<S_SCRATCH_STORE_DWORDX2_SGPR> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $soff_scaled)),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORDX2_SGPR_IMM_sem : InstSISemantic<S_SCRATCH_STORE_DWORDX2_SGPR_IMM> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), $soff_scaled)),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $total_offset)),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORDX4_IMM_sem : InstSISemantic<S_SCRATCH_STORE_DWORDX4_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   (GetNamedOperand $offset))),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORDX4_SGPR_sem : InstSISemantic<S_SCRATCH_STORE_DWORDX4_SGPR> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $soff_scaled)),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

def S_SCRATCH_STORE_DWORDX4_SGPR_IMM_sem : InstSISemantic<S_SCRATCH_STORE_DWORDX4_SGPR_IMM> {
  let Semantic = [
    (DefVal $soff_scaled, (LLVMMul (GetNamedOperand $soffset), (ConstantInt llvm_i32_ty, 64))),

    (DefVal $total_offset, (LLVMAdd (GetNamedOperand $offset), $soff_scaled)),

    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), scratch_ptr_ty),

                                   $total_offset)),
    (LLVMStore (GetNamedOperand $sdata), $tmp_addr)
  ];
}

//------------------------------------------------------------------------------
// S_ATOMIC Instructions (flat address space)
//------------------------------------------------------------------------------

def S_ATOMIC_ADD_IMM_sem : InstSISemantic<S_ATOMIC_ADD_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_ADD_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_ADD_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_ADD_X2_IMM_sem : InstSISemantic<S_ATOMIC_ADD_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_ADD_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_ADD_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_AND_IMM_sem : InstSISemantic<S_ATOMIC_AND_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_AND_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_AND_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_AND_X2_IMM_sem : InstSISemantic<S_ATOMIC_AND_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_AND_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_AND_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_OR_IMM_sem : InstSISemantic<S_ATOMIC_OR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_OR_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_OR_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_OR_X2_IMM_sem : InstSISemantic<S_ATOMIC_OR_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_OR_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_OR_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_XOR_IMM_sem : InstSISemantic<S_ATOMIC_XOR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_XOR_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_XOR_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_XOR_X2_IMM_sem : InstSISemantic<S_ATOMIC_XOR_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_XOR_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_XOR_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SUB_IMM_sem : InstSISemantic<S_ATOMIC_SUB_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SUB_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SUB_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SUB_X2_IMM_sem : InstSISemantic<S_ATOMIC_SUB_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SUB_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SUB_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SWAP_IMM_sem : InstSISemantic<S_ATOMIC_SWAP_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SWAP_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SWAP_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SWAP_X2_IMM_sem : InstSISemantic<S_ATOMIC_SWAP_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SWAP_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SWAP_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMAX_IMM_sem : InstSISemantic<S_ATOMIC_SMAX_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMAX_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SMAX_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMAX_X2_IMM_sem : InstSISemantic<S_ATOMIC_SMAX_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMAX_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SMAX_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMIN_IMM_sem : InstSISemantic<S_ATOMIC_SMIN_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMIN_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SMIN_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMIN_X2_IMM_sem : InstSISemantic<S_ATOMIC_SMIN_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMIN_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_SMIN_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMAX_IMM_sem : InstSISemantic<S_ATOMIC_UMAX_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMAX_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_UMAX_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMAX_X2_IMM_sem : InstSISemantic<S_ATOMIC_UMAX_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMAX_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_UMAX_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMIN_IMM_sem : InstSISemantic<S_ATOMIC_UMIN_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMIN_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_UMIN_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMIN_X2_IMM_sem : InstSISemantic<S_ATOMIC_UMIN_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMIN_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_UMIN_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

// S_ATOMIC_INC: tmp = MEM[addr]; MEM[addr] = (tmp >= data) ? 0 : tmp + 1.
def S_ATOMIC_INC_IMM_sem : InstSISemantic<S_ATOMIC_INC_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_INC_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_INC_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_INC_X2_IMM_sem : InstSISemantic<S_ATOMIC_INC_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i64_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_INC_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_INC_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i64_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

// S_ATOMIC_DEC: tmp = MEM[addr]; MEM[addr] = (tmp==0 || tmp>data) ? data : tmp-1.
def S_ATOMIC_DEC_IMM_sem : InstSISemantic<S_ATOMIC_DEC_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i32_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_DEC_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_DEC_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i32_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_DEC_X2_IMM_sem : InstSISemantic<S_ATOMIC_DEC_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i64_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_DEC_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_DEC_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i64_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_CMPSWAP_IMM_sem : InstSISemantic<S_ATOMIC_CMPSWAP_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETEQ, $tmp, $cmp), $src, $tmp)),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_CMPSWAP_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_CMPSWAP_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETEQ, $tmp, $cmp), $src, $tmp)),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_CMPSWAP_X2_IMM_sem : InstSISemantic<S_ATOMIC_CMPSWAP_X2_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETEQ, $tmp, $cmp), $src, $tmp)),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_CMPSWAP_X2_SGPR_IMM_sem : InstSISemantic<S_ATOMIC_CMPSWAP_X2_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $sdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETEQ, $tmp, $cmp), $src, $tmp)),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

//------------------------------------------------------------------------------
// Missing S_ATOMIC Flat Address Variants
// S_ATOMIC_*_SGPR: Uses only SGPR offset (no immediate).
// S_ATOMIC_*_RTN: Returns the original value before atomic operation.
// Sources (flat atomics introduced in GFX9/Vega):
//   vega-7nm-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 124.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 123.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 162.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 174.
//   (Not present in GCN3, RDNA1, RDNA2, RDNA3, RDNA4.)
//------------------------------------------------------------------------------

def S_ATOMIC_ADD_SGPR_sem : InstSISemantic<S_ATOMIC_ADD_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_ADD_IMM_RTN_sem : InstSISemantic<S_ATOMIC_ADD_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_ADD_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_ADD_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_ADD_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_ADD_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_ADD_X2_SGPR_sem : InstSISemantic<S_ATOMIC_ADD_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_ADD_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_ADD_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_ADD_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_ADD_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_ADD_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_ADD_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAdd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SUB_SGPR_sem : InstSISemantic<S_ATOMIC_SUB_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SUB_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SUB_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SUB_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SUB_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SUB_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SUB_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SUB_X2_SGPR_sem : InstSISemantic<S_ATOMIC_SUB_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SUB_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SUB_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SUB_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SUB_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SUB_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SUB_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWSub, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_AND_SGPR_sem : InstSISemantic<S_ATOMIC_AND_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_AND_IMM_RTN_sem : InstSISemantic<S_ATOMIC_AND_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_AND_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_AND_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_AND_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_AND_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_AND_X2_SGPR_sem : InstSISemantic<S_ATOMIC_AND_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_AND_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_AND_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_AND_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_AND_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_AND_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_AND_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWAnd, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_OR_SGPR_sem : InstSISemantic<S_ATOMIC_OR_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_OR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_OR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_OR_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_OR_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_OR_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_OR_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_OR_X2_SGPR_sem : InstSISemantic<S_ATOMIC_OR_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_OR_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_OR_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_OR_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_OR_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_OR_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_OR_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWOr, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_XOR_SGPR_sem : InstSISemantic<S_ATOMIC_XOR_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_XOR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_XOR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_XOR_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_XOR_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_XOR_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_XOR_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_XOR_X2_SGPR_sem : InstSISemantic<S_ATOMIC_XOR_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_XOR_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_XOR_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_XOR_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_XOR_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_XOR_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_XOR_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXor, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMIN_SGPR_sem : InstSISemantic<S_ATOMIC_SMIN_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMIN_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMIN_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMIN_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMIN_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMIN_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SMIN_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMIN_X2_SGPR_sem : InstSISemantic<S_ATOMIC_SMIN_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMIN_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMIN_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMIN_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMIN_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMIN_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SMIN_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMAX_SGPR_sem : InstSISemantic<S_ATOMIC_SMAX_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMAX_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMAX_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMAX_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMAX_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMAX_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SMAX_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMAX_X2_SGPR_sem : InstSISemantic<S_ATOMIC_SMAX_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SMAX_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMAX_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMAX_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SMAX_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SMAX_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SMAX_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMIN_SGPR_sem : InstSISemantic<S_ATOMIC_UMIN_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMIN_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMIN_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMIN_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMIN_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMIN_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_UMIN_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMIN_X2_SGPR_sem : InstSISemantic<S_ATOMIC_UMIN_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMIN_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMIN_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMIN_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMIN_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMIN_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_UMIN_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMin, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMAX_SGPR_sem : InstSISemantic<S_ATOMIC_UMAX_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMAX_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMAX_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMAX_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMAX_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMAX_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_UMAX_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMAX_X2_SGPR_sem : InstSISemantic<S_ATOMIC_UMAX_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_UMAX_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMAX_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMAX_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_UMAX_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_UMAX_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_UMAX_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWUMax, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SWAP_SGPR_sem : InstSISemantic<S_ATOMIC_SWAP_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SWAP_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SWAP_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SWAP_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SWAP_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SWAP_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SWAP_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SWAP_X2_SGPR_sem : InstSISemantic<S_ATOMIC_SWAP_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_SWAP_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SWAP_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SWAP_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_SWAP_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_SWAP_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_SWAP_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_CMPSWAP_SGPR_sem : InstSISemantic<S_ATOMIC_CMPSWAP_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_CMPSWAP_IMM_RTN_sem : InstSISemantic<S_ATOMIC_CMPSWAP_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_CMPSWAP_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_CMPSWAP_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_CMPSWAP_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_CMPSWAP_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_CMPSWAP_X2_SGPR_sem : InstSISemantic<S_ATOMIC_CMPSWAP_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata))
  ];
}

def S_ATOMIC_CMPSWAP_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_CMPSWAP_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_CMPSWAP_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_CMPSWAP_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_CMPSWAP_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_CMPSWAP_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMAtomicRMW AtomicRMWXchg, $tmp_addr, (GetNamedOperand $sdata)))
  ];
}

def S_ATOMIC_INC_SGPR_sem : InstSISemantic<S_ATOMIC_INC_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_INC_IMM_RTN_sem : InstSISemantic<S_ATOMIC_INC_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_INC_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_INC_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_INC_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_INC_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i32_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_INC_X2_SGPR_sem : InstSISemantic<S_ATOMIC_INC_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i64_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_INC_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_INC_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i64_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_INC_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_INC_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i64_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_INC_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_INC_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp_val, (GetNamedOperand $sdata)),
                                   (ConstantInt llvm_i64_ty, 0),
                                   (LLVMAdd $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_DEC_SGPR_sem : InstSISemantic<S_ATOMIC_DEC_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i32_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_DEC_IMM_RTN_sem : InstSISemantic<S_ATOMIC_DEC_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i32_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_DEC_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_DEC_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i32_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_DEC_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_DEC_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i32_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_DEC_X2_SGPR_sem : InstSISemantic<S_ATOMIC_DEC_X2_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i64_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr)
  ];
}

def S_ATOMIC_DEC_X2_IMM_RTN_sem : InstSISemantic<S_ATOMIC_DEC_X2_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i64_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_DEC_X2_SGPR_IMM_RTN_sem : InstSISemantic<S_ATOMIC_DEC_X2_SGPR_IMM_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i64_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

def S_ATOMIC_DEC_X2_SGPR_RTN_sem : InstSISemantic<S_ATOMIC_DEC_X2_SGPR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (DefVal $tmp_val, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp_val, (ConstantInt llvm_i64_ty, 0)),
                                           (LLVMCmp SETUGT, $tmp_val, (GetNamedOperand $sdata))),
                                   (GetNamedOperand $sdata),
                                   (LLVMSub $tmp_val, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $sdst, (GetVal $tmp_val))
  ];
}

//------------------------------------------------------------------------------
// S_LOAD Sub-word (byte/short) Instructions
//------------------------------------------------------------------------------

// S_LOAD_U8: Load 8 bits of unsigned data from the scalar memory,
// zero extend to 32 bits and store the result into a scalar register.
// SDATA.u32 = 32'U({ 24'0U, MEM[ADDR].u8 })
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 291 (opcode 9).
def S_LOAD_U8_IMM_sem : InstSISemantic<S_LOAD_U8_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMZExt (LLVMLoad llvm_i8_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_U8_SGPR_sem : InstSISemantic<S_LOAD_U8_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMZExt (LLVMLoad llvm_i8_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_U8_SGPR_IMM_sem : InstSISemantic<S_LOAD_U8_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMZExt (LLVMLoad llvm_i8_ty, $tmp_addr), llvm_i32_ty))
  ];
}

// S_LOAD_I8: Load 8 bits of signed data from the scalar memory,
// sign extend to 32 bits and store the result into a scalar register.
// SDATA.i32 = 32'I(signext(MEM[ADDR].i8))
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 291 (opcode 8).
def S_LOAD_I8_IMM_sem : InstSISemantic<S_LOAD_I8_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMSExt (LLVMLoad llvm_i8_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_I8_SGPR_sem : InstSISemantic<S_LOAD_I8_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMSExt (LLVMLoad llvm_i8_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_I8_SGPR_IMM_sem : InstSISemantic<S_LOAD_I8_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMSExt (LLVMLoad llvm_i8_ty, $tmp_addr), llvm_i32_ty))
  ];
}

// S_LOAD_U16: Load 16 bits of unsigned data from the scalar memory,
// zero extend to 32 bits and store the result into a scalar register.
// SDATA.u32 = 32'U({ 16'0U, MEM[ADDR].u16 })
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 292 (opcode 11).
def S_LOAD_U16_IMM_sem : InstSISemantic<S_LOAD_U16_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMZExt (LLVMLoad llvm_i16_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_U16_SGPR_sem : InstSISemantic<S_LOAD_U16_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMZExt (LLVMLoad llvm_i16_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_U16_SGPR_IMM_sem : InstSISemantic<S_LOAD_U16_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMZExt (LLVMLoad llvm_i16_ty, $tmp_addr), llvm_i32_ty))
  ];
}

// S_LOAD_I16: Load 16 bits of signed data from the scalar memory,
// sign extend to 32 bits and store the result into a scalar register.
// SDATA.i32 = 32'I(signext(MEM[ADDR].i16))
// Sources (RDNA4 only; not present in SI, CI, GCN3, Vega, Vega7nm,
//          CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2, or RDNA3):
//   rdna4-instruction-set-architecture.pdf, page 292 (opcode 10).
def S_LOAD_I16_IMM_sem : InstSISemantic<S_LOAD_I16_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $offset))),
    (SetNamedOperand $sdst, (LLVMSExt (LLVMLoad llvm_i16_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_I16_SGPR_sem : InstSISemantic<S_LOAD_I16_SGPR> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (GetNamedOperand $soffset))),
    (SetNamedOperand $sdst, (LLVMSExt (LLVMLoad llvm_i16_ty, $tmp_addr), llvm_i32_ty))
  ];
}

def S_LOAD_I16_SGPR_IMM_sem : InstSISemantic<S_LOAD_I16_SGPR_IMM> {
  let Semantic = [
    (DefVal $tmp_addr, (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $sbase), global_ptr_ty),
                               (LLVMAdd (GetNamedOperand $offset), (GetNamedOperand $soffset)))),
    (SetNamedOperand $sdst, (LLVMSExt (LLVMLoad llvm_i16_ty, $tmp_addr), llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// Cache Management Instructions
// Sources:
//   southern-islands-instruction-set-architecture.pdf, page 142.
//   sea-islands-instruction-set-architecture_0.pdf, page 144.
//   gcn3-instruction-set-architecture.pdf, page 157.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 124.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 121.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 119.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 154.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 166.
//   rdna-shader-instruction-set-architecture.pdf, page 139.
//   rdna2-shader-instruction-set-architecture.pdf, page 138.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 258.
//   rdna4-instruction-set-architecture.pdf, page 296.
//------------------------------------------------------------------------------

// S_DCACHE_INV: Invalidate data cache.
def S_DCACHE_INV_sem : InstSISemantic<S_DCACHE_INV>;

// S_DCACHE_WB: Writeback data cache (GFX8+).
def S_DCACHE_WB_sem : InstSISemantic<S_DCACHE_WB>;

// S_DCACHE_INV_VOL: Invalidate data cache volatile lines.
def S_DCACHE_INV_VOL_sem : InstSISemantic<S_DCACHE_INV_VOL>;

// S_DCACHE_WB_VOL: Writeback data cache volatile lines (GFX8+).
def S_DCACHE_WB_VOL_sem : InstSISemantic<S_DCACHE_WB_VOL>;

// S_GL1_INV: Invalidate GL1 cache (GFX10+).
def S_GL1_INV_sem : InstSISemantic<S_GL1_INV>;

// S_MEMTIME: Return 64-bit timestamp (deprecated in RDNA2+).
// Sources:
//   southern-islands-instruction-set-architecture.pdf, page 142.
//   sea-islands-instruction-set-architecture_0.pdf, page 146.
//   gcn3-instruction-set-architecture.pdf, page 162.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 124.
//   rdna-shader-instruction-set-architecture.pdf, page 139.
def S_MEMTIME_sem : InstSIIntrinsic<S_MEMTIME, int_amdgcn_s_memtime>;

// S_MEMREALTIME: Return 64-bit real timestamp (GFX8+, deprecated in RDNA2+).
def S_MEMREALTIME_sem : InstSIIntrinsic<S_MEMREALTIME, int_amdgcn_s_memrealtime>;

// S_GET_WAVEID_IN_WORKGROUP: Get wave ID within workgroup (GFX10+).
def S_GET_WAVEID_IN_WORKGROUP_sem : InstSIIntrinsic<S_GET_WAVEID_IN_WORKGROUP, 
                                                    int_amdgcn_s_get_waveid_in_workgroup>;

//------------------------------------------------------------------------------
// Prefetch Instructions (performance hints - no semantic effect)
// Sources (introduced in GFX12):
//   rdna4-instruction-set-architecture.pdf, page 296.
//------------------------------------------------------------------------------

def S_PREFETCH_DATA_sem : InstSISemantic<S_PREFETCH_DATA>;
def S_PREFETCH_DATA_PC_REL_sem : InstSISemantic<S_PREFETCH_DATA_PC_REL>;
def S_PREFETCH_INST_sem : InstSISemantic<S_PREFETCH_INST>;
def S_PREFETCH_INST_PC_REL_sem : InstSISemantic<S_PREFETCH_INST_PC_REL>;
def S_BUFFER_PREFETCH_DATA_sem : InstSISemantic<S_BUFFER_PREFETCH_DATA>;

//------------------------------------------------------------------------------
// Data Cache Discard Instructions (GFX12+)
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 296.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 166.
//------------------------------------------------------------------------------

def S_DCACHE_DISCARD_IMM_sem : InstSISemantic<S_DCACHE_DISCARD_IMM>;
def S_DCACHE_DISCARD_SGPR_sem : InstSISemantic<S_DCACHE_DISCARD_SGPR>;
def S_DCACHE_DISCARD_SGPR_IMM_sem : InstSISemantic<S_DCACHE_DISCARD_SGPR_IMM>;
def S_DCACHE_DISCARD_X2_IMM_sem : InstSISemantic<S_DCACHE_DISCARD_X2_IMM>;
def S_DCACHE_DISCARD_X2_SGPR_sem : InstSISemantic<S_DCACHE_DISCARD_X2_SGPR>;
def S_DCACHE_DISCARD_X2_SGPR_IMM_sem : InstSISemantic<S_DCACHE_DISCARD_X2_SGPR_IMM>;

//------------------------------------------------------------------------------
// ATC Probe Instructions (address translation cache probe, GFX8+)
// Hardware-specific operations for TLB management.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 124.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 123.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 162.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 174.
//------------------------------------------------------------------------------

def S_ATC_PROBE_IMM_sem : InstSISemantic<S_ATC_PROBE_IMM>;
def S_ATC_PROBE_SGPR_sem : InstSISemantic<S_ATC_PROBE_SGPR>;
def S_ATC_PROBE_SGPR_IMM_sem : InstSISemantic<S_ATC_PROBE_SGPR_IMM>;
def S_ATC_PROBE_SGPR_OPT_IMM_sem : InstSISemantic<S_ATC_PROBE_SGPR_OPT_IMM>;
def S_ATC_PROBE_BUFFER_IMM_sem : InstSISemantic<S_ATC_PROBE_BUFFER_IMM>;
def S_ATC_PROBE_BUFFER_SGPR_sem : InstSISemantic<S_ATC_PROBE_BUFFER_SGPR>;
def S_ATC_PROBE_BUFFER_SGPR_IMM_sem : InstSISemantic<S_ATC_PROBE_BUFFER_SGPR_IMM>;
def S_ATC_PROBE_BUFFER_SGPR_OPT_IMM_sem : InstSISemantic<S_ATC_PROBE_BUFFER_SGPR_OPT_IMM>;
