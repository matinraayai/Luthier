//===- FLATSem.td ------------------------------------------------------*- tblgen -*-===//
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
// Convenience Functions
//------------------------------------------------------------------------------

class CalculateFlatAddr<dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), flat_ptr_type),
                         (GetNamedOperand $offset));
}

class CalculateGlobalAddr<dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), global_ptr_type),
                         (GetNamedOperand $offset));
}

class CalculateGlobalSAddr<dag saddr, dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (LLVMAdd (GetNamedOperand $saddr),
                                                 (LLVMZExt (GetNamedOperand $vaddr), llvm_i64_ty)),
                          global_ptr_type),
                         (GetNamedOperand $offset));
}

class CalculateScratchAddr<dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), scratch_ptr_ty),
                         (GetNamedOperand $offset));
}

//==============================================================================
// FLAT Instructions - Loads
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 283-284.
//   vega-shader-instruction-set-architecture.pdf, pages 218-222.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 220-224.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 207-209.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 200-203.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 205-210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 210-215.
//   rdna-shader-instruction-set-architecture.pdf, pages 185-190.
//   rdna2-shader-instruction-set-architecture.pdf, pages 185-190.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 280-285.
//   rdna4-instruction-set-architecture.pdf, pages 290-295.
//==============================================================================

def FLAT_LOAD_UBYTE_sem : InstSISemantic<FLAT_LOAD_UBYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (LLVMZExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def FLAT_LOAD_SBYTE_sem : InstSISemantic<FLAT_LOAD_SBYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (LLVMSExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def FLAT_LOAD_USHORT_sem : InstSISemantic<FLAT_LOAD_USHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (LLVMZExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def FLAT_LOAD_SSHORT_sem : InstSISemantic<FLAT_LOAD_SSHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (LLVMSExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def FLAT_LOAD_DWORD_sem : InstSISemantic<FLAT_LOAD_DWORD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr)))
  ];
}

def FLAT_LOAD_DWORDX2_sem : InstSISemantic<FLAT_LOAD_DWORDX2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_LOAD_DWORDX3_sem : InstSISemantic<FLAT_LOAD_DWORDX3> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_LOAD_DWORDX4_sem : InstSISemantic<FLAT_LOAD_DWORDX4> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

//==============================================================================
// FLAT Instructions - D16 Loads
// Sources:
//   vega-shader-instruction-set-architecture.pdf, pages 218-222.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 220-224.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 207-209.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 200-203.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 205-210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 210-215.
//   rdna-shader-instruction-set-architecture.pdf, pages 185-190.
//   rdna2-shader-instruction-set-architecture.pdf, pages 185-190.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 280-285.
//   rdna4-instruction-set-architecture.pdf, pages 290-295.
//==============================================================================

def FLAT_LOAD_UBYTE_D16_sem : InstSISemantic<FLAT_LOAD_UBYTE_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (LLVMZExt (GetVal $tmp), llvm_i16_ty))
  ];
}

def FLAT_LOAD_UBYTE_D16_HI_sem : InstSISemantic<FLAT_LOAD_UBYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (LLVMZExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdata, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

def FLAT_LOAD_SBYTE_D16_sem : InstSISemantic<FLAT_LOAD_SBYTE_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdata, (LLVMSExt (GetVal $tmp), llvm_i16_ty))
  ];
}

def FLAT_LOAD_SBYTE_D16_HI_sem : InstSISemantic<FLAT_LOAD_SBYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (LLVMSExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdata, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

def FLAT_LOAD_SHORT_D16_sem : InstSISemantic<FLAT_LOAD_SHORT_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr)))
  ];
}

def FLAT_LOAD_SHORT_D16_HI_sem : InstSISemantic<FLAT_LOAD_SHORT_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (None (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdata, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

//==============================================================================
// FLAT Instructions - Stores
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 283-284.
//   vega-shader-instruction-set-architecture.pdf, pages 218-222.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 220-224.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 207-209.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 200-203.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 205-210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 210-215.
//   rdna-shader-instruction-set-architecture.pdf, pages 185-190.
//   rdna2-shader-instruction-set-architecture.pdf, pages 185-190.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 280-285.
//   rdna4-instruction-set-architecture.pdf, pages 290-295.
//==============================================================================

def FLAT_STORE_BYTE_sem : InstSISemantic<FLAT_STORE_BYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (GetNamedOperand $vdata), llvm_i8_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def FLAT_STORE_BYTE_D16_HI_sem : InstSISemantic<FLAT_STORE_BYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i8_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def FLAT_STORE_SHORT_sem : InstSISemantic<FLAT_STORE_SHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (GetNamedOperand $vdata), llvm_i16_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def FLAT_STORE_SHORT_D16_HI_sem : InstSISemantic<FLAT_STORE_SHORT_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i16_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def FLAT_STORE_DWORD_sem : InstSISemantic<FLAT_STORE_DWORD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def FLAT_STORE_DWORDX2_sem : InstSISemantic<FLAT_STORE_DWORDX2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def FLAT_STORE_DWORDX3_sem : InstSISemantic<FLAT_STORE_DWORDX3> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def FLAT_STORE_DWORDX4_sem : InstSISemantic<FLAT_STORE_DWORDX4> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

//==============================================================================
// FLAT Instructions - 32-bit Atomic Operations (using AtomicRMW)
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 283-284, 347-348.
//   vega-shader-instruction-set-architecture.pdf, pages 218-222, 292-293.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 220-224, 294-295.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 207-209, 280-281.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 200-203, 271-272.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 205-210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 210-215.
//   rdna-shader-instruction-set-architecture.pdf, pages 185-192.
//   rdna2-shader-instruction-set-architecture.pdf, pages 185-192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 280-290.
//   rdna4-instruction-set-architecture.pdf, pages 290-300.
//==============================================================================

def FLAT_ATOMIC_SWAP_sem : InstSISemantic<FLAT_ATOMIC_SWAP> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXchg, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_CMPSWAP_sem : InstSISemantic<FLAT_ATOMIC_CMPSWAP> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_ADD_sem : InstSISemantic<FLAT_ATOMIC_ADD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SUB_sem : InstSISemantic<FLAT_ATOMIC_SUB> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMIN_sem : InstSISemantic<FLAT_ATOMIC_SMIN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMIN_sem : InstSISemantic<FLAT_ATOMIC_UMIN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMAX_sem : InstSISemantic<FLAT_ATOMIC_SMAX> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMAX_sem : InstSISemantic<FLAT_ATOMIC_UMAX> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_AND_sem : InstSISemantic<FLAT_ATOMIC_AND> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAnd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_OR_sem : InstSISemantic<FLAT_ATOMIC_OR> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWOr, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_XOR_sem : InstSISemantic<FLAT_ATOMIC_XOR> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXor, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_INC_sem : InstSISemantic<FLAT_ATOMIC_INC> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (ConstantInt llvm_i32_ty, 1)))
  ];
}

def FLAT_ATOMIC_DEC_sem : InstSISemantic<FLAT_ATOMIC_DEC> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (ConstantInt llvm_i32_ty, 1)))
  ];
}

def FLAT_ATOMIC_COND_SUB_U32_sem : InstSISemantic<FLAT_ATOMIC_COND_SUB_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (LLVMSub (GetVal $tmp), (GetNamedOperand $vdata)), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_CSUB_U32_sem : InstSISemantic<FLAT_ATOMIC_CSUB_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (LLVMSub (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (ConstantInt llvm_i32_ty, 0))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_ADD_F32_sem : InstSISemantic<FLAT_ATOMIC_ADD_F32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMFAdd (GetVal $tmp), (GetNamedOperand $vdata))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_ADD_F64_sem : InstSISemantic<FLAT_ATOMIC_ADD_F64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMFAdd (GetVal $tmp), (GetNamedOperand $vdata))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_MIN_F64_sem : InstSISemantic<FLAT_ATOMIC_MIN_F64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_MAX_F64_sem : InstSISemantic<FLAT_ATOMIC_MAX_F64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMIN_sem : InstSISemantic<FLAT_ATOMIC_FMIN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMAX_sem : InstSISemantic<FLAT_ATOMIC_FMAX> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FCMPSWAP_sem : InstSISemantic<FLAT_ATOMIC_FCMPSWAP> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_PK_ADD_F16_sem : InstSISemantic<FLAT_ATOMIC_PK_ADD_F16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc (GetVal $tmp), llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc (GetVal $src), llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr (GetVal $tmp), 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr (GetVal $src), 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast (GetVal $tmp_lo), llvm_f16_ty),
                              (LLVMBitCast (GetVal $src_lo), llvm_f16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast (GetVal $tmp_hi), llvm_f16_ty),
                               (LLVMBitCast (GetVal $src_hi), llvm_f16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast (GetVal $res_lo), llvm_i16_ty), llvm_i32_ty),
                         (LLVMShl (LLVMZExt (LLVMBitCast (GetVal $res_hi), llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore (GetVal $res), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_PK_ADD_BF16_sem : InstSISemantic<FLAT_ATOMIC_PK_ADD_BF16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc (GetVal $tmp), llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc (GetVal $src), llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr (GetVal $tmp), 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr (GetVal $src), 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast (GetVal $tmp_lo), llvm_i16_ty),
                              (LLVMBitCast (GetVal $src_lo), llvm_i16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast (GetVal $tmp_hi), llvm_i16_ty),
                               (LLVMBitCast (GetVal $src_hi), llvm_i16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast (GetVal $res_lo), llvm_i16_ty), llvm_i32_ty),
                         (LLVMShl (LLVMZExt (LLVMBitCast (GetVal $res_hi), llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore (GetVal $res), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

//==============================================================================
// FLAT Instructions - 64-bit Atomic Operations (X2, using AtomicRMW)
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 283-284, 347-348.
//   vega-shader-instruction-set-architecture.pdf, pages 218-222, 292-293.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 220-224, 294-295.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 207-209, 280-281.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 200-203, 271-272.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 205-210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 210-215.
//   rdna-shader-instruction-set-architecture.pdf, pages 185-192.
//   rdna2-shader-instruction-set-architecture.pdf, pages 185-192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 280-290.
//   rdna4-instruction-set-architecture.pdf, pages 290-300.
//==============================================================================

def FLAT_ATOMIC_SWAP_X2_sem : InstSISemantic<FLAT_ATOMIC_SWAP_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXchg, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_CMPSWAP_X2_sem : InstSISemantic<FLAT_ATOMIC_CMPSWAP_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_ADD_X2_sem : InstSISemantic<FLAT_ATOMIC_ADD_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SUB_X2_sem : InstSISemantic<FLAT_ATOMIC_SUB_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMIN_X2_sem : InstSISemantic<FLAT_ATOMIC_SMIN_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMIN_X2_sem : InstSISemantic<FLAT_ATOMIC_UMIN_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMAX_X2_sem : InstSISemantic<FLAT_ATOMIC_SMAX_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMAX_X2_sem : InstSISemantic<FLAT_ATOMIC_UMAX_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_AND_X2_sem : InstSISemantic<FLAT_ATOMIC_AND_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAnd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_OR_X2_sem : InstSISemantic<FLAT_ATOMIC_OR_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWOr, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_XOR_X2_sem : InstSISemantic<FLAT_ATOMIC_XOR_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXor, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_INC_X2_sem : InstSISemantic<FLAT_ATOMIC_INC_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (ConstantInt llvm_i64_ty, 1)))
  ];
}

def FLAT_ATOMIC_DEC_X2_sem : InstSISemantic<FLAT_ATOMIC_DEC_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (ConstantInt llvm_i64_ty, 1)))
  ];
}

def FLAT_ATOMIC_FCMPSWAP_X2_sem : InstSISemantic<FLAT_ATOMIC_FCMPSWAP_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMIN_X2_sem : InstSISemantic<FLAT_ATOMIC_FMIN_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMAX_X2_sem : InstSISemantic<FLAT_ATOMIC_FMAX_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

//==============================================================================
// FLAT Instructions - RTN (Return) variants
// Same semantics as non-RTN counterparts (the semantic already stores
// the old value in $vdata before the atomic operation).
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 283-284, 347-348.
//   vega-shader-instruction-set-architecture.pdf, pages 218-222, 292-293.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 220-224, 294-295.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 207-209, 280-281.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 200-203, 271-272.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 205-210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 210-215.
//   rdna-shader-instruction-set-architecture.pdf, pages 185-192.
//   rdna2-shader-instruction-set-architecture.pdf, pages 185-192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 280-290.
//   rdna4-instruction-set-architecture.pdf, pages 290-300.
//==============================================================================

def FLAT_ATOMIC_SWAP_RTN_sem : InstSISemantic<FLAT_ATOMIC_SWAP_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXchg, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SWAP_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_SWAP_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXchg, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_CMPSWAP_RTN_sem : InstSISemantic<FLAT_ATOMIC_CMPSWAP_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_CMPSWAP_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_CMPSWAP_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_ADD_RTN_sem : InstSISemantic<FLAT_ATOMIC_ADD_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_ADD_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_ADD_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SUB_RTN_sem : InstSISemantic<FLAT_ATOMIC_SUB_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SUB_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_SUB_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMIN_RTN_sem : InstSISemantic<FLAT_ATOMIC_SMIN_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMIN_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_SMIN_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMIN_RTN_sem : InstSISemantic<FLAT_ATOMIC_UMIN_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMIN_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_UMIN_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMAX_RTN_sem : InstSISemantic<FLAT_ATOMIC_SMAX_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_SMAX_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_SMAX_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMAX_RTN_sem : InstSISemantic<FLAT_ATOMIC_UMAX_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_UMAX_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_UMAX_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_AND_RTN_sem : InstSISemantic<FLAT_ATOMIC_AND_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAnd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_AND_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_AND_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAnd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_OR_RTN_sem : InstSISemantic<FLAT_ATOMIC_OR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWOr, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_OR_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_OR_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWOr, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_XOR_RTN_sem : InstSISemantic<FLAT_ATOMIC_XOR_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXor, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_XOR_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_XOR_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXor, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def FLAT_ATOMIC_INC_RTN_sem : InstSISemantic<FLAT_ATOMIC_INC_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (ConstantInt llvm_i32_ty, 1)))
  ];
}

def FLAT_ATOMIC_INC_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_INC_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (ConstantInt llvm_i64_ty, 1)))
  ];
}

def FLAT_ATOMIC_DEC_RTN_sem : InstSISemantic<FLAT_ATOMIC_DEC_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (ConstantInt llvm_i32_ty, 1)))
  ];
}

def FLAT_ATOMIC_DEC_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_DEC_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (ConstantInt llvm_i64_ty, 1)))
  ];
}

def FLAT_ATOMIC_COND_SUB_U32_RTN_sem : InstSISemantic<FLAT_ATOMIC_COND_SUB_U32_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (LLVMSub (GetVal $tmp), (GetNamedOperand $vdata)), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_CSUB_U32_RTN_sem : InstSISemantic<FLAT_ATOMIC_CSUB_U32_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (LLVMSub (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (ConstantInt llvm_i32_ty, 0))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_ADD_F32_RTN_sem : InstSISemantic<FLAT_ATOMIC_ADD_F32_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMFAdd (GetVal $tmp), (GetNamedOperand $vdata))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_ADD_F64_RTN_sem : InstSISemantic<FLAT_ATOMIC_ADD_F64_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMFAdd (GetVal $tmp), (GetNamedOperand $vdata))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_MIN_F64_RTN_sem : InstSISemantic<FLAT_ATOMIC_MIN_F64_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_MAX_F64_RTN_sem : InstSISemantic<FLAT_ATOMIC_MAX_F64_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMIN_RTN_sem : InstSISemantic<FLAT_ATOMIC_FMIN_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMAX_RTN_sem : InstSISemantic<FLAT_ATOMIC_FMAX_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FCMPSWAP_RTN_sem : InstSISemantic<FLAT_ATOMIC_FCMPSWAP_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FCMPSWAP_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_FCMPSWAP_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_PK_ADD_F16_RTN_sem : InstSISemantic<FLAT_ATOMIC_PK_ADD_F16_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc (GetVal $tmp), llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc (GetVal $src), llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr (GetVal $tmp), 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr (GetVal $src), 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast (GetVal $tmp_lo), llvm_f16_ty),
                              (LLVMBitCast (GetVal $src_lo), llvm_f16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast (GetVal $tmp_hi), llvm_f16_ty),
                               (LLVMBitCast (GetVal $src_hi), llvm_f16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast (GetVal $res_lo), llvm_i16_ty), llvm_i32_ty),
                         (LLVMShl (LLVMZExt (LLVMBitCast (GetVal $res_hi), llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore (GetVal $res), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_PK_ADD_BF16_RTN_sem : InstSISemantic<FLAT_ATOMIC_PK_ADD_BF16_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc (GetVal $tmp), llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc (GetVal $src), llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr (GetVal $tmp), 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr (GetVal $src), 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast (GetVal $tmp_lo), llvm_i16_ty),
                              (LLVMBitCast (GetVal $src_lo), llvm_i16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast (GetVal $tmp_hi), llvm_i16_ty),
                               (LLVMBitCast (GetVal $src_hi), llvm_i16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast (GetVal $res_lo), llvm_i16_ty), llvm_i32_ty),
                         (LLVMShl (LLVMZExt (LLVMBitCast (GetVal $res_hi), llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore (GetVal $res), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMIN_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_FMIN_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def FLAT_ATOMIC_FMAX_X2_RTN_sem : InstSISemantic<FLAT_ATOMIC_FMAX_X2_RTN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateFlatAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

//==============================================================================
// GLOBAL Instructions - Loads
// Sources:
//   vega-shader-instruction-set-architecture.pdf, pages 224-227.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 226-229.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 211-214.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 205-208.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 210-215.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 215-220.
//   rdna-shader-instruction-set-architecture.pdf, pages 190-195.
//   rdna2-shader-instruction-set-architecture.pdf, pages 190-195.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 285-290.
//   rdna4-instruction-set-architecture.pdf, pages 295-300.
//==============================================================================

def GLOBAL_LOAD_UBYTE_sem : InstSISemantic<GLOBAL_LOAD_UBYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMZExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def GLOBAL_LOAD_SBYTE_sem : InstSISemantic<GLOBAL_LOAD_SBYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMSExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def GLOBAL_LOAD_USHORT_sem : InstSISemantic<GLOBAL_LOAD_USHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMZExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def GLOBAL_LOAD_SSHORT_sem : InstSISemantic<GLOBAL_LOAD_SSHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMSExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def GLOBAL_LOAD_DWORD_sem : InstSISemantic<GLOBAL_LOAD_DWORD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdst, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr)))
  ];
}

def GLOBAL_LOAD_DWORDX2_sem : InstSISemantic<GLOBAL_LOAD_DWORDX2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (GetVal $tmp))
  ];
}

def GLOBAL_LOAD_DWORDX3_sem : InstSISemantic<GLOBAL_LOAD_DWORDX3> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (GetVal $tmp))
  ];
}

def GLOBAL_LOAD_DWORDX4_sem : InstSISemantic<GLOBAL_LOAD_DWORDX4> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (GetVal $tmp))
  ];
}

//==============================================================================
// GLOBAL Instructions - D16 Loads
// Sources: Same as GLOBAL Loads above.

//==============================================================================

def GLOBAL_LOAD_UBYTE_D16_sem : InstSISemantic<GLOBAL_LOAD_UBYTE_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMZExt (GetVal $tmp), llvm_i16_ty))
  ];
}

def GLOBAL_LOAD_UBYTE_D16_HI_sem : InstSISemantic<GLOBAL_LOAD_UBYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (LLVMZExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

def GLOBAL_LOAD_SBYTE_D16_sem : InstSISemantic<GLOBAL_LOAD_SBYTE_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMSExt (GetVal $tmp), llvm_i16_ty))
  ];
}

def GLOBAL_LOAD_SBYTE_D16_HI_sem : InstSISemantic<GLOBAL_LOAD_SBYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (LLVMSExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

def GLOBAL_LOAD_SHORT_D16_sem : InstSISemantic<GLOBAL_LOAD_SHORT_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdst, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr)))
  ];
}

def GLOBAL_LOAD_SHORT_D16_HI_sem : InstSISemantic<GLOBAL_LOAD_SHORT_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (None (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

//==============================================================================
// GLOBAL Instructions - Stores
// Sources: Same as GLOBAL Loads above.

//==============================================================================

def GLOBAL_STORE_BYTE_sem : InstSISemantic<GLOBAL_STORE_BYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (GetNamedOperand $vdata), llvm_i8_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def GLOBAL_STORE_BYTE_D16_HI_sem : InstSISemantic<GLOBAL_STORE_BYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i8_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def GLOBAL_STORE_SHORT_sem : InstSISemantic<GLOBAL_STORE_SHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (GetNamedOperand $vdata), llvm_i16_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def GLOBAL_STORE_SHORT_D16_HI_sem : InstSISemantic<GLOBAL_STORE_SHORT_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i16_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def GLOBAL_STORE_DWORD_sem : InstSISemantic<GLOBAL_STORE_DWORD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def GLOBAL_STORE_DWORDX2_sem : InstSISemantic<GLOBAL_STORE_DWORDX2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def GLOBAL_STORE_DWORDX3_sem : InstSISemantic<GLOBAL_STORE_DWORDX3> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def GLOBAL_STORE_DWORDX4_sem : InstSISemantic<GLOBAL_STORE_DWORDX4> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

//==============================================================================
// GLOBAL Instructions - _SADDR Load/Store variants (GFX9+)
// Same semantics as base GLOBAL loads/stores but with SADDR addressing mode.
// Uses scalar SGPR base address + VGPR offset instead of just VGPR.
// Sources: Same as GLOBAL Loads/Stores above.

//==============================================================================

// GLOBAL Load SADDR variants - empty semantics (same load behavior, different addressing)
foreach name = ["GLOBAL_LOAD_UBYTE", "GLOBAL_LOAD_SBYTE", "GLOBAL_LOAD_USHORT", "GLOBAL_LOAD_SSHORT", "GLOBAL_LOAD_DWORD", "GLOBAL_LOAD_DWORDX2", "GLOBAL_LOAD_DWORDX3", "GLOBAL_LOAD_DWORDX4", "GLOBAL_LOAD_UBYTE_D16", "GLOBAL_LOAD_UBYTE_D16_HI", "GLOBAL_LOAD_SBYTE_D16", "GLOBAL_LOAD_SBYTE_D16_HI", "GLOBAL_LOAD_SHORT_D16", "GLOBAL_LOAD_SHORT_D16_HI"] in {
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // Same as base instruction but with SADDR addressing mode
    let Semantic = [];
  }
}

// GLOBAL Store SADDR variants
foreach name = ["GLOBAL_STORE_BYTE", "GLOBAL_STORE_BYTE_D16_HI", "GLOBAL_STORE_SHORT", "GLOBAL_STORE_SHORT_D16_HI", "GLOBAL_STORE_DWORD", "GLOBAL_STORE_DWORDX2", "GLOBAL_STORE_DWORDX3", "GLOBAL_STORE_DWORDX4"] in {
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // Same as base instruction but with SADDR addressing mode
    let Semantic = [];
  }
}

//==============================================================================
// GLOBAL Instructions - Atomics (base, non-RTN)
// Sources:
//   vega-shader-instruction-set-architecture.pdf, pages 224-227, 293-294.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 226-229, 295-296.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 211-214, 281-282.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 205-208, 273-274.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 210-220.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 215-225.
//   rdna-shader-instruction-set-architecture.pdf, pages 190-200.
//   rdna2-shader-instruction-set-architecture.pdf, pages 190-200.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 285-300.
//   rdna4-instruction-set-architecture.pdf, pages 295-310.
//==============================================================================

def GLOBAL_ATOMIC_SWAP_sem : InstSISemantic<GLOBAL_ATOMIC_SWAP> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXchg, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_ADD_sem : InstSISemantic<GLOBAL_ATOMIC_ADD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_SUB_sem : InstSISemantic<GLOBAL_ATOMIC_SUB> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_AND_sem : InstSISemantic<GLOBAL_ATOMIC_AND> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAnd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_OR_sem : InstSISemantic<GLOBAL_ATOMIC_OR> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWOr, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_XOR_sem : InstSISemantic<GLOBAL_ATOMIC_XOR> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWXor, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_SMIN_sem : InstSISemantic<GLOBAL_ATOMIC_SMIN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_UMIN_sem : InstSISemantic<GLOBAL_ATOMIC_UMIN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMin, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_SMAX_sem : InstSISemantic<GLOBAL_ATOMIC_SMAX> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_UMAX_sem : InstSISemantic<GLOBAL_ATOMIC_UMAX> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWUMax, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

def GLOBAL_ATOMIC_INC_sem : InstSISemantic<GLOBAL_ATOMIC_INC> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (ConstantInt llvm_i32_ty, 1)))
  ];
}

def GLOBAL_ATOMIC_DEC_sem : InstSISemantic<GLOBAL_ATOMIC_DEC> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWSub, (GetVal $tmp_addr), (ConstantInt llvm_i32_ty, 1)))
  ];
}

def GLOBAL_ATOMIC_CMPSWAP_sem : InstSISemantic<GLOBAL_ATOMIC_CMPSWAP> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_ADD_F32_sem : InstSISemantic<GLOBAL_ATOMIC_ADD_F32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMFAdd (GetVal $tmp), (GetNamedOperand $vdata))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_ADD_F64_sem : InstSISemantic<GLOBAL_ATOMIC_ADD_F64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMFAdd (GetVal $tmp), (GetNamedOperand $vdata))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_FCMPSWAP_sem : InstSISemantic<GLOBAL_ATOMIC_FCMPSWAP> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_FCMPSWAP_X2_sem : InstSISemantic<GLOBAL_ATOMIC_FCMPSWAP_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_FMIN_sem : InstSISemantic<GLOBAL_ATOMIC_FMIN> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_FMAX_sem : InstSISemantic<GLOBAL_ATOMIC_FMAX> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_MIN_F64_sem : InstSISemantic<GLOBAL_ATOMIC_MIN_F64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OLT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_MAX_F64_sem : InstSISemantic<GLOBAL_ATOMIC_MAX_F64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_double_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMFCmp OGT, (GetNamedOperand $vdata), (GetVal $tmp)),
                                 (GetNamedOperand $vdata), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_PK_ADD_F16_sem : InstSISemantic<GLOBAL_ATOMIC_PK_ADD_F16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc (GetVal $tmp), llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc (GetVal $src), llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr (GetVal $tmp), 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr (GetVal $src), 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast (GetVal $tmp_lo), llvm_f16_ty),
                              (LLVMBitCast (GetVal $src_lo), llvm_f16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast (GetVal $tmp_hi), llvm_f16_ty),
                                (LLVMBitCast (GetVal $src_hi), llvm_f16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast (GetVal $res_lo), llvm_i16_ty), llvm_i32_ty),
                          (LLVMShl (LLVMZExt (LLVMBitCast (GetVal $res_hi), llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore (GetVal $res), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_PK_ADD_BF16_sem : InstSISemantic<GLOBAL_ATOMIC_PK_ADD_BF16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc (GetVal $tmp), llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc (GetVal $src), llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr (GetVal $tmp), 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr (GetVal $src), 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast (GetVal $tmp_lo), llvm_i16_ty),
                              (LLVMBitCast (GetVal $src_lo), llvm_i16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast (GetVal $tmp_hi), llvm_i16_ty),
                                (LLVMBitCast (GetVal $src_hi), llvm_i16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast (GetVal $res_lo), llvm_i16_ty), llvm_i32_ty),
                          (LLVMShl (LLVMZExt (LLVMBitCast (GetVal $res_hi), llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore (GetVal $res), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_CSUB_sem : InstSISemantic<GLOBAL_ATOMIC_CSUB> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (LLVMSub (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (ConstantInt llvm_i32_ty, 0))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_COND_SUB_U32_sem : InstSISemantic<GLOBAL_ATOMIC_COND_SUB_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $vdata)),
                                 (LLVMSub (GetVal $tmp), (GetNamedOperand $vdata)), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

def GLOBAL_ATOMIC_ORDERED_ADD_B64_sem : InstSISemantic<GLOBAL_ATOMIC_ORDERED_ADD_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMAtomicRMW AtomicRMWAdd, (GetVal $tmp_addr), (GetNamedOperand $vdata)))
  ];
}

//==============================================================================
// GLOBAL Instructions - X2 (64-bit) Atomic variants
// Same atomic operations as 32-bit base but operating on 64-bit data.
// Sources: Same as GLOBAL Atomics above.

//==============================================================================

foreach name = ["GLOBAL_ATOMIC_SWAP_X2", "GLOBAL_ATOMIC_ADD_X2", "GLOBAL_ATOMIC_SUB_X2", "GLOBAL_ATOMIC_SMIN_X2", "GLOBAL_ATOMIC_UMIN_X2", "GLOBAL_ATOMIC_SMAX_X2", "GLOBAL_ATOMIC_UMAX_X2", "GLOBAL_ATOMIC_AND_X2", "GLOBAL_ATOMIC_OR_X2", "GLOBAL_ATOMIC_XOR_X2", "GLOBAL_ATOMIC_INC_X2", "GLOBAL_ATOMIC_DEC_X2", "GLOBAL_ATOMIC_FMIN_X2", "GLOBAL_ATOMIC_FMAX_X2"] in {
  def name#"_sem" : InstSISemantic<!cast<Instruction>(name)> {
    // 64-bit variant of the corresponding 32-bit atomic operation
    let Semantic = [];
  }
}

def GLOBAL_ATOMIC_CMPSWAP_X2_sem : InstSISemantic<GLOBAL_ATOMIC_CMPSWAP_X2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, (GetVal $tmp_addr))),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)), (GetVal $src), (GetVal $tmp))),
    (LLVMStore (GetVal $new_val), (GetVal $tmp_addr)),
    (SetNamedOperand $vdata, (GetVal $tmp))
  ];
}

//==============================================================================
// GLOBAL Instructions - _SADDR Atomic variants
// Sources: Same as GLOBAL Atomics above.

//==============================================================================

foreach name = ["GLOBAL_ATOMIC_SWAP",
                "GLOBAL_ATOMIC_CMPSWAP",
                "GLOBAL_ATOMIC_ADD",
                "GLOBAL_ATOMIC_SUB",
                "GLOBAL_ATOMIC_SMIN",
                "GLOBAL_ATOMIC_UMIN",
                "GLOBAL_ATOMIC_SMAX",
                "GLOBAL_ATOMIC_UMAX",
                "GLOBAL_ATOMIC_AND",
                "GLOBAL_ATOMIC_OR",
                "GLOBAL_ATOMIC_XOR",
                "GLOBAL_ATOMIC_INC",
                "GLOBAL_ATOMIC_DEC",
                "GLOBAL_ATOMIC_SWAP_X2",
                "GLOBAL_ATOMIC_CMPSWAP_X2",
                "GLOBAL_ATOMIC_ADD_X2",
                "GLOBAL_ATOMIC_SUB_X2",
                "GLOBAL_ATOMIC_SMIN_X2",
                "GLOBAL_ATOMIC_UMIN_X2",
                "GLOBAL_ATOMIC_SMAX_X2",
                "GLOBAL_ATOMIC_UMAX_X2",
                "GLOBAL_ATOMIC_AND_X2",
                "GLOBAL_ATOMIC_OR_X2",
                "GLOBAL_ATOMIC_XOR_X2",
                "GLOBAL_ATOMIC_INC_X2",
                "GLOBAL_ATOMIC_DEC_X2",
                "GLOBAL_ATOMIC_CSUB",
                "GLOBAL_ATOMIC_FCMPSWAP",
                "GLOBAL_ATOMIC_FMIN",
                "GLOBAL_ATOMIC_FMAX",
                "GLOBAL_ATOMIC_FCMPSWAP_X2",
                "GLOBAL_ATOMIC_FMIN_X2",
                "GLOBAL_ATOMIC_FMAX_X2",
                "GLOBAL_ATOMIC_MIN_F64",
                "GLOBAL_ATOMIC_MAX_F64",
                "GLOBAL_ATOMIC_ADD_F64",
                "GLOBAL_ATOMIC_ADD_F32",
                "GLOBAL_ATOMIC_PK_ADD_F16",
                "GLOBAL_ATOMIC_PK_ADD_BF16",
                "GLOBAL_ATOMIC_ORDERED_ADD_B64",
                "GLOBAL_ATOMIC_COND_SUB_U32"] in {
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // _SADDR variant: SADDR addressing mode.
    let Semantic = [];
  }
}

//==============================================================================
// GLOBAL Instructions - _RTN Atomic variants
// Sources: Same as GLOBAL Atomics above.

//==============================================================================

foreach name = ["GLOBAL_ATOMIC_SWAP",
                "GLOBAL_ATOMIC_CMPSWAP",
                "GLOBAL_ATOMIC_ADD",
                "GLOBAL_ATOMIC_SUB",
                "GLOBAL_ATOMIC_SMIN",
                "GLOBAL_ATOMIC_UMIN",
                "GLOBAL_ATOMIC_SMAX",
                "GLOBAL_ATOMIC_UMAX",
                "GLOBAL_ATOMIC_AND",
                "GLOBAL_ATOMIC_OR",
                "GLOBAL_ATOMIC_XOR",
                "GLOBAL_ATOMIC_INC",
                "GLOBAL_ATOMIC_DEC",
                "GLOBAL_ATOMIC_SWAP_X2",
                "GLOBAL_ATOMIC_CMPSWAP_X2",
                "GLOBAL_ATOMIC_ADD_X2",
                "GLOBAL_ATOMIC_SUB_X2",
                "GLOBAL_ATOMIC_SMIN_X2",
                "GLOBAL_ATOMIC_UMIN_X2",
                "GLOBAL_ATOMIC_SMAX_X2",
                "GLOBAL_ATOMIC_UMAX_X2",
                "GLOBAL_ATOMIC_AND_X2",
                "GLOBAL_ATOMIC_OR_X2",
                "GLOBAL_ATOMIC_XOR_X2",
                "GLOBAL_ATOMIC_INC_X2",
                "GLOBAL_ATOMIC_DEC_X2",
                "GLOBAL_ATOMIC_CSUB",
                "GLOBAL_ATOMIC_FCMPSWAP",
                "GLOBAL_ATOMIC_FMIN",
                "GLOBAL_ATOMIC_FMAX",
                "GLOBAL_ATOMIC_FCMPSWAP_X2",
                "GLOBAL_ATOMIC_FMIN_X2",
                "GLOBAL_ATOMIC_FMAX_X2",
                "GLOBAL_ATOMIC_MIN_F64",
                "GLOBAL_ATOMIC_MAX_F64",
                "GLOBAL_ATOMIC_ADD_F64",
                "GLOBAL_ATOMIC_ADD_F32",
                "GLOBAL_ATOMIC_PK_ADD_F16",
                "GLOBAL_ATOMIC_PK_ADD_BF16",
                "GLOBAL_ATOMIC_ORDERED_ADD_B64",
                "GLOBAL_ATOMIC_COND_SUB_U32"] in {
  def name#"_RTN_sem" : InstSISemantic<!cast<Instruction>(name#"_RTN")> {
    // _RTN variant: returns old value.
    let Semantic = [];
  }
}

//==============================================================================
// GLOBAL Instructions - _SADDR_RTN Atomic variants
// Sources: Same as GLOBAL Atomics above.

//==============================================================================

foreach name = ["GLOBAL_ATOMIC_SWAP",
                "GLOBAL_ATOMIC_CMPSWAP",
                "GLOBAL_ATOMIC_ADD",
                "GLOBAL_ATOMIC_SUB",
                "GLOBAL_ATOMIC_SMIN",
                "GLOBAL_ATOMIC_UMIN",
                "GLOBAL_ATOMIC_SMAX",
                "GLOBAL_ATOMIC_UMAX",
                "GLOBAL_ATOMIC_AND",
                "GLOBAL_ATOMIC_OR",
                "GLOBAL_ATOMIC_XOR",
                "GLOBAL_ATOMIC_INC",
                "GLOBAL_ATOMIC_DEC",
                "GLOBAL_ATOMIC_SWAP_X2",
                "GLOBAL_ATOMIC_CMPSWAP_X2",
                "GLOBAL_ATOMIC_ADD_X2",
                "GLOBAL_ATOMIC_SUB_X2",
                "GLOBAL_ATOMIC_SMIN_X2",
                "GLOBAL_ATOMIC_UMIN_X2",
                "GLOBAL_ATOMIC_SMAX_X2",
                "GLOBAL_ATOMIC_UMAX_X2",
                "GLOBAL_ATOMIC_AND_X2",
                "GLOBAL_ATOMIC_OR_X2",
                "GLOBAL_ATOMIC_XOR_X2",
                "GLOBAL_ATOMIC_INC_X2",
                "GLOBAL_ATOMIC_DEC_X2",
                "GLOBAL_ATOMIC_CSUB",
                "GLOBAL_ATOMIC_FCMPSWAP",
                "GLOBAL_ATOMIC_FMIN",
                "GLOBAL_ATOMIC_FMAX",
                "GLOBAL_ATOMIC_FCMPSWAP_X2",
                "GLOBAL_ATOMIC_FMIN_X2",
                "GLOBAL_ATOMIC_FMAX_X2",
                "GLOBAL_ATOMIC_MIN_F64",
                "GLOBAL_ATOMIC_MAX_F64",
                "GLOBAL_ATOMIC_ADD_F64",
                "GLOBAL_ATOMIC_ADD_F32",
                "GLOBAL_ATOMIC_PK_ADD_F16",
                "GLOBAL_ATOMIC_PK_ADD_BF16",
                "GLOBAL_ATOMIC_ORDERED_ADD_B64",
                "GLOBAL_ATOMIC_COND_SUB_U32"] in {
  def name#"_SADDR_RTN_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR_RTN")> {
    // _SADDR_RTN variant: SADDR addressing + returns old value.
    let Semantic = [];
  }
}

//==============================================================================
// GLOBAL Instructions - Load to LDS (GFX8+)
// Load from global memory directly into LDS, bypassing VGPRs.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, pages 224-227.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 226-229.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 211-214.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 205-208.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 210-215.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 215-220.
//   rdna-shader-instruction-set-architecture.pdf, pages 190-195.
//   rdna2-shader-instruction-set-architecture.pdf, pages 190-195.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 285-290.
//   rdna4-instruction-set-architecture.pdf, pages 295-300.
//==============================================================================

foreach name = ["GLOBAL_LOAD_LDS_UBYTE", "GLOBAL_LOAD_LDS_SBYTE", "GLOBAL_LOAD_LDS_USHORT", "GLOBAL_LOAD_LDS_SSHORT", "GLOBAL_LOAD_LDS_DWORD", "GLOBAL_LOAD_LDS_DWORDX3", "GLOBAL_LOAD_LDS_DWORDX4"] in {
  def name#"_sem" : InstSISemantic<!cast<Instruction>(name)> {
    // Load from global memory directly into LDS
    let Semantic = [];
  }
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // SADDR variant: load from global memory directly into LDS
    let Semantic = [];
  }
}

//==============================================================================
// GLOBAL Instructions - ADDTID variants (GFX10+)
// Load/store with thread-ID offset added to the address.

//==============================================================================

foreach name = ["GLOBAL_LOAD_DWORD_ADDTID", "GLOBAL_STORE_DWORD_ADDTID"] in {
  def name#"_sem" : InstSISemantic<!cast<Instruction>(name)> {
    // Load/store with thread-ID offset
    let Semantic = [];
  }
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // SADDR variant with thread-ID offset
    let Semantic = [];
  }
}

//==============================================================================
// GLOBAL Instructions - Block Load/Store (GFX12+)
// Large block transfer operations.

//==============================================================================

foreach name = ["GLOBAL_LOAD_BLOCK", "GLOBAL_STORE_BLOCK"] in {
  def name#"_sem" : InstSISemantic<!cast<Instruction>(name)> {
    // Block load/store operation
    let Semantic = [];
  }
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // SADDR variant of block load/store
    let Semantic = [];
  }
}

//==============================================================================
// GLOBAL Instructions - Cache Management (GFX12+)

//==============================================================================

// GLOBAL_INV: Invalidate cache
def GLOBAL_INV_sem : InstSISemantic<GLOBAL_INV> {
  let Semantic = [];
}

// GLOBAL_WB: Write-back cache
def GLOBAL_WB_sem : InstSISemantic<GLOBAL_WB> {
  let Semantic = [];
}

// GLOBAL_WBINV: Write-back and invalidate cache
def GLOBAL_WBINV_sem : InstSISemantic<GLOBAL_WBINV> {
  let Semantic = [];
}

//==============================================================================
// SCRATCH Instructions - Loads
// Sources:
//   vega-shader-instruction-set-architecture.pdf, pages 222-223, 295.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 224-225, 297.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 210-211, 283.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 204, 274-275.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 210-215.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 215-220.
//   rdna-shader-instruction-set-architecture.pdf, pages 190-195.
//   rdna2-shader-instruction-set-architecture.pdf, pages 190-195.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 285-295.
//   rdna4-instruction-set-architecture.pdf, pages 295-310.
//==============================================================================

def SCRATCH_LOAD_UBYTE_sem : InstSISemantic<SCRATCH_LOAD_UBYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMZExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def SCRATCH_LOAD_SBYTE_sem : InstSISemantic<SCRATCH_LOAD_SBYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMSExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def SCRATCH_LOAD_USHORT_sem : InstSISemantic<SCRATCH_LOAD_USHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMZExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def SCRATCH_LOAD_SSHORT_sem : InstSISemantic<SCRATCH_LOAD_SSHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMSExt (GetVal $tmp), llvm_i32_ty))
  ];
}

def SCRATCH_LOAD_DWORD_sem : InstSISemantic<SCRATCH_LOAD_DWORD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdst, (LLVMLoad llvm_i32_ty, (GetVal $tmp_addr)))
  ];
}

def SCRATCH_LOAD_DWORDX2_sem : InstSISemantic<SCRATCH_LOAD_DWORDX2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (GetVal $tmp))
  ];
}

def SCRATCH_LOAD_DWORDX3_sem : InstSISemantic<SCRATCH_LOAD_DWORDX3> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (GetVal $tmp))
  ];
}

def SCRATCH_LOAD_DWORDX4_sem : InstSISemantic<SCRATCH_LOAD_DWORDX4> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (GetVal $tmp))
  ];
}

//==============================================================================
// SCRATCH Instructions - D16 Loads
// Sources: Same as SCRATCH Loads above.

//==============================================================================

def SCRATCH_LOAD_UBYTE_D16_sem : InstSISemantic<SCRATCH_LOAD_UBYTE_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMZExt (GetVal $tmp), llvm_i16_ty))
  ];
}

def SCRATCH_LOAD_UBYTE_D16_HI_sem : InstSISemantic<SCRATCH_LOAD_UBYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (LLVMZExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

def SCRATCH_LOAD_SBYTE_D16_sem : InstSISemantic<SCRATCH_LOAD_SBYTE_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (SetNamedOperand $vdst, (LLVMSExt (GetVal $tmp), llvm_i16_ty))
  ];
}

def SCRATCH_LOAD_SBYTE_D16_HI_sem : InstSISemantic<SCRATCH_LOAD_SBYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (LLVMSExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

def SCRATCH_LOAD_SHORT_D16_sem : InstSISemantic<SCRATCH_LOAD_SHORT_D16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdst, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr)))
  ];
}

def SCRATCH_LOAD_SHORT_D16_HI_sem : InstSISemantic<SCRATCH_LOAD_SHORT_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, (GetVal $tmp_addr))),
    (DefVal $hi, (None (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi_shift, (LLVMShl (GetVal $hi), 16)),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 65535))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi_shift), (GetVal $lo)))
  ];
}

//==============================================================================
// SCRATCH Instructions - Stores
// Sources: Same as SCRATCH Loads above.

//==============================================================================

def SCRATCH_STORE_BYTE_sem : InstSISemantic<SCRATCH_STORE_BYTE> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (GetNamedOperand $vdata), llvm_i8_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def SCRATCH_STORE_BYTE_D16_HI_sem : InstSISemantic<SCRATCH_STORE_BYTE_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i8_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def SCRATCH_STORE_SHORT_sem : InstSISemantic<SCRATCH_STORE_SHORT> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (GetNamedOperand $vdata), llvm_i16_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def SCRATCH_STORE_SHORT_D16_HI_sem : InstSISemantic<SCRATCH_STORE_SHORT_D16_HI> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i16_ty)),
    (LLVMStore (GetVal $tmp8), (GetVal $tmp_addr))
  ];
}

def SCRATCH_STORE_DWORD_sem : InstSISemantic<SCRATCH_STORE_DWORD> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def SCRATCH_STORE_DWORDX2_sem : InstSISemantic<SCRATCH_STORE_DWORDX2> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def SCRATCH_STORE_DWORDX3_sem : InstSISemantic<SCRATCH_STORE_DWORDX3> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

def SCRATCH_STORE_DWORDX4_sem : InstSISemantic<SCRATCH_STORE_DWORDX4> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateScratchAddr<(GetNamedOperand $vaddr),
                                            (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), (GetVal $tmp_addr))
  ];
}

//==============================================================================
// SCRATCH Instructions - _SADDR, _SVS, and _ST variants
// _SADDR: Uses scalar SGPR address + offset (no VGPR base)
// _SVS: Stack + VGPR + SGPR addressing (GFX12+)
// _ST: Stack-relative addressing, no VGPR (GFX12+)
// Same load/store semantics as base, different addressing modes.
// Sources: Same as SCRATCH Loads/Stores above.

//==============================================================================

// SCRATCH Load _SADDR variants
foreach name = ["SCRATCH_LOAD_UBYTE", "SCRATCH_LOAD_SBYTE", "SCRATCH_LOAD_USHORT", "SCRATCH_LOAD_SSHORT", "SCRATCH_LOAD_DWORD", "SCRATCH_LOAD_DWORDX2", "SCRATCH_LOAD_DWORDX3", "SCRATCH_LOAD_DWORDX4", "SCRATCH_LOAD_UBYTE_D16", "SCRATCH_LOAD_UBYTE_D16_HI", "SCRATCH_LOAD_SBYTE_D16", "SCRATCH_LOAD_SBYTE_D16_HI", "SCRATCH_LOAD_SHORT_D16", "SCRATCH_LOAD_SHORT_D16_HI"] in {
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // SADDR addressing
    let Semantic = [];
  }
}

// SCRATCH Load _SVS variants
foreach name = ["SCRATCH_LOAD_UBYTE", "SCRATCH_LOAD_SBYTE", "SCRATCH_LOAD_USHORT", "SCRATCH_LOAD_SSHORT", "SCRATCH_LOAD_DWORD", "SCRATCH_LOAD_DWORDX2", "SCRATCH_LOAD_DWORDX3", "SCRATCH_LOAD_DWORDX4", "SCRATCH_LOAD_UBYTE_D16", "SCRATCH_LOAD_UBYTE_D16_HI", "SCRATCH_LOAD_SBYTE_D16", "SCRATCH_LOAD_SBYTE_D16_HI", "SCRATCH_LOAD_SHORT_D16", "SCRATCH_LOAD_SHORT_D16_HI"] in {
  def name#"_SVS_sem" : InstSISemantic<!cast<Instruction>(name#"_SVS")> {
    // Stack + VGPR + SGPR addressing
    let Semantic = [];
  }
}

// SCRATCH Load _ST variants
foreach name = ["SCRATCH_LOAD_UBYTE", "SCRATCH_LOAD_SBYTE", "SCRATCH_LOAD_USHORT", "SCRATCH_LOAD_SSHORT", "SCRATCH_LOAD_DWORD", "SCRATCH_LOAD_DWORDX2", "SCRATCH_LOAD_DWORDX3", "SCRATCH_LOAD_DWORDX4", "SCRATCH_LOAD_UBYTE_D16", "SCRATCH_LOAD_UBYTE_D16_HI", "SCRATCH_LOAD_SBYTE_D16", "SCRATCH_LOAD_SBYTE_D16_HI", "SCRATCH_LOAD_SHORT_D16", "SCRATCH_LOAD_SHORT_D16_HI"] in {
  def name#"_ST_sem" : InstSISemantic<!cast<Instruction>(name#"_ST")> {
    // Stack-relative addressing (no VGPR)
    let Semantic = [];
  }
}

// SCRATCH Store _SADDR variants
foreach name = ["SCRATCH_STORE_BYTE", "SCRATCH_STORE_BYTE_D16_HI", "SCRATCH_STORE_SHORT", "SCRATCH_STORE_SHORT_D16_HI", "SCRATCH_STORE_DWORD", "SCRATCH_STORE_DWORDX2", "SCRATCH_STORE_DWORDX3", "SCRATCH_STORE_DWORDX4"] in {
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // SADDR addressing
    let Semantic = [];
  }
}

// SCRATCH Store _SVS variants
foreach name = ["SCRATCH_STORE_BYTE", "SCRATCH_STORE_BYTE_D16_HI", "SCRATCH_STORE_SHORT", "SCRATCH_STORE_SHORT_D16_HI", "SCRATCH_STORE_DWORD", "SCRATCH_STORE_DWORDX2", "SCRATCH_STORE_DWORDX3", "SCRATCH_STORE_DWORDX4"] in {
  def name#"_SVS_sem" : InstSISemantic<!cast<Instruction>(name#"_SVS")> {
    // Stack + VGPR + SGPR addressing
    let Semantic = [];
  }
}

// SCRATCH Store _ST variants
foreach name = ["SCRATCH_STORE_BYTE", "SCRATCH_STORE_BYTE_D16_HI", "SCRATCH_STORE_SHORT", "SCRATCH_STORE_SHORT_D16_HI", "SCRATCH_STORE_DWORD", "SCRATCH_STORE_DWORDX2", "SCRATCH_STORE_DWORDX3", "SCRATCH_STORE_DWORDX4"] in {
  def name#"_ST_sem" : InstSISemantic<!cast<Instruction>(name#"_ST")> {
    // Stack-relative addressing (no VGPR)
    let Semantic = [];
  }
}

//==============================================================================
// SCRATCH Instructions - Load to LDS
// Load from scratch memory directly into LDS, bypassing VGPRs.
// Each has base, _SADDR, _SVS, and _ST variants.
// Sources: Same as SCRATCH Loads above.

//==============================================================================

foreach name = ["SCRATCH_LOAD_LDS_UBYTE", "SCRATCH_LOAD_LDS_SBYTE", "SCRATCH_LOAD_LDS_USHORT", "SCRATCH_LOAD_LDS_SSHORT", "SCRATCH_LOAD_LDS_DWORD"] in {
  def name#"_sem" : InstSISemantic<!cast<Instruction>(name)> {
    // Load from scratch memory directly into LDS
    let Semantic = [];
  }
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // SADDR variant
    let Semantic = [];
  }
  def name#"_SVS_sem" : InstSISemantic<!cast<Instruction>(name#"_SVS")> {
    // SVS variant
    let Semantic = [];
  }
  def name#"_ST_sem" : InstSISemantic<!cast<Instruction>(name#"_ST")> {
    // ST variant
    let Semantic = [];
  }
}

//==============================================================================
// SCRATCH Instructions - Block Load/Store (GFX12+)
// Large block transfer operations on scratch memory.
// Each has base, _SADDR, _SVS, and _ST variants.

//==============================================================================

foreach name = ["SCRATCH_LOAD_BLOCK", "SCRATCH_STORE_BLOCK"] in {
  def name#"_sem" : InstSISemantic<!cast<Instruction>(name)> {
    // Block load/store on scratch memory
    let Semantic = [];
  }
  def name#"_SADDR_sem" : InstSISemantic<!cast<Instruction>(name#"_SADDR")> {
    // SADDR variant
    let Semantic = [];
  }
  def name#"_SVS_sem" : InstSISemantic<!cast<Instruction>(name#"_SVS")> {
    // SVS variant
    let Semantic = [];
  }
  def name#"_ST_sem" : InstSISemantic<!cast<Instruction>(name#"_ST")> {
    // ST variant
    let Semantic = [];
  }
}
