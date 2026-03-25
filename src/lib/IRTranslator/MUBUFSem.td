//===- MUBUFSem.td ---------------------------------------------*- tblgen -*-===//
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

class CalculateBufferAddr<dag vaddr, dag rsrc, dag soffset, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), buffer_ptr_type),
                         (LLVMAdd (GetNamedOperand $soffset), (GetNamedOperand $offset))));
}

class CalculateBufferAddrWithIdx<dag vaddr, dag rsrc, dag soffset, dag offset,
                                  LLVMQualPointerType bufferPtrTy> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), bufferPtrTy),
                         (LLVMAdd (GetNamedOperand $soffset), (GetNamedOperand $offset))));
}

//------------------------------------------------------------------------------
// MUBUF Instructions - Buffer Load/Store (opcodes 0-39)
//------------------------------------------------------------------------------
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, Ch.8 (Vector Memory Buffer Instructions) & Sec 12.14 (MUBUF Instructions).
//   vega-shader-instruction-set-architecture.pdf, Sec 8.1 (p.50-62) & Sec 12.14 (MUBUF Instructions, p.198+).
//   vega-7nm-shader-instruction-set-architecture.pdf, Sec 8.1 (p.51-63) & Sec 12.14 (MUBUF Instructions, p.200+).
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, Sec 9.1 (p.55-67) & Sec 12.14 (MUBUF Instructions, p.188+).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, Sec 9.1 (p.55-67) & Sec 12.13 (MUBUF Instructions, p.182+).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, Sec 9.1 (p.63-72) & Sec 12.13 (MUBUF Instructions, p.450+).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, Sec 9.1 (p.74-83) & Sec 12.13 (MUBUF Instructions, p.495+).
//   rdna-shader-instruction-set-architecture.pdf, Sec 8.1 (p.56-67) & Sec 12.14 (MUBUF Instructions, p.216+).
//   rdna2-shader-instruction-set-architecture.pdf, Sec 8.1 (p.55-66) & Sec 12.14 (MUBUF Instructions, p.209+).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, Ch.9 (p.80-107) & Sec 16.16 (MUBUF Instructions, p.538+).
//   rdna4-instruction-set-architecture.pdf, Ch.9 (p.102-129) & Sec 16.16 (VBUFFER Instructions, p.616+).

def BUFFER_LOAD_FORMAT_X_sem : InstSISemantic<BUFFER_LOAD_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def BUFFER_LOAD_FORMAT_XY_sem : InstSISemantic<BUFFER_LOAD_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_LOAD_FORMAT_XYZ_sem : InstSISemantic<BUFFER_LOAD_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_LOAD_FORMAT_XYZW_sem : InstSISemantic<BUFFER_LOAD_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_STORE_FORMAT_X_sem : InstSISemantic<BUFFER_STORE_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_STORE_FORMAT_XY_sem : InstSISemantic<BUFFER_STORE_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_STORE_FORMAT_XYZ_sem : InstSISemantic<BUFFER_STORE_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_STORE_FORMAT_XYZW_sem : InstSISemantic<BUFFER_STORE_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_LOAD_D16_FORMAT_X_sem : InstSISemantic<BUFFER_LOAD_D16_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def BUFFER_LOAD_D16_FORMAT_XY_sem : InstSISemantic<BUFFER_LOAD_D16_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v2i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_v2i32_ty))
  ];
}

def BUFFER_LOAD_D16_FORMAT_XYZ_sem : InstSISemantic<BUFFER_LOAD_D16_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_v3i32_ty))
  ];
}

def BUFFER_LOAD_D16_FORMAT_XYZW_sem : InstSISemantic<BUFFER_LOAD_D16_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_v4i32_ty))
  ];
}

def BUFFER_STORE_D16_FORMAT_X_sem : InstSISemantic<BUFFER_STORE_D16_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMTrunc (GetNamedOperand $vdata), llvm_i16_ty)),
    (LLVMStore $tmp, $tmp_addr)
  ];
}

def BUFFER_STORE_D16_FORMAT_XY_sem : InstSISemantic<BUFFER_STORE_D16_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (GetNamedOperand $vdata), llvm_v2i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

def BUFFER_STORE_D16_FORMAT_XYZ_sem : InstSISemantic<BUFFER_STORE_D16_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (GetNamedOperand $vdata), llvm_v3i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

def BUFFER_STORE_D16_FORMAT_XYZW_sem : InstSISemantic<BUFFER_STORE_D16_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (GetNamedOperand $vdata), llvm_v4i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

def BUFFER_LOAD_U8_sem : InstSISemantic<BUFFER_LOAD_U8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def BUFFER_LOAD_I8_sem : InstSISemantic<BUFFER_LOAD_I8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

def BUFFER_LOAD_U16_sem : InstSISemantic<BUFFER_LOAD_U16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def BUFFER_LOAD_I16_sem : InstSISemantic<BUFFER_LOAD_I16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMSExt $tmp, llvm_i32_ty))
  ];
}

def BUFFER_LOAD_B32_sem : InstSISemantic<BUFFER_LOAD_B32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def BUFFER_LOAD_B64_sem : InstSISemantic<BUFFER_LOAD_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_LOAD_B96_sem : InstSISemantic<BUFFER_LOAD_B96> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_LOAD_B128_sem : InstSISemantic<BUFFER_LOAD_B128> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_STORE_B8_sem : InstSISemantic<BUFFER_STORE_B8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMTrunc (GetNamedOperand $vdata), llvm_i8_ty)),
    (LLVMStore $tmp, $tmp_addr)
  ];
}

def BUFFER_STORE_B16_sem : InstSISemantic<BUFFER_STORE_B16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMTrunc (GetNamedOperand $vdata), llvm_i16_ty)),
    (LLVMStore $tmp, $tmp_addr)
  ];
}

def BUFFER_STORE_B32_sem : InstSISemantic<BUFFER_STORE_B32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_STORE_B64_sem : InstSISemantic<BUFFER_STORE_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_STORE_B96_sem : InstSISemantic<BUFFER_STORE_B96> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_STORE_B128_sem : InstSISemantic<BUFFER_STORE_B128> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def BUFFER_LOAD_D16_U8_sem : InstSISemantic<BUFFER_LOAD_D16_U8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_i16_ty))
  ];
}

def BUFFER_LOAD_D16_I8_sem : InstSISemantic<BUFFER_LOAD_D16_I8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMSExt $tmp, llvm_i16_ty))
  ];
}

def BUFFER_LOAD_D16_B16_sem : InstSISemantic<BUFFER_LOAD_D16_B16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMLoad llvm_i16_ty, $tmp_addr))
  ];
}

def BUFFER_LOAD_D16_HI_U8_sem : InstSISemantic<BUFFER_LOAD_D16_HI_U8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, $tmp_addr)),
    (DefVal $hi16, (LLVMZExt $tmp, llvm_i16_ty)),
    (DefVal $vdata_hi, (LLVMShl $hi16, 16)),
    (DefVal $vdata_lo, (LLVMAnd (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdata, (LLVMOr $vdata_hi, $vdata_lo))
  ];
}

def BUFFER_LOAD_D16_HI_I8_sem : InstSISemantic<BUFFER_LOAD_D16_HI_I8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, $tmp_addr)),
    (DefVal $hi16, (LLVMSExt $tmp, llvm_i16_ty)),
    (DefVal $vdata_hi, (LLVMShl $hi16, 16)),
    (DefVal $vdata_lo, (LLVMAnd (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdata, (LLVMOr $vdata_hi, $vdata_lo))
  ];
}

def BUFFER_LOAD_D16_HI_B16_sem : InstSISemantic<BUFFER_LOAD_D16_HI_B16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $hi16, (LLVMLoad llvm_i16_ty, $tmp_addr)),
    (DefVal $vdata_hi, (LLVMShl $hi16, 16)),
    (DefVal $vdata_lo, (LLVMAnd (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdata, (LLVMOr $vdata_hi, $vdata_lo))
  ];
}

def BUFFER_STORE_D16_HI_B8_sem : InstSISemantic<BUFFER_STORE_D16_HI_B8> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp8, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i8_ty)),
    (LLVMStore $tmp8, $tmp_addr)
  ];
}

def BUFFER_STORE_D16_HI_B16_sem : InstSISemantic<BUFFER_STORE_D16_HI_B16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

def BUFFER_LOAD_D16_HI_FORMAT_X_sem : InstSISemantic<BUFFER_LOAD_D16_HI_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, $tmp_addr)),
    (DefVal $vdata_hi, (LLVMZExt $tmp, llvm_i32_ty)),
    (DefVal $vdata_shift, (LLVMShl $vdata_hi, 16)),
    (DefVal $vdata_lo, (LLVMAnd (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdata, (LLVMOr $vdata_shift, $vdata_lo))
  ];
}

def BUFFER_STORE_D16_HI_FORMAT_X_sem : InstSISemantic<BUFFER_STORE_D16_HI_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (LLVMShr (GetNamedOperand $vdata), 16), llvm_i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

//------------------------------------------------------------------------------
// MUBUF Instructions - Buffer Atomic Operations (opcodes 51-77)
//------------------------------------------------------------------------------
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, Ch.8 (Vector Memory Buffer Instructions) & Sec 12.14 (MUBUF Instructions).
//   vega-shader-instruction-set-architecture.pdf, Sec 8.1 (p.50-62) & Sec 12.14 (MUBUF Instructions, p.198+).
//   vega-7nm-shader-instruction-set-architecture.pdf, Sec 8.1 (p.51-63) & Sec 12.14 (MUBUF Instructions, p.200+).
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, Sec 9.1 (p.55-67) & Sec 12.14 (MUBUF Instructions, p.188+).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, Sec 9.1 (p.55-67) & Sec 12.13 (MUBUF Instructions, p.182+).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, Sec 9.1 (p.63-72) & Sec 12.13 (MUBUF Instructions, p.450+).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, Sec 9.1 (p.74-83) & Sec 12.13 (MUBUF Instructions, p.495+).
//   rdna-shader-instruction-set-architecture.pdf, Sec 8.1 (p.56-67) & Sec 12.14 (MUBUF Instructions, p.216+).
//   rdna2-shader-instruction-set-architecture.pdf, Sec 8.1 (p.55-66) & Sec 12.14 (MUBUF Instructions, p.209+).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, Ch.9 (p.80-107) & Sec 16.16 (MUBUF Instructions, p.538+).
//   rdna4-instruction-set-architecture.pdf, Ch.9 (p.102-129) & Sec 16.16 (VBUFFER Instructions, p.616+).

def BUFFER_ATOMIC_SWAP_B32_sem : InstSISemantic<BUFFER_ATOMIC_SWAP_B32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_CMPSWAP_B32_sem : InstSISemantic<BUFFER_ATOMIC_CMPSWAP_B32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i32_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETEQ, $tmp, $cmp), $src, $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_ADD_U32_sem : InstSISemantic<BUFFER_ATOMIC_ADD_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMAdd $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_SUB_U32_sem : InstSISemantic<BUFFER_ATOMIC_SUB_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSub $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_SUB_CLAMP_U32_sem : InstSISemantic<BUFFER_ATOMIC_SUB_CLAMP_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETULT, $tmp, (GetNamedOperand $vdata)),
                                 (ConstantInt llvm_i32_ty, 0),
                                 (LLVMSub $tmp, (GetNamedOperand $vdata)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MIN_I32_sem : InstSISemantic<BUFFER_ATOMIC_MIN_I32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETLT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MIN_U32_sem : InstSISemantic<BUFFER_ATOMIC_MIN_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETULT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MAX_I32_sem : InstSISemantic<BUFFER_ATOMIC_MAX_I32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETGT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MAX_U32_sem : InstSISemantic<BUFFER_ATOMIC_MAX_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_AND_B32_sem : InstSISemantic<BUFFER_ATOMIC_AND_B32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMAnd $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_OR_B32_sem : InstSISemantic<BUFFER_ATOMIC_OR_B32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMOr $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_XOR_B32_sem : InstSISemantic<BUFFER_ATOMIC_XOR_B32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMXor $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_INC_U32_sem : InstSISemantic<BUFFER_ATOMIC_INC_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp, (GetNamedOperand $vdata)),
                                 (ConstantInt llvm_i32_ty, 0),
                                 (LLVMAdd $tmp, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_DEC_U32_sem : InstSISemantic<BUFFER_ATOMIC_DEC_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp, (ConstantInt llvm_i32_ty, 0)),
                                      (LLVMCmp SETUGT, $tmp, (GetNamedOperand $vdata))),
                                 (GetNamedOperand $vdata),
                                 (LLVMSub $tmp, (ConstantInt llvm_i32_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_SWAP_B64_sem : InstSISemantic<BUFFER_ATOMIC_SWAP_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_CMPSWAP_B64_sem : InstSISemantic<BUFFER_ATOMIC_CMPSWAP_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 1))),
    (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata), (ConstantInt llvm_i64_ty, 0))),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETEQ, $tmp, $cmp), $src, $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_ADD_U64_sem : InstSISemantic<BUFFER_ATOMIC_ADD_U64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMAdd $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_SUB_U64_sem : InstSISemantic<BUFFER_ATOMIC_SUB_U64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSub $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MIN_I64_sem : InstSISemantic<BUFFER_ATOMIC_MIN_I64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETLT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MIN_U64_sem : InstSISemantic<BUFFER_ATOMIC_MIN_U64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETULT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MAX_I64_sem : InstSISemantic<BUFFER_ATOMIC_MAX_I64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETGT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_MAX_U64_sem : InstSISemantic<BUFFER_ATOMIC_MAX_U64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGT, (GetNamedOperand $vdata), $tmp),
                                 (GetNamedOperand $vdata), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_AND_B64_sem : InstSISemantic<BUFFER_ATOMIC_AND_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMAnd $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_OR_B64_sem : InstSISemantic<BUFFER_ATOMIC_OR_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMOr $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_XOR_B64_sem : InstSISemantic<BUFFER_ATOMIC_XOR_B64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMXor $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_INC_U64_sem : InstSISemantic<BUFFER_ATOMIC_INC_U64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp, (GetNamedOperand $vdata)),
                                 (ConstantInt llvm_i64_ty, 0),
                                 (LLVMAdd $tmp, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_DEC_U64_sem : InstSISemantic<BUFFER_ATOMIC_DEC_U64> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMOr (LLVMCmp SETEQ, $tmp, (ConstantInt llvm_i64_ty, 0)),
                                      (LLVMCmp SETUGT, $tmp, (GetNamedOperand $vdata))),
                                 (GetNamedOperand $vdata),
                                 (LLVMSub $tmp, (ConstantInt llvm_i64_ty, 1)))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_COND_SUB_U32_sem : InstSISemantic<BUFFER_ATOMIC_COND_SUB_U32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMSelect (LLVMCmp SETUGE, $tmp, (GetNamedOperand $vdata)),
                                 (LLVMSub $tmp, (GetNamedOperand $vdata)), $tmp)),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_ADD_F32_sem : InstSISemantic<BUFFER_ATOMIC_ADD_F32> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_float_ty, $tmp_addr)),
    (DefVal $new_val, (LLVMFAdd $tmp, (GetNamedOperand $vdata))),
    (LLVMStore $new_val, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_PK_ADD_F16_sem : InstSISemantic<BUFFER_ATOMIC_PK_ADD_F16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc $tmp, llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc $src, llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr $tmp, 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr $src, 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast $tmp_lo, llvm_f16_ty), 
                              (LLVMBitCast $src_lo, llvm_f16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast $tmp_hi, llvm_f16_ty),
                               (LLVMBitCast $src_hi, llvm_f16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast $res_lo, llvm_i16_ty), llvm_i32_ty),
                         (LLVMShl (LLVMZExt (LLVMBitCast $res_hi, llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore $res, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def BUFFER_ATOMIC_PK_ADD_BF16_sem : InstSISemantic<BUFFER_ATOMIC_PK_ADD_BF16> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, $tmp_addr)),
    (DefVal $src, (GetNamedOperand $vdata)),
    (DefVal $tmp_lo, (LLVMTrunc $tmp, llvm_i16_ty)),
    (DefVal $src_lo, (LLVMTrunc $src, llvm_i16_ty)),
    (DefVal $tmp_hi, (LLVMTrunc (LLVMShr $tmp, 16), llvm_i16_ty)),
    (DefVal $src_hi, (LLVMTrunc (LLVMShr $src, 16), llvm_i16_ty)),
    (DefVal $res_lo, (LLVMFAdd (LLVMBitCast $tmp_lo, llvm_i16_ty), 
                              (LLVMBitCast $src_lo, llvm_i16_ty))),
    (DefVal $res_hi, (LLVMFAdd (LLVMBitCast $tmp_hi, llvm_i16_ty),
                               (LLVMBitCast $src_hi, llvm_i16_ty))),
    (DefVal $res, (LLVMOr (LLVMZExt (LLVMBitCast $res_lo, llvm_i16_ty), llvm_i32_ty),
                         (LLVMShl (LLVMZExt (LLVMBitCast $res_hi, llvm_i16_ty), llvm_i32_ty), 16))),
    (LLVMStore $res, $tmp_addr),
    (SetNamedOperand $vdata, $tmp)
  ];
}

//------------------------------------------------------------------------------
// TBUFFER Instructions (opcodes 128+)
//------------------------------------------------------------------------------
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, Ch.8 & Sec 12.15 (MTBUF Instructions).
//   vega-shader-instruction-set-architecture.pdf, Sec 8.1 & Sec 12.15 (MTBUF Instructions).
//   vega-7nm-shader-instruction-set-architecture.pdf, Sec 8.1 & Sec 12.15 (MTBUF Instructions).
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, Sec 9.1 & Sec 12.15 (MTBUF Instructions).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, Sec 9.1 & Sec 12.14 (MTBUF Instructions).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, Sec 9.1 & Sec 12.14 (MTBUF Instructions).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, Sec 9.1 & Sec 12.14 (MTBUF Instructions).
//   rdna-shader-instruction-set-architecture.pdf, Sec 8.1 & Sec 12.15 (MTBUF Instructions).
//   rdna2-shader-instruction-set-architecture.pdf, Sec 8.1 & Sec 12.15 (MTBUF Instructions).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, Ch.9 & Sec 16.17 (MTBUF Instructions).
//   rdna4-instruction-set-architecture.pdf, Ch.9 & Sec 16.17 (MTBUF Instructions).

def TBUFFER_LOAD_FORMAT_X_sem : InstSISemantic<TBUFFER_LOAD_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (SetNamedOperand $vdata, (LLVMLoad llvm_i32_ty, $tmp_addr))
  ];
}

def TBUFFER_LOAD_FORMAT_XY_sem : InstSISemantic<TBUFFER_LOAD_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v2i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def TBUFFER_LOAD_FORMAT_XYZ_sem : InstSISemantic<TBUFFER_LOAD_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def TBUFFER_LOAD_FORMAT_XYZW_sem : InstSISemantic<TBUFFER_LOAD_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i32_ty, $tmp_addr)),
    (SetNamedOperand $vdata, $tmp)
  ];
}

def TBUFFER_STORE_FORMAT_X_sem : InstSISemantic<TBUFFER_STORE_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def TBUFFER_STORE_FORMAT_XY_sem : InstSISemantic<TBUFFER_STORE_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def TBUFFER_STORE_FORMAT_XYZ_sem : InstSISemantic<TBUFFER_STORE_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def TBUFFER_STORE_FORMAT_XYZW_sem : InstSISemantic<TBUFFER_STORE_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $vdata), $tmp_addr)
  ];
}

def TBUFFER_LOAD_D16_FORMAT_X_sem : InstSISemantic<TBUFFER_LOAD_D16_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_i32_ty))
  ];
}

def TBUFFER_LOAD_D16_FORMAT_XY_sem : InstSISemantic<TBUFFER_LOAD_D16_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v2i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_v2i32_ty))
  ];
}

def TBUFFER_LOAD_D16_FORMAT_XYZ_sem : InstSISemantic<TBUFFER_LOAD_D16_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v3i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_v3i32_ty))
  ];
}

def TBUFFER_LOAD_D16_FORMAT_XYZW_sem : InstSISemantic<TBUFFER_LOAD_D16_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMLoad llvm_v4i16_ty, $tmp_addr)),
    (SetNamedOperand $vdata, (LLVMZExt $tmp, llvm_v4i32_ty))
  ];
}

def TBUFFER_STORE_D16_FORMAT_X_sem : InstSISemantic<TBUFFER_STORE_D16_FORMAT_X> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp, (LLVMTrunc (GetNamedOperand $vdata), llvm_i16_ty)),
    (LLVMStore $tmp, $tmp_addr)
  ];
}

def TBUFFER_STORE_D16_FORMAT_XY_sem : InstSISemantic<TBUFFER_STORE_D16_FORMAT_XY> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (GetNamedOperand $vdata), llvm_v2i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

def TBUFFER_STORE_D16_FORMAT_XYZ_sem : InstSISemantic<TBUFFER_STORE_D16_FORMAT_XYZ> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (GetNamedOperand $vdata), llvm_v3i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

def TBUFFER_STORE_D16_FORMAT_XYZW_sem : InstSISemantic<TBUFFER_STORE_D16_FORMAT_XYZW> {
  let Semantic = [
    (DefVal $tmp_addr, (CalculateBufferAddr<(GetNamedOperand $vaddr),
                                             (GetNamedOperand $rsrc),
                                             (GetNamedOperand $soffset),
                                             (GetNamedOperand $offset)>.ret)),
    (DefVal $tmp16, (LLVMTrunc (GetNamedOperand $vdata), llvm_v4i16_ty)),
    (LLVMStore $tmp16, $tmp_addr)
  ];
}

//==============================================================================
// Address mode, RTN, TFE, LDS, and legacy naming variants
// (empty semantics -- same core operation as base instruction)
//==============================================================================
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, Ch.8 (Vector Memory Buffer Instructions) & Sec 12.14 (MUBUF Instructions).
//   vega-shader-instruction-set-architecture.pdf, Sec 8.1 (p.50-62) & Sec 12.14 (MUBUF Instructions, p.198+).
//   vega-7nm-shader-instruction-set-architecture.pdf, Sec 8.1 (p.51-63) & Sec 12.14 (MUBUF Instructions, p.200+).
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, Sec 9.1 (p.55-67) & Sec 12.14 (MUBUF Instructions, p.188+).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, Sec 9.1 (p.55-67) & Sec 12.13 (MUBUF Instructions, p.182+).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, Sec 9.1 (p.63-72) & Sec 12.13 (MUBUF Instructions, p.450+).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, Sec 9.1 (p.74-83) & Sec 12.13 (MUBUF Instructions, p.495+).
//   rdna-shader-instruction-set-architecture.pdf, Sec 8.1 (p.56-67) & Sec 12.14 (MUBUF Instructions, p.216+).
//   rdna2-shader-instruction-set-architecture.pdf, Sec 8.1 (p.55-66) & Sec 12.14 (MUBUF Instructions, p.209+).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, Ch.9 (p.80-107) & Sec 16.16 (MUBUF Instructions, p.538+).
//   rdna4-instruction-set-architecture.pdf, Ch.9 (p.102-129) & Sec 16.16 (VBUFFER Instructions, p.616+).

// Buffer Load Format address mode / TFE / LDS variants -- same semantics as base.

foreach inst = [
    BUFFER_LOAD_FORMAT_D16_HI_X_ADDR64,
    BUFFER_LOAD_FORMAT_D16_HI_X_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_IDXEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_OFFEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_OFFSET,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_D16_HI_X_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_HI_X_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_HI_X_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_D16_XYZW_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZW_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZW_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_D16_XYZ_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZ_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XYZ_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_D16_XY_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XY_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XY_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XY_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XY_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XY_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_D16_XY_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_XY_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_XY_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_XY_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_XY_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_D16_X_ADDR64,
    BUFFER_LOAD_FORMAT_D16_X_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_X_IDXEN,
    BUFFER_LOAD_FORMAT_D16_X_OFFEN,
    BUFFER_LOAD_FORMAT_D16_X_OFFSET,
    BUFFER_LOAD_FORMAT_D16_X_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_D16_X_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_X_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_D16_X_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_D16_X_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_D16_X_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_X_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_X_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_X_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_X_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_D16_X_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_D16_X_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_D16_X_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_D16_X_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_D16_X_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_XYZW_ADDR64,
    BUFFER_LOAD_FORMAT_XYZW_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZW_IDXEN,
    BUFFER_LOAD_FORMAT_XYZW_OFFEN,
    BUFFER_LOAD_FORMAT_XYZW_OFFSET,
    BUFFER_LOAD_FORMAT_XYZW_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_XYZW_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZW_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_XYZW_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_XYZW_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_XYZW_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_XYZW_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZW_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_XYZW_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_XYZW_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_XYZW_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_XYZW_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZW_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_XYZW_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_XYZW_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_XYZ_ADDR64,
    BUFFER_LOAD_FORMAT_XYZ_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZ_IDXEN,
    BUFFER_LOAD_FORMAT_XYZ_OFFEN,
    BUFFER_LOAD_FORMAT_XYZ_OFFSET,
    BUFFER_LOAD_FORMAT_XYZ_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_XYZ_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZ_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_XYZ_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_XYZ_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_XYZ_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_XYZ_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZ_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_XYZ_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_XYZ_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_XYZ_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_XYZ_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_XYZ_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_XYZ_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_XYZ_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_XY_ADDR64,
    BUFFER_LOAD_FORMAT_XY_BOTHEN,
    BUFFER_LOAD_FORMAT_XY_IDXEN,
    BUFFER_LOAD_FORMAT_XY_OFFEN,
    BUFFER_LOAD_FORMAT_XY_OFFSET,
    BUFFER_LOAD_FORMAT_XY_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_XY_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_XY_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_XY_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_XY_TFE_OFFSET,
    BUFFER_LOAD_FORMAT_XY_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_XY_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_XY_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_XY_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_XY_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_XY_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_XY_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_XY_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_XY_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_XY_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_X_ADDR64,
    BUFFER_LOAD_FORMAT_X_BOTHEN,
    BUFFER_LOAD_FORMAT_X_IDXEN,
    BUFFER_LOAD_FORMAT_X_LDS_ADDR64,
    BUFFER_LOAD_FORMAT_X_LDS_BOTHEN,
    BUFFER_LOAD_FORMAT_X_LDS_IDXEN,
    BUFFER_LOAD_FORMAT_X_LDS_OFFEN,
    BUFFER_LOAD_FORMAT_X_LDS_OFFSET,
    BUFFER_LOAD_FORMAT_X_LDS_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_X_LDS_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_X_LDS_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_X_LDS_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_X_LDS_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_X_OFFEN,
    BUFFER_LOAD_FORMAT_X_OFFSET,
    BUFFER_LOAD_FORMAT_X_TFE_ADDR64,
    BUFFER_LOAD_FORMAT_X_TFE_BOTHEN,
    BUFFER_LOAD_FORMAT_X_TFE_IDXEN,
    BUFFER_LOAD_FORMAT_X_TFE_OFFEN,
    BUFFER_LOAD_FORMAT_X_TFE_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_FORMAT_X_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_X_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_X_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_X_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_X_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_FORMAT_X_VBUFFER_ADDR64,
    BUFFER_LOAD_FORMAT_X_VBUFFER_BOTHEN,
    BUFFER_LOAD_FORMAT_X_VBUFFER_IDXEN,
    BUFFER_LOAD_FORMAT_X_VBUFFER_OFFEN,
    BUFFER_LOAD_FORMAT_X_VBUFFER_OFFSET,
    BUFFER_LOAD_LDS_FORMAT_X_BOTHEN,
    BUFFER_LOAD_LDS_FORMAT_X_IDXEN,
    BUFFER_LOAD_LDS_FORMAT_X_OFFEN,
    BUFFER_LOAD_LDS_FORMAT_X_OFFSET,
    BUFFER_LOAD_LDS_FORMAT_X_VBUFFER_BOTHEN,
    BUFFER_LOAD_LDS_FORMAT_X_VBUFFER_IDXEN,
    BUFFER_LOAD_LDS_FORMAT_X_VBUFFER_OFFEN,
    BUFFER_LOAD_LDS_FORMAT_X_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

// Buffer Store Format address mode / TFE variants -- same semantics as base.

foreach inst = [
    BUFFER_STORE_FORMAT_D16_HI_X_ADDR64,
    BUFFER_STORE_FORMAT_D16_HI_X_BOTHEN,
    BUFFER_STORE_FORMAT_D16_HI_X_IDXEN,
    BUFFER_STORE_FORMAT_D16_HI_X_OFFEN,
    BUFFER_STORE_FORMAT_D16_HI_X_OFFSET,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_ADDR64,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_IDXEN,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_OFFEN,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_OFFSET,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_HI_X_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_D16_HI_X_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_HI_X_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_HI_X_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_HI_X_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_HI_X_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_D16_XYZW_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZW_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZW_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZW_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZW_OFFSET,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_OFFSET,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZW_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZW_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_D16_XYZ_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZ_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZ_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZ_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZ_OFFSET,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_OFFSET,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZ_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_XYZ_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_D16_XY_ADDR64,
    BUFFER_STORE_FORMAT_D16_XY_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XY_IDXEN,
    BUFFER_STORE_FORMAT_D16_XY_OFFEN,
    BUFFER_STORE_FORMAT_D16_XY_OFFSET,
    BUFFER_STORE_FORMAT_D16_XY_TFE_ADDR64,
    BUFFER_STORE_FORMAT_D16_XY_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XY_TFE_IDXEN,
    BUFFER_STORE_FORMAT_D16_XY_TFE_OFFEN,
    BUFFER_STORE_FORMAT_D16_XY_TFE_OFFSET,
    BUFFER_STORE_FORMAT_D16_XY_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_XY_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XY_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_XY_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_XY_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_D16_XY_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_XY_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_XY_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_XY_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_XY_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_D16_X_ADDR64,
    BUFFER_STORE_FORMAT_D16_X_BOTHEN,
    BUFFER_STORE_FORMAT_D16_X_IDXEN,
    BUFFER_STORE_FORMAT_D16_X_OFFEN,
    BUFFER_STORE_FORMAT_D16_X_OFFSET,
    BUFFER_STORE_FORMAT_D16_X_TFE_ADDR64,
    BUFFER_STORE_FORMAT_D16_X_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_D16_X_TFE_IDXEN,
    BUFFER_STORE_FORMAT_D16_X_TFE_OFFEN,
    BUFFER_STORE_FORMAT_D16_X_TFE_OFFSET,
    BUFFER_STORE_FORMAT_D16_X_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_X_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_X_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_X_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_X_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_D16_X_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_D16_X_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_D16_X_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_D16_X_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_D16_X_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_XYZW_ADDR64,
    BUFFER_STORE_FORMAT_XYZW_BOTHEN,
    BUFFER_STORE_FORMAT_XYZW_IDXEN,
    BUFFER_STORE_FORMAT_XYZW_OFFEN,
    BUFFER_STORE_FORMAT_XYZW_OFFSET,
    BUFFER_STORE_FORMAT_XYZW_TFE_ADDR64,
    BUFFER_STORE_FORMAT_XYZW_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_XYZW_TFE_IDXEN,
    BUFFER_STORE_FORMAT_XYZW_TFE_OFFEN,
    BUFFER_STORE_FORMAT_XYZW_TFE_OFFSET,
    BUFFER_STORE_FORMAT_XYZW_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_XYZW_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_XYZW_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_XYZW_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_XYZW_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_XYZW_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_XYZW_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_XYZW_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_XYZW_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_XYZW_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_XYZ_ADDR64,
    BUFFER_STORE_FORMAT_XYZ_BOTHEN,
    BUFFER_STORE_FORMAT_XYZ_IDXEN,
    BUFFER_STORE_FORMAT_XYZ_OFFEN,
    BUFFER_STORE_FORMAT_XYZ_OFFSET,
    BUFFER_STORE_FORMAT_XYZ_TFE_ADDR64,
    BUFFER_STORE_FORMAT_XYZ_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_XYZ_TFE_IDXEN,
    BUFFER_STORE_FORMAT_XYZ_TFE_OFFEN,
    BUFFER_STORE_FORMAT_XYZ_TFE_OFFSET,
    BUFFER_STORE_FORMAT_XYZ_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_XYZ_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_XYZ_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_XYZ_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_XYZ_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_XYZ_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_XYZ_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_XYZ_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_XYZ_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_XYZ_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_XY_ADDR64,
    BUFFER_STORE_FORMAT_XY_BOTHEN,
    BUFFER_STORE_FORMAT_XY_IDXEN,
    BUFFER_STORE_FORMAT_XY_OFFEN,
    BUFFER_STORE_FORMAT_XY_OFFSET,
    BUFFER_STORE_FORMAT_XY_TFE_ADDR64,
    BUFFER_STORE_FORMAT_XY_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_XY_TFE_IDXEN,
    BUFFER_STORE_FORMAT_XY_TFE_OFFEN,
    BUFFER_STORE_FORMAT_XY_TFE_OFFSET,
    BUFFER_STORE_FORMAT_XY_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_XY_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_XY_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_XY_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_XY_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_XY_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_XY_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_XY_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_XY_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_XY_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_FORMAT_X_ADDR64,
    BUFFER_STORE_FORMAT_X_BOTHEN,
    BUFFER_STORE_FORMAT_X_IDXEN,
    BUFFER_STORE_FORMAT_X_OFFEN,
    BUFFER_STORE_FORMAT_X_OFFSET,
    BUFFER_STORE_FORMAT_X_TFE_ADDR64,
    BUFFER_STORE_FORMAT_X_TFE_BOTHEN,
    BUFFER_STORE_FORMAT_X_TFE_IDXEN,
    BUFFER_STORE_FORMAT_X_TFE_OFFEN,
    BUFFER_STORE_FORMAT_X_TFE_OFFSET,
    BUFFER_STORE_FORMAT_X_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_X_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_X_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_X_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_X_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_FORMAT_X_VBUFFER_ADDR64,
    BUFFER_STORE_FORMAT_X_VBUFFER_BOTHEN,
    BUFFER_STORE_FORMAT_X_VBUFFER_IDXEN,
    BUFFER_STORE_FORMAT_X_VBUFFER_OFFEN,
    BUFFER_STORE_FORMAT_X_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

// Buffer Load address mode / TFE / LDS / legacy naming variants -- same semantics as base.

foreach inst = [
    BUFFER_LOAD_DWORDX2_ADDR64,
    BUFFER_LOAD_DWORDX2_BOTHEN,
    BUFFER_LOAD_DWORDX2_IDXEN,
    BUFFER_LOAD_DWORDX2_OFFEN,
    BUFFER_LOAD_DWORDX2_OFFSET,
    BUFFER_LOAD_DWORDX2_TFE_ADDR64,
    BUFFER_LOAD_DWORDX2_TFE_BOTHEN,
    BUFFER_LOAD_DWORDX2_TFE_IDXEN,
    BUFFER_LOAD_DWORDX2_TFE_OFFEN,
    BUFFER_LOAD_DWORDX2_TFE_OFFSET,
    BUFFER_LOAD_DWORDX2_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORDX2_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORDX2_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORDX2_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORDX2_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_DWORDX2_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORDX2_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORDX2_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORDX2_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORDX2_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_DWORDX3_ADDR64,
    BUFFER_LOAD_DWORDX3_BOTHEN,
    BUFFER_LOAD_DWORDX3_IDXEN,
    BUFFER_LOAD_DWORDX3_OFFEN,
    BUFFER_LOAD_DWORDX3_OFFSET,
    BUFFER_LOAD_DWORDX3_TFE_ADDR64,
    BUFFER_LOAD_DWORDX3_TFE_BOTHEN,
    BUFFER_LOAD_DWORDX3_TFE_IDXEN,
    BUFFER_LOAD_DWORDX3_TFE_OFFEN,
    BUFFER_LOAD_DWORDX3_TFE_OFFSET,
    BUFFER_LOAD_DWORDX3_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORDX3_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORDX3_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORDX3_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORDX3_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_DWORDX3_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORDX3_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORDX3_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORDX3_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORDX3_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_DWORDX4_ADDR64,
    BUFFER_LOAD_DWORDX4_BOTHEN,
    BUFFER_LOAD_DWORDX4_IDXEN,
    BUFFER_LOAD_DWORDX4_OFFEN,
    BUFFER_LOAD_DWORDX4_OFFSET,
    BUFFER_LOAD_DWORDX4_TFE_ADDR64,
    BUFFER_LOAD_DWORDX4_TFE_BOTHEN,
    BUFFER_LOAD_DWORDX4_TFE_IDXEN,
    BUFFER_LOAD_DWORDX4_TFE_OFFEN,
    BUFFER_LOAD_DWORDX4_TFE_OFFSET,
    BUFFER_LOAD_DWORDX4_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORDX4_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORDX4_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORDX4_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORDX4_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_DWORDX4_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORDX4_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORDX4_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORDX4_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORDX4_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_DWORD_ADDR64,
    BUFFER_LOAD_DWORD_BOTHEN,
    BUFFER_LOAD_DWORD_IDXEN,
    BUFFER_LOAD_DWORD_LDS_ADDR64,
    BUFFER_LOAD_DWORD_LDS_BOTHEN,
    BUFFER_LOAD_DWORD_LDS_IDXEN,
    BUFFER_LOAD_DWORD_LDS_OFFEN,
    BUFFER_LOAD_DWORD_LDS_OFFSET,
    BUFFER_LOAD_DWORD_LDS_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORD_LDS_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORD_LDS_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORD_LDS_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORD_LDS_VBUFFER_OFFSET,
    BUFFER_LOAD_DWORD_OFFEN,
    BUFFER_LOAD_DWORD_OFFSET,
    BUFFER_LOAD_DWORD_TFE_ADDR64,
    BUFFER_LOAD_DWORD_TFE_BOTHEN,
    BUFFER_LOAD_DWORD_TFE_IDXEN,
    BUFFER_LOAD_DWORD_TFE_OFFEN,
    BUFFER_LOAD_DWORD_TFE_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_DWORD_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORD_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORD_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORD_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORD_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_DWORD_VBUFFER_ADDR64,
    BUFFER_LOAD_DWORD_VBUFFER_BOTHEN,
    BUFFER_LOAD_DWORD_VBUFFER_IDXEN,
    BUFFER_LOAD_DWORD_VBUFFER_OFFEN,
    BUFFER_LOAD_DWORD_VBUFFER_OFFSET,
    BUFFER_LOAD_LDS_B32_BOTHEN,
    BUFFER_LOAD_LDS_B32_IDXEN,
    BUFFER_LOAD_LDS_B32_OFFEN,
    BUFFER_LOAD_LDS_B32_OFFSET,
    BUFFER_LOAD_LDS_B32_VBUFFER_BOTHEN,
    BUFFER_LOAD_LDS_B32_VBUFFER_IDXEN,
    BUFFER_LOAD_LDS_B32_VBUFFER_OFFEN,
    BUFFER_LOAD_LDS_B32_VBUFFER_OFFSET,
    BUFFER_LOAD_LDS_I16_BOTHEN,
    BUFFER_LOAD_LDS_I16_IDXEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_LDS_I16_OFFEN,
    BUFFER_LOAD_LDS_I16_OFFSET,
    BUFFER_LOAD_LDS_I16_VBUFFER_BOTHEN,
    BUFFER_LOAD_LDS_I16_VBUFFER_IDXEN,
    BUFFER_LOAD_LDS_I16_VBUFFER_OFFEN,
    BUFFER_LOAD_LDS_I16_VBUFFER_OFFSET,
    BUFFER_LOAD_LDS_I8_BOTHEN,
    BUFFER_LOAD_LDS_I8_IDXEN,
    BUFFER_LOAD_LDS_I8_OFFEN,
    BUFFER_LOAD_LDS_I8_OFFSET,
    BUFFER_LOAD_LDS_I8_VBUFFER_BOTHEN,
    BUFFER_LOAD_LDS_I8_VBUFFER_IDXEN,
    BUFFER_LOAD_LDS_I8_VBUFFER_OFFEN,
    BUFFER_LOAD_LDS_I8_VBUFFER_OFFSET,
    BUFFER_LOAD_LDS_U16_BOTHEN,
    BUFFER_LOAD_LDS_U16_IDXEN,
    BUFFER_LOAD_LDS_U16_OFFEN,
    BUFFER_LOAD_LDS_U16_OFFSET,
    BUFFER_LOAD_LDS_U16_VBUFFER_BOTHEN,
    BUFFER_LOAD_LDS_U16_VBUFFER_IDXEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_LDS_U16_VBUFFER_OFFEN,
    BUFFER_LOAD_LDS_U16_VBUFFER_OFFSET,
    BUFFER_LOAD_LDS_U8_BOTHEN,
    BUFFER_LOAD_LDS_U8_IDXEN,
    BUFFER_LOAD_LDS_U8_OFFEN,
    BUFFER_LOAD_LDS_U8_OFFSET,
    BUFFER_LOAD_LDS_U8_VBUFFER_BOTHEN,
    BUFFER_LOAD_LDS_U8_VBUFFER_IDXEN,
    BUFFER_LOAD_LDS_U8_VBUFFER_OFFEN,
    BUFFER_LOAD_LDS_U8_VBUFFER_OFFSET,
    BUFFER_LOAD_SBYTE_ADDR64,
    BUFFER_LOAD_SBYTE_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_ADDR64,
    BUFFER_LOAD_SBYTE_D16_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_HI_ADDR64,
    BUFFER_LOAD_SBYTE_D16_HI_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_HI_IDXEN,
    BUFFER_LOAD_SBYTE_D16_HI_OFFEN,
    BUFFER_LOAD_SBYTE_D16_HI_OFFSET,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_ADDR64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_SBYTE_D16_HI_TFE_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_IDXEN,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_OFFEN,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_OFFSET,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_SBYTE_D16_HI_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_SBYTE_D16_HI_VBUFFER_ADDR64,
    BUFFER_LOAD_SBYTE_D16_HI_VBUFFER_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_HI_VBUFFER_IDXEN,
    BUFFER_LOAD_SBYTE_D16_HI_VBUFFER_OFFEN,
    BUFFER_LOAD_SBYTE_D16_HI_VBUFFER_OFFSET,
    BUFFER_LOAD_SBYTE_D16_IDXEN,
    BUFFER_LOAD_SBYTE_D16_OFFEN,
    BUFFER_LOAD_SBYTE_D16_OFFSET,
    BUFFER_LOAD_SBYTE_D16_TFE_ADDR64,
    BUFFER_LOAD_SBYTE_D16_TFE_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_TFE_IDXEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_SBYTE_D16_TFE_OFFEN,
    BUFFER_LOAD_SBYTE_D16_TFE_OFFSET,
    BUFFER_LOAD_SBYTE_D16_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_SBYTE_D16_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_SBYTE_D16_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_SBYTE_D16_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_SBYTE_D16_VBUFFER_ADDR64,
    BUFFER_LOAD_SBYTE_D16_VBUFFER_BOTHEN,
    BUFFER_LOAD_SBYTE_D16_VBUFFER_IDXEN,
    BUFFER_LOAD_SBYTE_D16_VBUFFER_OFFEN,
    BUFFER_LOAD_SBYTE_D16_VBUFFER_OFFSET,
    BUFFER_LOAD_SBYTE_IDXEN,
    BUFFER_LOAD_SBYTE_LDS_ADDR64,
    BUFFER_LOAD_SBYTE_LDS_BOTHEN,
    BUFFER_LOAD_SBYTE_LDS_IDXEN,
    BUFFER_LOAD_SBYTE_LDS_OFFEN,
    BUFFER_LOAD_SBYTE_LDS_OFFSET,
    BUFFER_LOAD_SBYTE_LDS_VBUFFER_ADDR64,
    BUFFER_LOAD_SBYTE_LDS_VBUFFER_BOTHEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_SBYTE_LDS_VBUFFER_IDXEN,
    BUFFER_LOAD_SBYTE_LDS_VBUFFER_OFFEN,
    BUFFER_LOAD_SBYTE_LDS_VBUFFER_OFFSET,
    BUFFER_LOAD_SBYTE_OFFEN,
    BUFFER_LOAD_SBYTE_OFFSET,
    BUFFER_LOAD_SBYTE_TFE_ADDR64,
    BUFFER_LOAD_SBYTE_TFE_BOTHEN,
    BUFFER_LOAD_SBYTE_TFE_IDXEN,
    BUFFER_LOAD_SBYTE_TFE_OFFEN,
    BUFFER_LOAD_SBYTE_TFE_OFFSET,
    BUFFER_LOAD_SBYTE_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_SBYTE_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_SBYTE_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_SBYTE_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_SBYTE_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_SBYTE_VBUFFER_ADDR64,
    BUFFER_LOAD_SBYTE_VBUFFER_BOTHEN,
    BUFFER_LOAD_SBYTE_VBUFFER_IDXEN,
    BUFFER_LOAD_SBYTE_VBUFFER_OFFEN,
    BUFFER_LOAD_SBYTE_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_SHORT_D16_ADDR64,
    BUFFER_LOAD_SHORT_D16_BOTHEN,
    BUFFER_LOAD_SHORT_D16_HI_ADDR64,
    BUFFER_LOAD_SHORT_D16_HI_BOTHEN,
    BUFFER_LOAD_SHORT_D16_HI_IDXEN,
    BUFFER_LOAD_SHORT_D16_HI_OFFEN,
    BUFFER_LOAD_SHORT_D16_HI_OFFSET,
    BUFFER_LOAD_SHORT_D16_HI_TFE_ADDR64,
    BUFFER_LOAD_SHORT_D16_HI_TFE_BOTHEN,
    BUFFER_LOAD_SHORT_D16_HI_TFE_IDXEN,
    BUFFER_LOAD_SHORT_D16_HI_TFE_OFFEN,
    BUFFER_LOAD_SHORT_D16_HI_TFE_OFFSET,
    BUFFER_LOAD_SHORT_D16_HI_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_SHORT_D16_HI_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_SHORT_D16_HI_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_SHORT_D16_HI_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_SHORT_D16_HI_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_SHORT_D16_HI_VBUFFER_ADDR64,
    BUFFER_LOAD_SHORT_D16_HI_VBUFFER_BOTHEN,
    BUFFER_LOAD_SHORT_D16_HI_VBUFFER_IDXEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_SHORT_D16_HI_VBUFFER_OFFEN,
    BUFFER_LOAD_SHORT_D16_HI_VBUFFER_OFFSET,
    BUFFER_LOAD_SHORT_D16_IDXEN,
    BUFFER_LOAD_SHORT_D16_OFFEN,
    BUFFER_LOAD_SHORT_D16_OFFSET,
    BUFFER_LOAD_SHORT_D16_TFE_ADDR64,
    BUFFER_LOAD_SHORT_D16_TFE_BOTHEN,
    BUFFER_LOAD_SHORT_D16_TFE_IDXEN,
    BUFFER_LOAD_SHORT_D16_TFE_OFFEN,
    BUFFER_LOAD_SHORT_D16_TFE_OFFSET,
    BUFFER_LOAD_SHORT_D16_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_SHORT_D16_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_SHORT_D16_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_SHORT_D16_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_SHORT_D16_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_SHORT_D16_VBUFFER_ADDR64,
    BUFFER_LOAD_SHORT_D16_VBUFFER_BOTHEN,
    BUFFER_LOAD_SHORT_D16_VBUFFER_IDXEN,
    BUFFER_LOAD_SHORT_D16_VBUFFER_OFFEN,
    BUFFER_LOAD_SHORT_D16_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_SSHORT_ADDR64,
    BUFFER_LOAD_SSHORT_BOTHEN,
    BUFFER_LOAD_SSHORT_IDXEN,
    BUFFER_LOAD_SSHORT_LDS_ADDR64,
    BUFFER_LOAD_SSHORT_LDS_BOTHEN,
    BUFFER_LOAD_SSHORT_LDS_IDXEN,
    BUFFER_LOAD_SSHORT_LDS_OFFEN,
    BUFFER_LOAD_SSHORT_LDS_OFFSET,
    BUFFER_LOAD_SSHORT_LDS_VBUFFER_ADDR64,
    BUFFER_LOAD_SSHORT_LDS_VBUFFER_BOTHEN,
    BUFFER_LOAD_SSHORT_LDS_VBUFFER_IDXEN,
    BUFFER_LOAD_SSHORT_LDS_VBUFFER_OFFEN,
    BUFFER_LOAD_SSHORT_LDS_VBUFFER_OFFSET,
    BUFFER_LOAD_SSHORT_OFFEN,
    BUFFER_LOAD_SSHORT_OFFSET,
    BUFFER_LOAD_SSHORT_TFE_ADDR64,
    BUFFER_LOAD_SSHORT_TFE_BOTHEN,
    BUFFER_LOAD_SSHORT_TFE_IDXEN,
    BUFFER_LOAD_SSHORT_TFE_OFFEN,
    BUFFER_LOAD_SSHORT_TFE_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_SSHORT_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_SSHORT_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_SSHORT_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_SSHORT_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_SSHORT_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_SSHORT_VBUFFER_ADDR64,
    BUFFER_LOAD_SSHORT_VBUFFER_BOTHEN,
    BUFFER_LOAD_SSHORT_VBUFFER_IDXEN,
    BUFFER_LOAD_SSHORT_VBUFFER_OFFEN,
    BUFFER_LOAD_SSHORT_VBUFFER_OFFSET,
    BUFFER_LOAD_UBYTE_ADDR64,
    BUFFER_LOAD_UBYTE_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_ADDR64,
    BUFFER_LOAD_UBYTE_D16_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_HI_ADDR64,
    BUFFER_LOAD_UBYTE_D16_HI_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_HI_IDXEN,
    BUFFER_LOAD_UBYTE_D16_HI_OFFEN,
    BUFFER_LOAD_UBYTE_D16_HI_OFFSET,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_ADDR64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_UBYTE_D16_HI_TFE_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_IDXEN,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_OFFEN,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_OFFSET,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_UBYTE_D16_HI_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_UBYTE_D16_HI_VBUFFER_ADDR64,
    BUFFER_LOAD_UBYTE_D16_HI_VBUFFER_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_HI_VBUFFER_IDXEN,
    BUFFER_LOAD_UBYTE_D16_HI_VBUFFER_OFFEN,
    BUFFER_LOAD_UBYTE_D16_HI_VBUFFER_OFFSET,
    BUFFER_LOAD_UBYTE_D16_IDXEN,
    BUFFER_LOAD_UBYTE_D16_OFFEN,
    BUFFER_LOAD_UBYTE_D16_OFFSET,
    BUFFER_LOAD_UBYTE_D16_TFE_ADDR64,
    BUFFER_LOAD_UBYTE_D16_TFE_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_TFE_IDXEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_UBYTE_D16_TFE_OFFEN,
    BUFFER_LOAD_UBYTE_D16_TFE_OFFSET,
    BUFFER_LOAD_UBYTE_D16_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_UBYTE_D16_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_UBYTE_D16_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_UBYTE_D16_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_UBYTE_D16_VBUFFER_ADDR64,
    BUFFER_LOAD_UBYTE_D16_VBUFFER_BOTHEN,
    BUFFER_LOAD_UBYTE_D16_VBUFFER_IDXEN,
    BUFFER_LOAD_UBYTE_D16_VBUFFER_OFFEN,
    BUFFER_LOAD_UBYTE_D16_VBUFFER_OFFSET,
    BUFFER_LOAD_UBYTE_IDXEN,
    BUFFER_LOAD_UBYTE_LDS_ADDR64,
    BUFFER_LOAD_UBYTE_LDS_BOTHEN,
    BUFFER_LOAD_UBYTE_LDS_IDXEN,
    BUFFER_LOAD_UBYTE_LDS_OFFEN,
    BUFFER_LOAD_UBYTE_LDS_OFFSET,
    BUFFER_LOAD_UBYTE_LDS_VBUFFER_ADDR64,
    BUFFER_LOAD_UBYTE_LDS_VBUFFER_BOTHEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_UBYTE_LDS_VBUFFER_IDXEN,
    BUFFER_LOAD_UBYTE_LDS_VBUFFER_OFFEN,
    BUFFER_LOAD_UBYTE_LDS_VBUFFER_OFFSET,
    BUFFER_LOAD_UBYTE_OFFEN,
    BUFFER_LOAD_UBYTE_OFFSET,
    BUFFER_LOAD_UBYTE_TFE_ADDR64,
    BUFFER_LOAD_UBYTE_TFE_BOTHEN,
    BUFFER_LOAD_UBYTE_TFE_IDXEN,
    BUFFER_LOAD_UBYTE_TFE_OFFEN,
    BUFFER_LOAD_UBYTE_TFE_OFFSET,
    BUFFER_LOAD_UBYTE_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_UBYTE_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_UBYTE_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_UBYTE_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_UBYTE_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_UBYTE_VBUFFER_ADDR64,
    BUFFER_LOAD_UBYTE_VBUFFER_BOTHEN,
    BUFFER_LOAD_UBYTE_VBUFFER_IDXEN,
    BUFFER_LOAD_UBYTE_VBUFFER_OFFEN,
    BUFFER_LOAD_UBYTE_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_USHORT_ADDR64,
    BUFFER_LOAD_USHORT_BOTHEN,
    BUFFER_LOAD_USHORT_IDXEN,
    BUFFER_LOAD_USHORT_LDS_ADDR64,
    BUFFER_LOAD_USHORT_LDS_BOTHEN,
    BUFFER_LOAD_USHORT_LDS_IDXEN,
    BUFFER_LOAD_USHORT_LDS_OFFEN,
    BUFFER_LOAD_USHORT_LDS_OFFSET,
    BUFFER_LOAD_USHORT_LDS_VBUFFER_ADDR64,
    BUFFER_LOAD_USHORT_LDS_VBUFFER_BOTHEN,
    BUFFER_LOAD_USHORT_LDS_VBUFFER_IDXEN,
    BUFFER_LOAD_USHORT_LDS_VBUFFER_OFFEN,
    BUFFER_LOAD_USHORT_LDS_VBUFFER_OFFSET,
    BUFFER_LOAD_USHORT_OFFEN,
    BUFFER_LOAD_USHORT_OFFSET,
    BUFFER_LOAD_USHORT_TFE_ADDR64,
    BUFFER_LOAD_USHORT_TFE_BOTHEN,
    BUFFER_LOAD_USHORT_TFE_IDXEN,
    BUFFER_LOAD_USHORT_TFE_OFFEN,
    BUFFER_LOAD_USHORT_TFE_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_LOAD_USHORT_TFE_VBUFFER_ADDR64,
    BUFFER_LOAD_USHORT_TFE_VBUFFER_BOTHEN,
    BUFFER_LOAD_USHORT_TFE_VBUFFER_IDXEN,
    BUFFER_LOAD_USHORT_TFE_VBUFFER_OFFEN,
    BUFFER_LOAD_USHORT_TFE_VBUFFER_OFFSET,
    BUFFER_LOAD_USHORT_VBUFFER_ADDR64,
    BUFFER_LOAD_USHORT_VBUFFER_BOTHEN,
    BUFFER_LOAD_USHORT_VBUFFER_IDXEN,
    BUFFER_LOAD_USHORT_VBUFFER_OFFEN,
    BUFFER_LOAD_USHORT_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

// Buffer Store address mode / TFE / legacy naming variants -- same semantics as base.

foreach inst = [
    BUFFER_STORE_BYTE_ADDR64,
    BUFFER_STORE_BYTE_BOTHEN,
    BUFFER_STORE_BYTE_D16_HI_ADDR64,
    BUFFER_STORE_BYTE_D16_HI_BOTHEN,
    BUFFER_STORE_BYTE_D16_HI_IDXEN,
    BUFFER_STORE_BYTE_D16_HI_OFFEN,
    BUFFER_STORE_BYTE_D16_HI_OFFSET,
    BUFFER_STORE_BYTE_D16_HI_TFE_ADDR64,
    BUFFER_STORE_BYTE_D16_HI_TFE_BOTHEN,
    BUFFER_STORE_BYTE_D16_HI_TFE_IDXEN,
    BUFFER_STORE_BYTE_D16_HI_TFE_OFFEN,
    BUFFER_STORE_BYTE_D16_HI_TFE_OFFSET,
    BUFFER_STORE_BYTE_D16_HI_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_BYTE_D16_HI_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_BYTE_D16_HI_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_BYTE_D16_HI_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_BYTE_D16_HI_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_BYTE_D16_HI_VBUFFER_ADDR64,
    BUFFER_STORE_BYTE_D16_HI_VBUFFER_BOTHEN,
    BUFFER_STORE_BYTE_D16_HI_VBUFFER_IDXEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_BYTE_D16_HI_VBUFFER_OFFEN,
    BUFFER_STORE_BYTE_D16_HI_VBUFFER_OFFSET,
    BUFFER_STORE_BYTE_IDXEN,
    BUFFER_STORE_BYTE_OFFEN,
    BUFFER_STORE_BYTE_OFFSET,
    BUFFER_STORE_BYTE_TFE_ADDR64,
    BUFFER_STORE_BYTE_TFE_BOTHEN,
    BUFFER_STORE_BYTE_TFE_IDXEN,
    BUFFER_STORE_BYTE_TFE_OFFEN,
    BUFFER_STORE_BYTE_TFE_OFFSET,
    BUFFER_STORE_BYTE_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_BYTE_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_BYTE_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_BYTE_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_BYTE_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_BYTE_VBUFFER_ADDR64,
    BUFFER_STORE_BYTE_VBUFFER_BOTHEN,
    BUFFER_STORE_BYTE_VBUFFER_IDXEN,
    BUFFER_STORE_BYTE_VBUFFER_OFFEN,
    BUFFER_STORE_BYTE_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_DWORDX2_ADDR64,
    BUFFER_STORE_DWORDX2_BOTHEN,
    BUFFER_STORE_DWORDX2_IDXEN,
    BUFFER_STORE_DWORDX2_OFFEN,
    BUFFER_STORE_DWORDX2_OFFSET,
    BUFFER_STORE_DWORDX2_TFE_ADDR64,
    BUFFER_STORE_DWORDX2_TFE_BOTHEN,
    BUFFER_STORE_DWORDX2_TFE_IDXEN,
    BUFFER_STORE_DWORDX2_TFE_OFFEN,
    BUFFER_STORE_DWORDX2_TFE_OFFSET,
    BUFFER_STORE_DWORDX2_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_DWORDX2_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORDX2_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_DWORDX2_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_DWORDX2_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_DWORDX2_VBUFFER_ADDR64,
    BUFFER_STORE_DWORDX2_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORDX2_VBUFFER_IDXEN,
    BUFFER_STORE_DWORDX2_VBUFFER_OFFEN,
    BUFFER_STORE_DWORDX2_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_DWORDX3_ADDR64,
    BUFFER_STORE_DWORDX3_BOTHEN,
    BUFFER_STORE_DWORDX3_IDXEN,
    BUFFER_STORE_DWORDX3_OFFEN,
    BUFFER_STORE_DWORDX3_OFFSET,
    BUFFER_STORE_DWORDX3_TFE_ADDR64,
    BUFFER_STORE_DWORDX3_TFE_BOTHEN,
    BUFFER_STORE_DWORDX3_TFE_IDXEN,
    BUFFER_STORE_DWORDX3_TFE_OFFEN,
    BUFFER_STORE_DWORDX3_TFE_OFFSET,
    BUFFER_STORE_DWORDX3_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_DWORDX3_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORDX3_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_DWORDX3_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_DWORDX3_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_DWORDX3_VBUFFER_ADDR64,
    BUFFER_STORE_DWORDX3_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORDX3_VBUFFER_IDXEN,
    BUFFER_STORE_DWORDX3_VBUFFER_OFFEN,
    BUFFER_STORE_DWORDX3_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_DWORDX4_ADDR64,
    BUFFER_STORE_DWORDX4_BOTHEN,
    BUFFER_STORE_DWORDX4_IDXEN,
    BUFFER_STORE_DWORDX4_OFFEN,
    BUFFER_STORE_DWORDX4_OFFSET,
    BUFFER_STORE_DWORDX4_TFE_ADDR64,
    BUFFER_STORE_DWORDX4_TFE_BOTHEN,
    BUFFER_STORE_DWORDX4_TFE_IDXEN,
    BUFFER_STORE_DWORDX4_TFE_OFFEN,
    BUFFER_STORE_DWORDX4_TFE_OFFSET,
    BUFFER_STORE_DWORDX4_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_DWORDX4_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORDX4_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_DWORDX4_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_DWORDX4_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_DWORDX4_VBUFFER_ADDR64,
    BUFFER_STORE_DWORDX4_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORDX4_VBUFFER_IDXEN,
    BUFFER_STORE_DWORDX4_VBUFFER_OFFEN,
    BUFFER_STORE_DWORDX4_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_DWORD_ADDR64,
    BUFFER_STORE_DWORD_BOTHEN,
    BUFFER_STORE_DWORD_IDXEN,
    BUFFER_STORE_DWORD_OFFEN,
    BUFFER_STORE_DWORD_OFFSET,
    BUFFER_STORE_DWORD_TFE_ADDR64,
    BUFFER_STORE_DWORD_TFE_BOTHEN,
    BUFFER_STORE_DWORD_TFE_IDXEN,
    BUFFER_STORE_DWORD_TFE_OFFEN,
    BUFFER_STORE_DWORD_TFE_OFFSET,
    BUFFER_STORE_DWORD_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_DWORD_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORD_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_DWORD_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_DWORD_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_DWORD_VBUFFER_ADDR64,
    BUFFER_STORE_DWORD_VBUFFER_BOTHEN,
    BUFFER_STORE_DWORD_VBUFFER_IDXEN,
    BUFFER_STORE_DWORD_VBUFFER_OFFEN,
    BUFFER_STORE_DWORD_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_LDS_DWORD,
    BUFFER_STORE_SHORT_ADDR64,
    BUFFER_STORE_SHORT_BOTHEN,
    BUFFER_STORE_SHORT_D16_HI_ADDR64,
    BUFFER_STORE_SHORT_D16_HI_BOTHEN,
    BUFFER_STORE_SHORT_D16_HI_IDXEN,
    BUFFER_STORE_SHORT_D16_HI_OFFEN,
    BUFFER_STORE_SHORT_D16_HI_OFFSET,
    BUFFER_STORE_SHORT_D16_HI_TFE_ADDR64,
    BUFFER_STORE_SHORT_D16_HI_TFE_BOTHEN,
    BUFFER_STORE_SHORT_D16_HI_TFE_IDXEN,
    BUFFER_STORE_SHORT_D16_HI_TFE_OFFEN,
    BUFFER_STORE_SHORT_D16_HI_TFE_OFFSET,
    BUFFER_STORE_SHORT_D16_HI_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_SHORT_D16_HI_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_SHORT_D16_HI_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_SHORT_D16_HI_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_SHORT_D16_HI_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_SHORT_D16_HI_VBUFFER_ADDR64,
    BUFFER_STORE_SHORT_D16_HI_VBUFFER_BOTHEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_SHORT_D16_HI_VBUFFER_IDXEN,
    BUFFER_STORE_SHORT_D16_HI_VBUFFER_OFFEN,
    BUFFER_STORE_SHORT_D16_HI_VBUFFER_OFFSET,
    BUFFER_STORE_SHORT_IDXEN,
    BUFFER_STORE_SHORT_OFFEN,
    BUFFER_STORE_SHORT_OFFSET,
    BUFFER_STORE_SHORT_TFE_ADDR64,
    BUFFER_STORE_SHORT_TFE_BOTHEN,
    BUFFER_STORE_SHORT_TFE_IDXEN,
    BUFFER_STORE_SHORT_TFE_OFFEN,
    BUFFER_STORE_SHORT_TFE_OFFSET,
    BUFFER_STORE_SHORT_TFE_VBUFFER_ADDR64,
    BUFFER_STORE_SHORT_TFE_VBUFFER_BOTHEN,
    BUFFER_STORE_SHORT_TFE_VBUFFER_IDXEN,
    BUFFER_STORE_SHORT_TFE_VBUFFER_OFFEN,
    BUFFER_STORE_SHORT_TFE_VBUFFER_OFFSET,
    BUFFER_STORE_SHORT_VBUFFER_ADDR64,
    BUFFER_STORE_SHORT_VBUFFER_BOTHEN,
    BUFFER_STORE_SHORT_VBUFFER_IDXEN,
    BUFFER_STORE_SHORT_VBUFFER_OFFEN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_STORE_SHORT_VBUFFER_OFFSET
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

// Buffer Atomic address mode / RTN / legacy naming / float variants -- same semantics as base.

foreach inst = [
    BUFFER_ATOMIC_ADD_ADDR64,
    BUFFER_ATOMIC_ADD_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_BOTHEN,
    BUFFER_ATOMIC_ADD_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_F32_ADDR64,
    BUFFER_ATOMIC_ADD_F32_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_F32_BOTHEN,
    BUFFER_ATOMIC_ADD_F32_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_F32_IDXEN,
    BUFFER_ATOMIC_ADD_F32_IDXEN_RTN,
    BUFFER_ATOMIC_ADD_F32_OFFEN,
    BUFFER_ATOMIC_ADD_F32_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_F32_OFFSET,
    BUFFER_ATOMIC_ADD_F32_OFFSET_RTN,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_ADDR64,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_IDXEN,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_IDXEN_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_ADD_F32_VBUFFER_OFFEN,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_OFFSET,
    BUFFER_ATOMIC_ADD_F32_VBUFFER_OFFSET_RTN,
    BUFFER_ATOMIC_ADD_F64_ADDR64,
    BUFFER_ATOMIC_ADD_F64_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_F64_BOTHEN,
    BUFFER_ATOMIC_ADD_F64_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_F64_IDXEN,
    BUFFER_ATOMIC_ADD_F64_IDXEN_RTN,
    BUFFER_ATOMIC_ADD_F64_OFFEN,
    BUFFER_ATOMIC_ADD_F64_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_F64_OFFSET,
    BUFFER_ATOMIC_ADD_F64_OFFSET_RTN,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_ADDR64,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_IDXEN,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_IDXEN_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_ADD_F64_VBUFFER_OFFEN,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_OFFSET,
    BUFFER_ATOMIC_ADD_F64_VBUFFER_OFFSET_RTN,
    BUFFER_ATOMIC_ADD_IDXEN,
    BUFFER_ATOMIC_ADD_IDXEN_RTN,
    BUFFER_ATOMIC_ADD_OFFEN,
    BUFFER_ATOMIC_ADD_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_OFFSET,
    BUFFER_ATOMIC_ADD_OFFSET_RTN,
    BUFFER_ATOMIC_ADD_VBUFFER_ADDR64,
    BUFFER_ATOMIC_ADD_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_ADD_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_VBUFFER_IDXEN,
    BUFFER_ATOMIC_ADD_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_ADD_VBUFFER_OFFEN,
    BUFFER_ATOMIC_ADD_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_VBUFFER_OFFSET,
    BUFFER_ATOMIC_ADD_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_ADD_X2_ADDR64,
    BUFFER_ATOMIC_ADD_X2_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_X2_BOTHEN,
    BUFFER_ATOMIC_ADD_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_X2_IDXEN,
    BUFFER_ATOMIC_ADD_X2_IDXEN_RTN,
    BUFFER_ATOMIC_ADD_X2_OFFEN,
    BUFFER_ATOMIC_ADD_X2_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_X2_OFFSET,
    BUFFER_ATOMIC_ADD_X2_OFFSET_RTN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_ADD_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_AND_ADDR64,
    BUFFER_ATOMIC_AND_ADDR64_RTN,
    BUFFER_ATOMIC_AND_BOTHEN,
    BUFFER_ATOMIC_AND_BOTHEN_RTN,
    BUFFER_ATOMIC_AND_IDXEN,
    BUFFER_ATOMIC_AND_IDXEN_RTN,
    BUFFER_ATOMIC_AND_OFFEN,
    BUFFER_ATOMIC_AND_OFFEN_RTN,
    BUFFER_ATOMIC_AND_OFFSET,
    BUFFER_ATOMIC_AND_OFFSET_RTN,
    BUFFER_ATOMIC_AND_VBUFFER_ADDR64,
    BUFFER_ATOMIC_AND_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_AND_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_AND_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_AND_VBUFFER_IDXEN,
    BUFFER_ATOMIC_AND_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_AND_VBUFFER_OFFEN,
    BUFFER_ATOMIC_AND_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_AND_VBUFFER_OFFSET,
    BUFFER_ATOMIC_AND_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_AND_X2_ADDR64,
    BUFFER_ATOMIC_AND_X2_ADDR64_RTN,
    BUFFER_ATOMIC_AND_X2_BOTHEN,
    BUFFER_ATOMIC_AND_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_AND_X2_IDXEN,
    BUFFER_ATOMIC_AND_X2_IDXEN_RTN,
    BUFFER_ATOMIC_AND_X2_OFFEN,
    BUFFER_ATOMIC_AND_X2_OFFEN_RTN,
    BUFFER_ATOMIC_AND_X2_OFFSET,
    BUFFER_ATOMIC_AND_X2_OFFSET_RTN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_AND_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_AND_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_AND_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_CMPSWAP_ADDR64,
    BUFFER_ATOMIC_CMPSWAP_ADDR64_RTN,
    BUFFER_ATOMIC_CMPSWAP_BOTHEN,
    BUFFER_ATOMIC_CMPSWAP_BOTHEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_IDXEN,
    BUFFER_ATOMIC_CMPSWAP_IDXEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_OFFEN,
    BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_OFFSET,
    BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_ADDR64,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_IDXEN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_OFFEN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_OFFSET,
    BUFFER_ATOMIC_CMPSWAP_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_CMPSWAP_X2_ADDR64,
    BUFFER_ATOMIC_CMPSWAP_X2_ADDR64_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN,
    BUFFER_ATOMIC_CMPSWAP_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_IDXEN,
    BUFFER_ATOMIC_CMPSWAP_X2_IDXEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_OFFEN,
    BUFFER_ATOMIC_CMPSWAP_X2_OFFEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_OFFSET,
    BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_CMPSWAP_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_COND_SUB_U32_ADDR64,
    BUFFER_ATOMIC_COND_SUB_U32_ADDR64_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_BOTHEN,
    BUFFER_ATOMIC_COND_SUB_U32_BOTHEN_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_IDXEN,
    BUFFER_ATOMIC_COND_SUB_U32_IDXEN_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_OFFEN,
    BUFFER_ATOMIC_COND_SUB_U32_OFFEN_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_OFFSET,
    BUFFER_ATOMIC_COND_SUB_U32_OFFSET_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_ADDR64,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_IDXEN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_OFFEN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_OFFSET,
    BUFFER_ATOMIC_COND_SUB_U32_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_CSUB_ADDR64,
    BUFFER_ATOMIC_CSUB_ADDR64_RTN,
    BUFFER_ATOMIC_CSUB_BOTHEN,
    BUFFER_ATOMIC_CSUB_BOTHEN_RTN,
    BUFFER_ATOMIC_CSUB_IDXEN,
    BUFFER_ATOMIC_CSUB_IDXEN_RTN,
    BUFFER_ATOMIC_CSUB_OFFEN,
    BUFFER_ATOMIC_CSUB_OFFEN_RTN,
    BUFFER_ATOMIC_CSUB_OFFSET,
    BUFFER_ATOMIC_CSUB_OFFSET_RTN,
    BUFFER_ATOMIC_CSUB_VBUFFER_ADDR64,
    BUFFER_ATOMIC_CSUB_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_CSUB_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_CSUB_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_CSUB_VBUFFER_IDXEN,
    BUFFER_ATOMIC_CSUB_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_CSUB_VBUFFER_OFFEN,
    BUFFER_ATOMIC_CSUB_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_CSUB_VBUFFER_OFFSET,
    BUFFER_ATOMIC_CSUB_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_DEC_ADDR64,
    BUFFER_ATOMIC_DEC_ADDR64_RTN,
    BUFFER_ATOMIC_DEC_BOTHEN,
    BUFFER_ATOMIC_DEC_BOTHEN_RTN,
    BUFFER_ATOMIC_DEC_IDXEN,
    BUFFER_ATOMIC_DEC_IDXEN_RTN,
    BUFFER_ATOMIC_DEC_OFFEN,
    BUFFER_ATOMIC_DEC_OFFEN_RTN,
    BUFFER_ATOMIC_DEC_OFFSET,
    BUFFER_ATOMIC_DEC_OFFSET_RTN,
    BUFFER_ATOMIC_DEC_VBUFFER_ADDR64,
    BUFFER_ATOMIC_DEC_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_DEC_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_DEC_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_DEC_VBUFFER_IDXEN,
    BUFFER_ATOMIC_DEC_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_DEC_VBUFFER_OFFEN,
    BUFFER_ATOMIC_DEC_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_DEC_VBUFFER_OFFSET,
    BUFFER_ATOMIC_DEC_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_DEC_X2_ADDR64,
    BUFFER_ATOMIC_DEC_X2_ADDR64_RTN,
    BUFFER_ATOMIC_DEC_X2_BOTHEN,
    BUFFER_ATOMIC_DEC_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_DEC_X2_IDXEN,
    BUFFER_ATOMIC_DEC_X2_IDXEN_RTN,
    BUFFER_ATOMIC_DEC_X2_OFFEN,
    BUFFER_ATOMIC_DEC_X2_OFFEN_RTN,
    BUFFER_ATOMIC_DEC_X2_OFFSET,
    BUFFER_ATOMIC_DEC_X2_OFFSET_RTN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_DEC_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_FCMPSWAP_ADDR64,
    BUFFER_ATOMIC_FCMPSWAP_ADDR64_RTN,
    BUFFER_ATOMIC_FCMPSWAP_BOTHEN,
    BUFFER_ATOMIC_FCMPSWAP_BOTHEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_IDXEN,
    BUFFER_ATOMIC_FCMPSWAP_IDXEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_OFFEN,
    BUFFER_ATOMIC_FCMPSWAP_OFFEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_OFFSET,
    BUFFER_ATOMIC_FCMPSWAP_OFFSET_RTN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_ADDR64,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_IDXEN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_OFFEN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_OFFSET,
    BUFFER_ATOMIC_FCMPSWAP_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_FCMPSWAP_X2_ADDR64,
    BUFFER_ATOMIC_FCMPSWAP_X2_ADDR64_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_BOTHEN,
    BUFFER_ATOMIC_FCMPSWAP_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_IDXEN,
    BUFFER_ATOMIC_FCMPSWAP_X2_IDXEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_OFFEN,
    BUFFER_ATOMIC_FCMPSWAP_X2_OFFEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_OFFSET,
    BUFFER_ATOMIC_FCMPSWAP_X2_OFFSET_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_FCMPSWAP_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_FMAX_ADDR64,
    BUFFER_ATOMIC_FMAX_ADDR64_RTN,
    BUFFER_ATOMIC_FMAX_BOTHEN,
    BUFFER_ATOMIC_FMAX_BOTHEN_RTN,
    BUFFER_ATOMIC_FMAX_IDXEN,
    BUFFER_ATOMIC_FMAX_IDXEN_RTN,
    BUFFER_ATOMIC_FMAX_OFFEN,
    BUFFER_ATOMIC_FMAX_OFFEN_RTN,
    BUFFER_ATOMIC_FMAX_OFFSET,
    BUFFER_ATOMIC_FMAX_OFFSET_RTN,
    BUFFER_ATOMIC_FMAX_VBUFFER_ADDR64,
    BUFFER_ATOMIC_FMAX_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_FMAX_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_FMAX_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_FMAX_VBUFFER_IDXEN,
    BUFFER_ATOMIC_FMAX_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_FMAX_VBUFFER_OFFEN,
    BUFFER_ATOMIC_FMAX_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_FMAX_VBUFFER_OFFSET,
    BUFFER_ATOMIC_FMAX_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_FMIN_ADDR64,
    BUFFER_ATOMIC_FMIN_ADDR64_RTN,
    BUFFER_ATOMIC_FMIN_BOTHEN,
    BUFFER_ATOMIC_FMIN_BOTHEN_RTN,
    BUFFER_ATOMIC_FMIN_IDXEN,
    BUFFER_ATOMIC_FMIN_IDXEN_RTN,
    BUFFER_ATOMIC_FMIN_OFFEN,
    BUFFER_ATOMIC_FMIN_OFFEN_RTN,
    BUFFER_ATOMIC_FMIN_OFFSET,
    BUFFER_ATOMIC_FMIN_OFFSET_RTN,
    BUFFER_ATOMIC_FMIN_VBUFFER_ADDR64,
    BUFFER_ATOMIC_FMIN_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_FMIN_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_FMIN_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_FMIN_VBUFFER_IDXEN,
    BUFFER_ATOMIC_FMIN_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_FMIN_VBUFFER_OFFEN,
    BUFFER_ATOMIC_FMIN_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_FMIN_VBUFFER_OFFSET,
    BUFFER_ATOMIC_FMIN_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_INC_ADDR64,
    BUFFER_ATOMIC_INC_ADDR64_RTN,
    BUFFER_ATOMIC_INC_BOTHEN,
    BUFFER_ATOMIC_INC_BOTHEN_RTN,
    BUFFER_ATOMIC_INC_IDXEN,
    BUFFER_ATOMIC_INC_IDXEN_RTN,
    BUFFER_ATOMIC_INC_OFFEN,
    BUFFER_ATOMIC_INC_OFFEN_RTN,
    BUFFER_ATOMIC_INC_OFFSET,
    BUFFER_ATOMIC_INC_OFFSET_RTN,
    BUFFER_ATOMIC_INC_VBUFFER_ADDR64,
    BUFFER_ATOMIC_INC_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_INC_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_INC_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_INC_VBUFFER_IDXEN,
    BUFFER_ATOMIC_INC_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_INC_VBUFFER_OFFEN,
    BUFFER_ATOMIC_INC_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_INC_VBUFFER_OFFSET,
    BUFFER_ATOMIC_INC_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_INC_X2_ADDR64,
    BUFFER_ATOMIC_INC_X2_ADDR64_RTN,
    BUFFER_ATOMIC_INC_X2_BOTHEN,
    BUFFER_ATOMIC_INC_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_INC_X2_IDXEN,
    BUFFER_ATOMIC_INC_X2_IDXEN_RTN,
    BUFFER_ATOMIC_INC_X2_OFFEN,
    BUFFER_ATOMIC_INC_X2_OFFEN_RTN,
    BUFFER_ATOMIC_INC_X2_OFFSET,
    BUFFER_ATOMIC_INC_X2_OFFSET_RTN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_INC_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_INC_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_INC_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_MAX_F64_ADDR64,
    BUFFER_ATOMIC_MAX_F64_ADDR64_RTN,
    BUFFER_ATOMIC_MAX_F64_BOTHEN,
    BUFFER_ATOMIC_MAX_F64_BOTHEN_RTN,
    BUFFER_ATOMIC_MAX_F64_IDXEN,
    BUFFER_ATOMIC_MAX_F64_IDXEN_RTN,
    BUFFER_ATOMIC_MAX_F64_OFFEN,
    BUFFER_ATOMIC_MAX_F64_OFFEN_RTN,
    BUFFER_ATOMIC_MAX_F64_OFFSET,
    BUFFER_ATOMIC_MAX_F64_OFFSET_RTN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_ADDR64,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_IDXEN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_OFFEN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_OFFSET,
    BUFFER_ATOMIC_MAX_F64_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_MIN_F64_ADDR64,
    BUFFER_ATOMIC_MIN_F64_ADDR64_RTN,
    BUFFER_ATOMIC_MIN_F64_BOTHEN,
    BUFFER_ATOMIC_MIN_F64_BOTHEN_RTN,
    BUFFER_ATOMIC_MIN_F64_IDXEN,
    BUFFER_ATOMIC_MIN_F64_IDXEN_RTN,
    BUFFER_ATOMIC_MIN_F64_OFFEN,
    BUFFER_ATOMIC_MIN_F64_OFFEN_RTN,
    BUFFER_ATOMIC_MIN_F64_OFFSET,
    BUFFER_ATOMIC_MIN_F64_OFFSET_RTN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_ADDR64,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_IDXEN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_OFFEN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_OFFSET,
    BUFFER_ATOMIC_MIN_F64_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_OR_ADDR64,
    BUFFER_ATOMIC_OR_ADDR64_RTN,
    BUFFER_ATOMIC_OR_BOTHEN,
    BUFFER_ATOMIC_OR_BOTHEN_RTN,
    BUFFER_ATOMIC_OR_IDXEN,
    BUFFER_ATOMIC_OR_IDXEN_RTN,
    BUFFER_ATOMIC_OR_OFFEN,
    BUFFER_ATOMIC_OR_OFFEN_RTN,
    BUFFER_ATOMIC_OR_OFFSET,
    BUFFER_ATOMIC_OR_OFFSET_RTN,
    BUFFER_ATOMIC_OR_VBUFFER_ADDR64,
    BUFFER_ATOMIC_OR_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_OR_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_OR_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_OR_VBUFFER_IDXEN,
    BUFFER_ATOMIC_OR_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_OR_VBUFFER_OFFEN,
    BUFFER_ATOMIC_OR_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_OR_VBUFFER_OFFSET,
    BUFFER_ATOMIC_OR_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_OR_X2_ADDR64,
    BUFFER_ATOMIC_OR_X2_ADDR64_RTN,
    BUFFER_ATOMIC_OR_X2_BOTHEN,
    BUFFER_ATOMIC_OR_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_OR_X2_IDXEN,
    BUFFER_ATOMIC_OR_X2_IDXEN_RTN,
    BUFFER_ATOMIC_OR_X2_OFFEN,
    BUFFER_ATOMIC_OR_X2_OFFEN_RTN,
    BUFFER_ATOMIC_OR_X2_OFFSET,
    BUFFER_ATOMIC_OR_X2_OFFSET_RTN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_OR_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_OR_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_OR_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_PK_ADD_BF16_ADDR64,
    BUFFER_ATOMIC_PK_ADD_BF16_ADDR64_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_BOTHEN,
    BUFFER_ATOMIC_PK_ADD_BF16_BOTHEN_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_IDXEN,
    BUFFER_ATOMIC_PK_ADD_BF16_IDXEN_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_OFFEN,
    BUFFER_ATOMIC_PK_ADD_BF16_OFFEN_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_OFFSET,
    BUFFER_ATOMIC_PK_ADD_BF16_OFFSET_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_ADDR64,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_IDXEN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_OFFEN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_OFFSET,
    BUFFER_ATOMIC_PK_ADD_BF16_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_PK_ADD_F16_ADDR64,
    BUFFER_ATOMIC_PK_ADD_F16_ADDR64_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_BOTHEN,
    BUFFER_ATOMIC_PK_ADD_F16_BOTHEN_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_IDXEN,
    BUFFER_ATOMIC_PK_ADD_F16_IDXEN_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_OFFEN,
    BUFFER_ATOMIC_PK_ADD_F16_OFFEN_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_OFFSET,
    BUFFER_ATOMIC_PK_ADD_F16_OFFSET_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_ADDR64,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_IDXEN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_OFFEN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_OFFSET,
    BUFFER_ATOMIC_PK_ADD_F16_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SMAX_ADDR64,
    BUFFER_ATOMIC_SMAX_ADDR64_RTN,
    BUFFER_ATOMIC_SMAX_BOTHEN,
    BUFFER_ATOMIC_SMAX_BOTHEN_RTN,
    BUFFER_ATOMIC_SMAX_IDXEN,
    BUFFER_ATOMIC_SMAX_IDXEN_RTN,
    BUFFER_ATOMIC_SMAX_OFFEN,
    BUFFER_ATOMIC_SMAX_OFFEN_RTN,
    BUFFER_ATOMIC_SMAX_OFFSET,
    BUFFER_ATOMIC_SMAX_OFFSET_RTN,
    BUFFER_ATOMIC_SMAX_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SMAX_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SMAX_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SMAX_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SMAX_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SMAX_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SMAX_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SMAX_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SMAX_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SMAX_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SMAX_X2_ADDR64,
    BUFFER_ATOMIC_SMAX_X2_ADDR64_RTN,
    BUFFER_ATOMIC_SMAX_X2_BOTHEN,
    BUFFER_ATOMIC_SMAX_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_SMAX_X2_IDXEN,
    BUFFER_ATOMIC_SMAX_X2_IDXEN_RTN,
    BUFFER_ATOMIC_SMAX_X2_OFFEN,
    BUFFER_ATOMIC_SMAX_X2_OFFEN_RTN,
    BUFFER_ATOMIC_SMAX_X2_OFFSET,
    BUFFER_ATOMIC_SMAX_X2_OFFSET_RTN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SMAX_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SMIN_ADDR64,
    BUFFER_ATOMIC_SMIN_ADDR64_RTN,
    BUFFER_ATOMIC_SMIN_BOTHEN,
    BUFFER_ATOMIC_SMIN_BOTHEN_RTN,
    BUFFER_ATOMIC_SMIN_IDXEN,
    BUFFER_ATOMIC_SMIN_IDXEN_RTN,
    BUFFER_ATOMIC_SMIN_OFFEN,
    BUFFER_ATOMIC_SMIN_OFFEN_RTN,
    BUFFER_ATOMIC_SMIN_OFFSET,
    BUFFER_ATOMIC_SMIN_OFFSET_RTN,
    BUFFER_ATOMIC_SMIN_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SMIN_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SMIN_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SMIN_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SMIN_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SMIN_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SMIN_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SMIN_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SMIN_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SMIN_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SMIN_X2_ADDR64,
    BUFFER_ATOMIC_SMIN_X2_ADDR64_RTN,
    BUFFER_ATOMIC_SMIN_X2_BOTHEN,
    BUFFER_ATOMIC_SMIN_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_SMIN_X2_IDXEN,
    BUFFER_ATOMIC_SMIN_X2_IDXEN_RTN,
    BUFFER_ATOMIC_SMIN_X2_OFFEN,
    BUFFER_ATOMIC_SMIN_X2_OFFEN_RTN,
    BUFFER_ATOMIC_SMIN_X2_OFFSET,
    BUFFER_ATOMIC_SMIN_X2_OFFSET_RTN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SMIN_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SUB_ADDR64,
    BUFFER_ATOMIC_SUB_ADDR64_RTN,
    BUFFER_ATOMIC_SUB_BOTHEN,
    BUFFER_ATOMIC_SUB_BOTHEN_RTN,
    BUFFER_ATOMIC_SUB_IDXEN,
    BUFFER_ATOMIC_SUB_IDXEN_RTN,
    BUFFER_ATOMIC_SUB_OFFEN,
    BUFFER_ATOMIC_SUB_OFFEN_RTN,
    BUFFER_ATOMIC_SUB_OFFSET,
    BUFFER_ATOMIC_SUB_OFFSET_RTN,
    BUFFER_ATOMIC_SUB_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SUB_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SUB_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SUB_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SUB_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SUB_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SUB_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SUB_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SUB_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SUB_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SUB_X2_ADDR64,
    BUFFER_ATOMIC_SUB_X2_ADDR64_RTN,
    BUFFER_ATOMIC_SUB_X2_BOTHEN,
    BUFFER_ATOMIC_SUB_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_SUB_X2_IDXEN,
    BUFFER_ATOMIC_SUB_X2_IDXEN_RTN,
    BUFFER_ATOMIC_SUB_X2_OFFEN,
    BUFFER_ATOMIC_SUB_X2_OFFEN_RTN,
    BUFFER_ATOMIC_SUB_X2_OFFSET,
    BUFFER_ATOMIC_SUB_X2_OFFSET_RTN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SUB_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SWAP_ADDR64,
    BUFFER_ATOMIC_SWAP_ADDR64_RTN,
    BUFFER_ATOMIC_SWAP_BOTHEN,
    BUFFER_ATOMIC_SWAP_BOTHEN_RTN,
    BUFFER_ATOMIC_SWAP_IDXEN,
    BUFFER_ATOMIC_SWAP_IDXEN_RTN,
    BUFFER_ATOMIC_SWAP_OFFEN,
    BUFFER_ATOMIC_SWAP_OFFEN_RTN,
    BUFFER_ATOMIC_SWAP_OFFSET,
    BUFFER_ATOMIC_SWAP_OFFSET_RTN,
    BUFFER_ATOMIC_SWAP_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SWAP_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SWAP_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SWAP_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SWAP_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SWAP_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SWAP_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SWAP_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SWAP_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SWAP_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_SWAP_X2_ADDR64,
    BUFFER_ATOMIC_SWAP_X2_ADDR64_RTN,
    BUFFER_ATOMIC_SWAP_X2_BOTHEN,
    BUFFER_ATOMIC_SWAP_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_SWAP_X2_IDXEN,
    BUFFER_ATOMIC_SWAP_X2_IDXEN_RTN,
    BUFFER_ATOMIC_SWAP_X2_OFFEN,
    BUFFER_ATOMIC_SWAP_X2_OFFEN_RTN,
    BUFFER_ATOMIC_SWAP_X2_OFFSET,
    BUFFER_ATOMIC_SWAP_X2_OFFSET_RTN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_SWAP_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_UMAX_ADDR64,
    BUFFER_ATOMIC_UMAX_ADDR64_RTN,
    BUFFER_ATOMIC_UMAX_BOTHEN,
    BUFFER_ATOMIC_UMAX_BOTHEN_RTN,
    BUFFER_ATOMIC_UMAX_IDXEN,
    BUFFER_ATOMIC_UMAX_IDXEN_RTN,
    BUFFER_ATOMIC_UMAX_OFFEN,
    BUFFER_ATOMIC_UMAX_OFFEN_RTN,
    BUFFER_ATOMIC_UMAX_OFFSET,
    BUFFER_ATOMIC_UMAX_OFFSET_RTN,
    BUFFER_ATOMIC_UMAX_VBUFFER_ADDR64,
    BUFFER_ATOMIC_UMAX_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_UMAX_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_UMAX_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_UMAX_VBUFFER_IDXEN,
    BUFFER_ATOMIC_UMAX_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_UMAX_VBUFFER_OFFEN,
    BUFFER_ATOMIC_UMAX_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_UMAX_VBUFFER_OFFSET,
    BUFFER_ATOMIC_UMAX_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_UMAX_X2_ADDR64,
    BUFFER_ATOMIC_UMAX_X2_ADDR64_RTN,
    BUFFER_ATOMIC_UMAX_X2_BOTHEN,
    BUFFER_ATOMIC_UMAX_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_UMAX_X2_IDXEN,
    BUFFER_ATOMIC_UMAX_X2_IDXEN_RTN,
    BUFFER_ATOMIC_UMAX_X2_OFFEN,
    BUFFER_ATOMIC_UMAX_X2_OFFEN_RTN,
    BUFFER_ATOMIC_UMAX_X2_OFFSET,
    BUFFER_ATOMIC_UMAX_X2_OFFSET_RTN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_UMAX_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_UMIN_ADDR64,
    BUFFER_ATOMIC_UMIN_ADDR64_RTN,
    BUFFER_ATOMIC_UMIN_BOTHEN,
    BUFFER_ATOMIC_UMIN_BOTHEN_RTN,
    BUFFER_ATOMIC_UMIN_IDXEN,
    BUFFER_ATOMIC_UMIN_IDXEN_RTN,
    BUFFER_ATOMIC_UMIN_OFFEN,
    BUFFER_ATOMIC_UMIN_OFFEN_RTN,
    BUFFER_ATOMIC_UMIN_OFFSET,
    BUFFER_ATOMIC_UMIN_OFFSET_RTN,
    BUFFER_ATOMIC_UMIN_VBUFFER_ADDR64,
    BUFFER_ATOMIC_UMIN_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_UMIN_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_UMIN_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_UMIN_VBUFFER_IDXEN,
    BUFFER_ATOMIC_UMIN_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_UMIN_VBUFFER_OFFEN,
    BUFFER_ATOMIC_UMIN_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_UMIN_VBUFFER_OFFSET,
    BUFFER_ATOMIC_UMIN_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_UMIN_X2_ADDR64,
    BUFFER_ATOMIC_UMIN_X2_ADDR64_RTN,
    BUFFER_ATOMIC_UMIN_X2_BOTHEN,
    BUFFER_ATOMIC_UMIN_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_UMIN_X2_IDXEN,
    BUFFER_ATOMIC_UMIN_X2_IDXEN_RTN,
    BUFFER_ATOMIC_UMIN_X2_OFFEN,
    BUFFER_ATOMIC_UMIN_X2_OFFEN_RTN,
    BUFFER_ATOMIC_UMIN_X2_OFFSET,
    BUFFER_ATOMIC_UMIN_X2_OFFSET_RTN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_UMIN_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_XOR_ADDR64,
    BUFFER_ATOMIC_XOR_ADDR64_RTN,
    BUFFER_ATOMIC_XOR_BOTHEN,
    BUFFER_ATOMIC_XOR_BOTHEN_RTN,
    BUFFER_ATOMIC_XOR_IDXEN,
    BUFFER_ATOMIC_XOR_IDXEN_RTN,
    BUFFER_ATOMIC_XOR_OFFEN,
    BUFFER_ATOMIC_XOR_OFFEN_RTN,
    BUFFER_ATOMIC_XOR_OFFSET,
    BUFFER_ATOMIC_XOR_OFFSET_RTN,
    BUFFER_ATOMIC_XOR_VBUFFER_ADDR64,
    BUFFER_ATOMIC_XOR_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_XOR_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_XOR_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_XOR_VBUFFER_IDXEN,
    BUFFER_ATOMIC_XOR_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_XOR_VBUFFER_OFFEN,
    BUFFER_ATOMIC_XOR_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_XOR_VBUFFER_OFFSET,
    BUFFER_ATOMIC_XOR_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

foreach inst = [
    BUFFER_ATOMIC_XOR_X2_ADDR64,
    BUFFER_ATOMIC_XOR_X2_ADDR64_RTN,
    BUFFER_ATOMIC_XOR_X2_BOTHEN,
    BUFFER_ATOMIC_XOR_X2_BOTHEN_RTN,
    BUFFER_ATOMIC_XOR_X2_IDXEN,
    BUFFER_ATOMIC_XOR_X2_IDXEN_RTN,
    BUFFER_ATOMIC_XOR_X2_OFFEN,
    BUFFER_ATOMIC_XOR_X2_OFFEN_RTN,
    BUFFER_ATOMIC_XOR_X2_OFFSET,
    BUFFER_ATOMIC_XOR_X2_OFFSET_RTN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_ADDR64,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_ADDR64_RTN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_BOTHEN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_BOTHEN_RTN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_IDXEN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_IDXEN_RTN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_OFFEN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_OFFEN_RTN,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_OFFSET,
    BUFFER_ATOMIC_XOR_X2_VBUFFER_OFFSET_RTN
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

// Buffer cache management instructions.

foreach inst = [
    BUFFER_GL0_INV,
    BUFFER_GL1_INV,
    BUFFER_INV,
    BUFFER_INVL2,
    BUFFER_WBINVL1,
    BUFFER_WBINVL1_SC,
    BUFFER_WBINVL1_VOL,
    BUFFER_WBL2
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

