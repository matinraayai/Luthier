//===- DSSem.td ---------------------------------------------------*- tblgen -*-===//
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

/// Calculate a single-offset DS address: IntToPtr(addr) + offset
/// The pointer type is local (LDS, AS 3) or region (GDS, AS 2) depending on
/// the gds modifier.
class CalculateSingleOffsetDSAddr<dag addr, dag offset,
                                  LLVMQualPointerType localOrRegionPtr> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr addr, localOrRegionPtr), offset);
}

/// Calculate a double-offset DS address for READ2/WRITE2 instructions.
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

/// Helper: select LDS (local_ptr_ty, AS 3) or GDS (region_ptr_type, AS 2)
/// based on the gds modifier.
class DSAddr<dag addr, dag offset> {
  dag ret = CalculateSingleOffsetDSAddr<addr, offset,
    (SelectOperand (hasModifierSet $gds), region_ptr_type, local_ptr_ty)>.ret;
}

//------------------------------------------------------------------------------
// DS 32-bit Atomic Operations (no return)
// tmp = MEM[addr]; MEM[addr] = OP(tmp, data)
//------------------------------------------------------------------------------

// Sources (present in all architectures):
//   gcn3-instruction-set-architecture.pdf, page 267.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 184.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 175.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 173.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 418.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 465.
//   rdna-shader-instruction-set-architecture.pdf, page 201.
//   rdna2-shader-instruction-set-architecture.pdf, page 196.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 511.
//   rdna4-instruction-set-architecture.pdf, page 585.

// DS_ADD_U32: tmp = MEM[addr]; MEM[addr] = tmp + data.
def DS_ADD_U32_sem : InstSISemantic<DS_ADD_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAdd (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_SUB_U32: tmp = MEM[addr]; MEM[addr] = tmp - data.
def DS_SUB_U32_sem : InstSISemantic<DS_SUB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_RSUB_U32: tmp = MEM[addr]; MEM[addr] = data - tmp.
def DS_RSUB_U32_sem : InstSISemantic<DS_RSUB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetNamedOperand $data), (GetVal $tmp)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_INC_U32: tmp = MEM[addr]; MEM[addr] = (tmp >= data) ? 0 : tmp + 1.
def DS_INC_U32_sem : InstSISemantic<DS_INC_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETUGE, (GetVal $tmp), (GetNamedOperand $data)),
                            (ConstantInt llvm_i32_ty, 0),
                            (LLVMAdd (GetVal $tmp), (ConstantInt llvm_i32_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_DEC_U32: tmp = MEM[addr]; MEM[addr] = (tmp == 0 || tmp > data) ? data : tmp - 1.
// Sources: same pages as DS_ADD_U32, offset by +1 page in most PDFs.
def DS_DEC_U32_sem : InstSISemantic<DS_DEC_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMOr (LLVMCmp SETEQ, (GetVal $tmp), (ConstantInt llvm_i32_ty, 0)),
                                    (LLVMCmp SETUGT, (GetVal $tmp), (GetNamedOperand $data))),
                            (GetNamedOperand $data),
                            (LLVMSub (GetVal $tmp), (ConstantInt llvm_i32_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MIN_I32: tmp = MEM[addr]; MEM[addr] = min(tmp, data) (signed).
def DS_MIN_I32_sem : InstSISemantic<DS_MIN_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETLT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_I32: tmp = MEM[addr]; MEM[addr] = max(tmp, data) (signed).
def DS_MAX_I32_sem : InstSISemantic<DS_MAX_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETGT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MIN_U32: tmp = MEM[addr]; MEM[addr] = min(tmp, data) (unsigned).
def DS_MIN_U32_sem : InstSISemantic<DS_MIN_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETULT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_U32: tmp = MEM[addr]; MEM[addr] = max(tmp, data) (unsigned).
def DS_MAX_U32_sem : InstSISemantic<DS_MAX_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETUGT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_AND_B32: tmp = MEM[addr]; MEM[addr] = tmp & data.
def DS_AND_B32_sem : InstSISemantic<DS_AND_B32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAnd (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_OR_B32: tmp = MEM[addr]; MEM[addr] = tmp | data.
def DS_OR_B32_sem : InstSISemantic<DS_OR_B32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_XOR_B32: tmp = MEM[addr]; MEM[addr] = tmp ^ data.
def DS_XOR_B32_sem : InstSISemantic<DS_XOR_B32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMXor (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MSKOR_B32: tmp = MEM[addr]; MEM[addr] = (tmp & ~data0) | data1.
// Masked store: data0 is mask, data1 is value.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 267.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 176.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 174.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 421.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 467.
//   rdna-shader-instruction-set-architecture.pdf, page 203.
//   rdna2-shader-instruction-set-architecture.pdf, page 197.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 513.
//   rdna4-instruction-set-architecture.pdf, page 588.
def DS_MSKOR_B32_sem : InstSISemantic<DS_MSKOR_B32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (LLVMAnd (GetVal $tmp),
                                  (LLVMXor (GetNamedOperand $data0), (ConstantInt llvm_i32_ty, -1))),
                        (GetNamedOperand $data1)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

//------------------------------------------------------------------------------
// DS Float Atomic Operations (no return)
//------------------------------------------------------------------------------

// DS_ADD_F32: tmp = MEM[addr]; MEM[addr] = as_float(tmp) + as_float(data).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 187.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 175.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 424.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 471.
//   rdna-shader-instruction-set-architecture.pdf, page 204.
//   rdna2-shader-instruction-set-architecture.pdf, page 198.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 516.
//   rdna4-instruction-set-architecture.pdf, page 590.
def DS_ADD_F32_sem : InstSISemantic<DS_ADD_F32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMFAdd (LLVMBitCast (GetVal $tmp), llvm_float_ty),
                                       (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)),
                             llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MIN_F32: tmp = MEM[addr]; MEM[addr] = min(as_float(tmp), as_float(data)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 423.
//   rdna4-instruction-set-architecture.pdf, page 589 (as DS_MIN_NUM_F32).
def DS_MIN_F32_sem : InstSISemantic<DS_MIN_F32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp SETOLT,
                  (LLVMBitCast (GetVal $tmp), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)),
                (LLVMBitCast (GetVal $tmp), llvm_float_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_F32: tmp = MEM[addr]; MEM[addr] = max(as_float(tmp), as_float(data)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 187.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 423.
//   rdna4-instruction-set-architecture.pdf, page 590 (as DS_MAX_NUM_F32).
def DS_MAX_F32_sem : InstSISemantic<DS_MAX_F32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp SETOGT,
                  (LLVMBitCast (GetVal $tmp), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)),
                (LLVMBitCast (GetVal $tmp), llvm_float_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

//------------------------------------------------------------------------------
// DS Write (Store) Instructions
//------------------------------------------------------------------------------

// DS_WRITE_B32: MEM[addr] = data.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 267.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 176.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 174.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 421.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 468.
//   rdna-shader-instruction-set-architecture.pdf, page 203.
//   rdna2-shader-instruction-set-architecture.pdf, page 197.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 514 (as DS_STORE_B32).
//   rdna4-instruction-set-architecture.pdf, page 588 (as DS_STORE_B32).
def DS_WRITE_B32_sem : InstSISemantic<DS_WRITE_B32> {
  let Semantic = [
    (LLVMStore (GetNamedOperand $data0),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_WRITE_B64: MEM[addr] = data (64-bit).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 269.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 194.
//   rdna-shader-instruction-set-architecture.pdf, page 214.
//   rdna2-shader-instruction-set-architecture.pdf, page 208.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 530 (as DS_STORE_B64).
//   rdna4-instruction-set-architecture.pdf, page 605 (as DS_STORE_B64).
def DS_WRITE_B64_sem : InstSISemantic<DS_WRITE_B64> {
  let Semantic = [
    (LLVMStore (GetNamedOperand $data0),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_WRITE_B8: MEM[addr] = data[7:0] (byte store).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 187.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 425.
//   rdna4-instruction-set-architecture.pdf, page 591 (as DS_STORE_B8).
def DS_WRITE_B8_sem : InstSISemantic<DS_WRITE_B8> {
  let Semantic = [
    (LLVMStore (LLVMTrunc (GetNamedOperand $data0), llvm_i8_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_WRITE_B16: MEM[addr] = data[15:0] (halfword store).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 187.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 425.
//   rdna4-instruction-set-architecture.pdf, page 591 (as DS_STORE_B16).
def DS_WRITE_B16_sem : InstSISemantic<DS_WRITE_B16> {
  let Semantic = [
    (LLVMStore (LLVMTrunc (GetNamedOperand $data0), llvm_i16_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

//------------------------------------------------------------------------------
// DS Read (Load) Instructions
//------------------------------------------------------------------------------

// DS_READ_B32: D = MEM[addr].
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 190.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 178.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 432.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 478.
//   rdna-shader-instruction-set-architecture.pdf, page 209.
//   rdna2-shader-instruction-set-architecture.pdf, page 203.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 524 (as DS_LOAD_B32).
//   rdna4-instruction-set-architecture.pdf, page 599 (as DS_LOAD_B32).
def DS_READ_B32_sem : InstSISemantic<DS_READ_B32> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMLoad llvm_i32_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret))
  ];
}

// DS_READ_B64: D = MEM[addr] (64-bit).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 270.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 198.
//   rdna-shader-instruction-set-architecture.pdf, page 217.
//   rdna2-shader-instruction-set-architecture.pdf, page 211.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 536 (as DS_LOAD_B64).
//   rdna4-instruction-set-architecture.pdf, page 612 (as DS_LOAD_B64).
def DS_READ_B64_sem : InstSISemantic<DS_READ_B64> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMLoad llvm_i64_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret))
  ];
}

// DS_READ_U8: D = zext(MEM[addr][7:0]).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 190.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 432.
//   rdna4-instruction-set-architecture.pdf, page 600 (as DS_LOAD_U8).
def DS_READ_U8_sem : InstSISemantic<DS_READ_U8> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMZExt (LLVMLoad llvm_i8_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), llvm_i32_ty))
  ];
}

// DS_READ_I8: D = sext(MEM[addr][7:0]).
def DS_READ_I8_sem : InstSISemantic<DS_READ_I8> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMSExt (LLVMLoad llvm_i8_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), llvm_i32_ty))
  ];
}

// DS_READ_U16: D = zext(MEM[addr][15:0]).
def DS_READ_U16_sem : InstSISemantic<DS_READ_U16> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMZExt (LLVMLoad llvm_i16_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), llvm_i32_ty))
  ];
}

// DS_READ_I16: D = sext(MEM[addr][15:0]).
def DS_READ_I16_sem : InstSISemantic<DS_READ_I16> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMSExt (LLVMLoad llvm_i16_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// DS 32-bit Atomic Operations (with return)
// D = MEM[addr]; MEM[addr] = OP(D, data)
// The return value is the ORIGINAL memory value before the operation.
//------------------------------------------------------------------------------

// Sources (present in all architectures):
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 187.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 176.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 425.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 472.
//   rdna-shader-instruction-set-architecture.pdf, page 206.
//   rdna2-shader-instruction-set-architecture.pdf, page 201.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 517.
//   rdna4-instruction-set-architecture.pdf, page 591.

// DS_ADD_RTN_U32: D = MEM[addr]; MEM[addr] += data.
def DS_ADD_RTN_U32_sem : InstSISemantic<DS_ADD_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAdd (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_SUB_RTN_U32: D = MEM[addr]; MEM[addr] -= data.
def DS_SUB_RTN_U32_sem : InstSISemantic<DS_SUB_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_RSUB_RTN_U32: D = MEM[addr]; MEM[addr] = data - D.
def DS_RSUB_RTN_U32_sem : InstSISemantic<DS_RSUB_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetNamedOperand $data), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_INC_RTN_U32: D = MEM[addr]; MEM[addr] = (D >= data) ? 0 : D + 1.
def DS_INC_RTN_U32_sem : InstSISemantic<DS_INC_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETUGE, (GetVal $old), (GetNamedOperand $data)),
                            (ConstantInt llvm_i32_ty, 0),
                            (LLVMAdd (GetVal $old), (ConstantInt llvm_i32_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_DEC_RTN_U32: D = MEM[addr]; MEM[addr] = (D==0 || D>data) ? data : D-1.
def DS_DEC_RTN_U32_sem : InstSISemantic<DS_DEC_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMOr (LLVMCmp SETEQ, (GetVal $old), (ConstantInt llvm_i32_ty, 0)),
                                    (LLVMCmp SETUGT, (GetVal $old), (GetNamedOperand $data))),
                            (GetNamedOperand $data),
                            (LLVMSub (GetVal $old), (ConstantInt llvm_i32_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MIN_RTN_I32: D = MEM[addr]; MEM[addr] = min(D, data) (signed).
def DS_MIN_RTN_I32_sem : InstSISemantic<DS_MIN_RTN_I32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETLT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MAX_RTN_I32: D = MEM[addr]; MEM[addr] = max(D, data) (signed).
def DS_MAX_RTN_I32_sem : InstSISemantic<DS_MAX_RTN_I32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETGT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MIN_RTN_U32: D = MEM[addr]; MEM[addr] = min(D, data) (unsigned).
def DS_MIN_RTN_U32_sem : InstSISemantic<DS_MIN_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETULT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MAX_RTN_U32: D = MEM[addr]; MEM[addr] = max(D, data) (unsigned).
def DS_MAX_RTN_U32_sem : InstSISemantic<DS_MAX_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETUGT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_AND_RTN_B32: D = MEM[addr]; MEM[addr] &= data.
def DS_AND_RTN_B32_sem : InstSISemantic<DS_AND_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAnd (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_OR_RTN_B32: D = MEM[addr]; MEM[addr] |= data.
def DS_OR_RTN_B32_sem : InstSISemantic<DS_OR_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_XOR_RTN_B32: D = MEM[addr]; MEM[addr] ^= data.
def DS_XOR_RTN_B32_sem : InstSISemantic<DS_XOR_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMXor (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_WRXCHG_RTN_B32: D = MEM[addr]; MEM[addr] = data (exchange).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 188.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 429.
//   rdna4-instruction-set-architecture.pdf, page 594 (as DS_STOREXCHG_RTN_B32).
def DS_WRXCHG_RTN_B32_sem : InstSISemantic<DS_WRXCHG_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $data),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS Compare-and-Store Instructions
//------------------------------------------------------------------------------

// DS_CMPST_B32: if (MEM[addr] == data0) MEM[addr] = data1.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 422.
//   rdna-shader-instruction-set-architecture.pdf, page 203.
//   rdna2-shader-instruction-set-architecture.pdf, page 198.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 514 (as DS_CMPSTORE_B32).
//   rdna4-instruction-set-architecture.pdf, page 589 (as DS_CMPSTORE_B32).
def DS_CMPST_B32_sem : InstSISemantic<DS_CMPST_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPST_RTN_B32: D = MEM[addr]; if (D == data0) MEM[addr] = data1.
def DS_CMPST_RTN_B32_sem : InstSISemantic<DS_CMPST_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp SETEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_CMPST_F32: if (as_float(MEM[addr]) == as_float(data0)) MEM[addr] = data1.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 422.
def DS_CMPST_F32_sem : InstSISemantic<DS_CMPST_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp SETOEQ,
                  (LLVMBitCast (GetVal $old), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_float_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPST_RTN_F32: D = MEM[addr]; if (as_float(D) == as_float(data0)) MEM[addr] = data1.
def DS_CMPST_RTN_F32_sem : InstSISemantic<DS_CMPST_RTN_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp SETOEQ,
                  (LLVMBitCast (GetVal $old), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_float_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS Swizzle / Permute Instructions
//------------------------------------------------------------------------------

// DS_SWIZZLE_B32: Swizzle data between lanes without using LDS memory.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 269.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 190.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 178.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 433.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 480.
//   rdna-shader-instruction-set-architecture.pdf, page 209.
//   rdna2-shader-instruction-set-architecture.pdf, page 203.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 522.
//   rdna4-instruction-set-architecture.pdf, page 597.
def DS_SWIZZLE_B32_sem : InstSIIntrinsic<DS_SWIZZLE_B32, int_amdgcn_ds_swizzle>;

// DS_PERMUTE_B32: Forward permute. D[lane] = data[offset[lane]].
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 269.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 191.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 179.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 435.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 482.
//   rdna-shader-instruction-set-architecture.pdf, page 219.
//   rdna2-shader-instruction-set-architecture.pdf, page 213.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 543.
//   rdna4-instruction-set-architecture.pdf, page 618.
def DS_PERMUTE_B32_sem : InstSIIntrinsic<DS_PERMUTE_B32, int_amdgcn_ds_permute>;

// DS_BPERMUTE_B32: Backward permute. D[lane] = data[offset[lane]].
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 269.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 192.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 180.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 437.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 483.
//   rdna-shader-instruction-set-architecture.pdf, page 220.
//   rdna2-shader-instruction-set-architecture.pdf, page 214.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 544.
//   rdna4-instruction-set-architecture.pdf, page 618.
def DS_BPERMUTE_B32_sem : InstSIIntrinsic<DS_BPERMUTE_B32, int_amdgcn_ds_bpermute>;

//------------------------------------------------------------------------------
// DS NOP
//------------------------------------------------------------------------------

// DS_NOP: No operation.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 187.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 175.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 423.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 470.
//   rdna-shader-instruction-set-architecture.pdf, page 204.
//   rdna2-shader-instruction-set-architecture.pdf, page 198.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 516.
//   rdna4-instruction-set-architecture.pdf, page 590.
def DS_NOP_sem : InstSISemantic<DS_NOP>;

//------------------------------------------------------------------------------
// DS Append / Consume
//------------------------------------------------------------------------------

// DS_APPEND: Append a dword to a buffer. D = MEM[GDS_BASE]; GDS_BASE += 4.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 271.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 203.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 191.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 188.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 456.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 500.
//   rdna-shader-instruction-set-architecture.pdf, page 211.
//   rdna2-shader-instruction-set-architecture.pdf, page 205.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 526.
//   rdna4-instruction-set-architecture.pdf, page 601.
def DS_APPEND_sem : InstSIIntrinsic<DS_APPEND, int_amdgcn_ds_append>;

// DS_CONSUME: Consume a dword from a buffer. GDS_BASE -= 4; D = MEM[GDS_BASE].
// Sources: same as DS_APPEND (same pages or adjacent).
def DS_CONSUME_sem : InstSIIntrinsic<DS_CONSUME, int_amdgcn_ds_consume>;
