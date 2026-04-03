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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $data)),
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
    (LLVMStore (LLVMSelect (LLVMOr (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (ConstantInt llvm_i32_ty, 0)),
                                    (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $data))),
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_I32: tmp = MEM[addr]; MEM[addr] = max(tmp, data) (signed).
def DS_MAX_I32_sem : InstSISemantic<DS_MAX_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MIN_U32: tmp = MEM[addr]; MEM[addr] = min(tmp, data) (unsigned).
def DS_MIN_U32_sem : InstSISemantic<DS_MIN_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_U32: tmp = MEM[addr]; MEM[addr] = max(tmp, data) (unsigned).
def DS_MAX_U32_sem : InstSISemantic<DS_MAX_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $data)),
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 423.
//   rdna4-instruction-set-architecture.pdf, page 589 (as DS_MIN_NUM_F32).
def DS_MIN_F32_sem : InstSISemantic<DS_MIN_F32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOLT,
                  (LLVMBitCast (GetVal $tmp), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)),
                (LLVMBitCast (GetVal $tmp), llvm_float_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_F32: tmp = MEM[addr]; MEM[addr] = max(as_float(tmp), as_float(data)).
// Sources:
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 187.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 423.
//   rdna4-instruction-set-architecture.pdf, page 590 (as DS_MAX_NUM_F32).
def DS_MAX_F32_sem : InstSISemantic<DS_MAX_F32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOGT,
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $old), (GetNamedOperand $data)),
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
    (LLVMStore (LLVMSelect (LLVMOr (LLVMCmp LLVMICmpEQ, (GetVal $old), (ConstantInt llvm_i32_ty, 0)),
                                    (LLVMCmp LLVMICmpUGT, (GetVal $old), (GetNamedOperand $data))),
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $old), (GetNamedOperand $data)),
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $old), (GetNamedOperand $data)),
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $old), (GetNamedOperand $data)),
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $old), (GetNamedOperand $data)),
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPST_RTN_B32: D = MEM[addr]; if (D == data0) MEM[addr] = data1.
def DS_CMPST_RTN_B32_sem : InstSISemantic<DS_CMPST_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_CMPST_F32: if (as_float(MEM[addr]) == as_float(data0)) MEM[addr] = data1.
// Sources:
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 422.
def DS_CMPST_F32_sem : InstSISemantic<DS_CMPST_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
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
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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
//   southern-islands-instruction-set-architecture.pdf, pages 230-237.
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
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

//------------------------------------------------------------------------------
// CI+ DS Instructions
//------------------------------------------------------------------------------

// DS_GWS_SEMA_RELEASE_ALL: GWS semaphore release all (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
def DS_GWS_SEMA_RELEASE_ALL_sem : InstSISemantic<DS_GWS_SEMA_RELEASE_ALL> {
  let Semantic = [];
}

// DS_WRAP_RTN_B32: Wrapped exchange (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
def DS_WRAP_RTN_B32_sem : InstSISemantic<DS_WRAP_RTN_B32> {
  let Semantic = [];
}

// DS_CONDXCHG32_RTN_B64: Conditional exchange (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
def DS_CONDXCHG32_RTN_B64_sem : InstSISemantic<DS_CONDXCHG32_RTN_B64> {
  let Semantic = [];
}

// DS_WRITE_B96: 96-bit write (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
def DS_WRITE_B96_sem : InstSISemantic<DS_WRITE_B96> {
  let Semantic = [];
}

// DS_WRITE_B128: 128-bit write (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
def DS_WRITE_B128_sem : InstSISemantic<DS_WRITE_B128> {
  let Semantic = [];
}

// DS_READ_B96: 96-bit read (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
def DS_READ_B96_sem : InstSISemantic<DS_READ_B96> {
  let Semantic = [];
}

// DS_READ_B128: 128-bit read (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
def DS_READ_B128_sem : InstSISemantic<DS_READ_B128> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// DS 64-bit Atomic Operations (no return)
// tmp = MEM[addr]; MEM[addr] = OP(tmp, data)
//------------------------------------------------------------------------------

// DS_ADD_U64: tmp = MEM[addr]; MEM[addr] = tmp + data (64-bit).
def DS_ADD_U64_sem : InstSISemantic<DS_ADD_U64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAdd (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_SUB_U64: tmp = MEM[addr]; MEM[addr] = tmp - data (64-bit).
def DS_SUB_U64_sem : InstSISemantic<DS_SUB_U64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_RSUB_U64: tmp = MEM[addr]; MEM[addr] = data - tmp (64-bit).
def DS_RSUB_U64_sem : InstSISemantic<DS_RSUB_U64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetNamedOperand $data), (GetVal $tmp)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_INC_U64: tmp = MEM[addr]; MEM[addr] = (tmp >= data) ? 0 : tmp + 1 (64-bit).
def DS_INC_U64_sem : InstSISemantic<DS_INC_U64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $data)),
                            (ConstantInt llvm_i64_ty, 0),
                            (LLVMAdd (GetVal $tmp), (ConstantInt llvm_i64_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_DEC_U64: tmp = MEM[addr]; MEM[addr] = (tmp==0 || tmp>data) ? data : tmp-1 (64-bit).
def DS_DEC_U64_sem : InstSISemantic<DS_DEC_U64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMOr (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (ConstantInt llvm_i64_ty, 0)),
                                    (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $data))),
                            (GetNamedOperand $data),
                            (LLVMSub (GetVal $tmp), (ConstantInt llvm_i64_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MIN_I64: tmp = MEM[addr]; MEM[addr] = min(tmp, data) (signed, 64-bit).
def DS_MIN_I64_sem : InstSISemantic<DS_MIN_I64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_I64: tmp = MEM[addr]; MEM[addr] = max(tmp, data) (signed, 64-bit).
def DS_MAX_I64_sem : InstSISemantic<DS_MAX_I64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MIN_U64: tmp = MEM[addr]; MEM[addr] = min(tmp, data) (unsigned, 64-bit).
def DS_MIN_U64_sem : InstSISemantic<DS_MIN_U64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_U64: tmp = MEM[addr]; MEM[addr] = max(tmp, data) (unsigned, 64-bit).
def DS_MAX_U64_sem : InstSISemantic<DS_MAX_U64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_AND_B64: tmp = MEM[addr]; MEM[addr] = tmp & data (64-bit).
def DS_AND_B64_sem : InstSISemantic<DS_AND_B64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAnd (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_OR_B64: tmp = MEM[addr]; MEM[addr] = tmp | data (64-bit).
def DS_OR_B64_sem : InstSISemantic<DS_OR_B64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_XOR_B64: tmp = MEM[addr]; MEM[addr] = tmp ^ data (64-bit).
def DS_XOR_B64_sem : InstSISemantic<DS_XOR_B64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMXor (GetVal $tmp), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MSKOR_B64: tmp = MEM[addr]; MEM[addr] = (tmp & ~data0) | data1 (64-bit).
def DS_MSKOR_B64_sem : InstSISemantic<DS_MSKOR_B64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (LLVMAnd (GetVal $tmp),
                                  (LLVMXor (GetNamedOperand $data0), (ConstantInt llvm_i64_ty, -1))),
                        (GetNamedOperand $data1)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

//------------------------------------------------------------------------------
// DS 64-bit Atomic Operations (with return)
// D = MEM[addr]; MEM[addr] = OP(D, data); return old value.
//------------------------------------------------------------------------------

// DS_ADD_RTN_U64: D = MEM[addr]; MEM[addr] += data (64-bit).
def DS_ADD_RTN_U64_sem : InstSISemantic<DS_ADD_RTN_U64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAdd (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_SUB_RTN_U64: D = MEM[addr]; MEM[addr] -= data (64-bit).
def DS_SUB_RTN_U64_sem : InstSISemantic<DS_SUB_RTN_U64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_RSUB_RTN_U64: D = MEM[addr]; MEM[addr] = data - D (64-bit).
def DS_RSUB_RTN_U64_sem : InstSISemantic<DS_RSUB_RTN_U64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSub (GetNamedOperand $data), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_INC_RTN_U64: D = MEM[addr]; MEM[addr] = (D >= data) ? 0 : D + 1 (64-bit).
def DS_INC_RTN_U64_sem : InstSISemantic<DS_INC_RTN_U64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $old), (GetNamedOperand $data)),
                            (ConstantInt llvm_i64_ty, 0),
                            (LLVMAdd (GetVal $old), (ConstantInt llvm_i64_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_DEC_RTN_U64: D = MEM[addr]; MEM[addr] = (D==0 || D>data) ? data : D-1 (64-bit).
def DS_DEC_RTN_U64_sem : InstSISemantic<DS_DEC_RTN_U64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMOr (LLVMCmp LLVMICmpEQ, (GetVal $old), (ConstantInt llvm_i64_ty, 0)),
                                    (LLVMCmp LLVMICmpUGT, (GetVal $old), (GetNamedOperand $data))),
                            (GetNamedOperand $data),
                            (LLVMSub (GetVal $old), (ConstantInt llvm_i64_ty, 1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MIN_RTN_I64: D = MEM[addr]; MEM[addr] = min(D, data) (signed, 64-bit).
def DS_MIN_RTN_I64_sem : InstSISemantic<DS_MIN_RTN_I64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MAX_RTN_I64: D = MEM[addr]; MEM[addr] = max(D, data) (signed, 64-bit).
def DS_MAX_RTN_I64_sem : InstSISemantic<DS_MAX_RTN_I64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MIN_RTN_U64: D = MEM[addr]; MEM[addr] = min(D, data) (unsigned, 64-bit).
def DS_MIN_RTN_U64_sem : InstSISemantic<DS_MIN_RTN_U64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MAX_RTN_U64: D = MEM[addr]; MEM[addr] = max(D, data) (unsigned, 64-bit).
def DS_MAX_RTN_U64_sem : InstSISemantic<DS_MAX_RTN_U64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_AND_RTN_B64: D = MEM[addr]; MEM[addr] &= data (64-bit).
def DS_AND_RTN_B64_sem : InstSISemantic<DS_AND_RTN_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMAnd (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_OR_RTN_B64: D = MEM[addr]; MEM[addr] |= data (64-bit).
def DS_OR_RTN_B64_sem : InstSISemantic<DS_OR_RTN_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_XOR_RTN_B64: D = MEM[addr]; MEM[addr] ^= data (64-bit).
def DS_XOR_RTN_B64_sem : InstSISemantic<DS_XOR_RTN_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMXor (GetVal $old), (GetNamedOperand $data)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MSKOR_RTN_B64: D = MEM[addr]; MEM[addr] = (D & ~data0) | data1 (64-bit).
def DS_MSKOR_RTN_B64_sem : InstSISemantic<DS_MSKOR_RTN_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (LLVMAnd (GetVal $old),
                                  (LLVMXor (GetNamedOperand $data0), (ConstantInt llvm_i64_ty, -1))),
                        (GetNamedOperand $data1)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MSKOR_RTN_B32: D = MEM[addr]; MEM[addr] = (D & ~data0) | data1 (32-bit, with return).
def DS_MSKOR_RTN_B32_sem : InstSISemantic<DS_MSKOR_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMOr (LLVMAnd (GetVal $old),
                                  (LLVMXor (GetNamedOperand $data0), (ConstantInt llvm_i32_ty, -1))),
                        (GetNamedOperand $data1)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS 64-bit Compare-and-Store Instructions
//------------------------------------------------------------------------------

// DS_CMPST_B64: if (MEM[addr] == data0) MEM[addr] = data1 (64-bit).
def DS_CMPST_B64_sem : InstSISemantic<DS_CMPST_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPST_RTN_B64: D = MEM[addr]; if (D == data0) MEM[addr] = data1 (64-bit).
def DS_CMPST_RTN_B64_sem : InstSISemantic<DS_CMPST_RTN_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_CMPST_F64: if (as_double(MEM[addr]) == as_double(data0)) MEM[addr] = data1 (64-bit float compare).
def DS_CMPST_F64_sem : InstSISemantic<DS_CMPST_F64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
                  (LLVMBitCast (GetVal $old), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_double_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPST_RTN_F64: D = MEM[addr]; if (as_double(D) == as_double(data0)) MEM[addr] = data1 (64-bit float).
def DS_CMPST_RTN_F64_sem : InstSISemantic<DS_CMPST_RTN_F64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
                  (LLVMBitCast (GetVal $old), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_double_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS CMPSTORE variants (GFX12+ renamed CMPST)
// Identical semantics to their CMPST counterparts.
//------------------------------------------------------------------------------

// DS_CMPSTORE_B32: same as DS_CMPST_B32.
def DS_CMPSTORE_B32_sem : InstSISemantic<DS_CMPSTORE_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPSTORE_B64: same as DS_CMPST_B64.
def DS_CMPSTORE_B64_sem : InstSISemantic<DS_CMPSTORE_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPSTORE_F32: same as DS_CMPST_F32.
def DS_CMPSTORE_F32_sem : InstSISemantic<DS_CMPSTORE_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
                  (LLVMBitCast (GetVal $old), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_float_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPSTORE_F64: same as DS_CMPST_F64.
def DS_CMPSTORE_F64_sem : InstSISemantic<DS_CMPSTORE_F64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
                  (LLVMBitCast (GetVal $old), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_double_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_CMPSTORE_RTN_B32: same as DS_CMPST_RTN_B32.
def DS_CMPSTORE_RTN_B32_sem : InstSISemantic<DS_CMPSTORE_RTN_B32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_CMPSTORE_RTN_B64: same as DS_CMPST_RTN_B64.
def DS_CMPSTORE_RTN_B64_sem : InstSISemantic<DS_CMPSTORE_RTN_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $old), (GetNamedOperand $data0)),
                            (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_CMPSTORE_RTN_F32: same as DS_CMPST_RTN_F32.
def DS_CMPSTORE_RTN_F32_sem : InstSISemantic<DS_CMPSTORE_RTN_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
                  (LLVMBitCast (GetVal $old), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_float_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_CMPSTORE_RTN_F64: same as DS_CMPST_RTN_F64.
def DS_CMPSTORE_RTN_F64_sem : InstSISemantic<DS_CMPSTORE_RTN_F64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMFCmp LLVMFCmpOEQ,
                  (LLVMBitCast (GetVal $old), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data0), llvm_double_ty)),
                (GetNamedOperand $data1), (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS Float Atomics (64-bit and return variants)
//------------------------------------------------------------------------------

// DS_ADD_F64: tmp = MEM[addr]; MEM[addr] = as_double(tmp) + as_double(data) (64-bit float add).
def DS_ADD_F64_sem : InstSISemantic<DS_ADD_F64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMFAdd (LLVMBitCast (GetVal $tmp), llvm_double_ty),
                                       (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)),
                             llvm_i64_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_ADD_RTN_F32: D = MEM[addr]; MEM[addr] = as_float(D) + as_float(data) (32-bit float add with return).
def DS_ADD_RTN_F32_sem : InstSISemantic<DS_ADD_RTN_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMFAdd (LLVMBitCast (GetVal $old), llvm_float_ty),
                                       (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)),
                             llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_ADD_RTN_F64: D = MEM[addr]; MEM[addr] = as_double(D) + as_double(data) (64-bit float add with return).
def DS_ADD_RTN_F64_sem : InstSISemantic<DS_ADD_RTN_F64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMFAdd (LLVMBitCast (GetVal $old), llvm_double_ty),
                                       (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)),
                             llvm_i64_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MIN_F64: tmp = MEM[addr]; MEM[addr] = min(as_double(tmp), as_double(data)) (64-bit float min).
def DS_MIN_F64_sem : InstSISemantic<DS_MIN_F64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOLT,
                  (LLVMBitCast (GetVal $tmp), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)),
                (LLVMBitCast (GetVal $tmp), llvm_double_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)), llvm_i64_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MAX_F64: tmp = MEM[addr]; MEM[addr] = max(as_double(tmp), as_double(data)) (64-bit float max).
def DS_MAX_F64_sem : InstSISemantic<DS_MAX_F64> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOGT,
                  (LLVMBitCast (GetVal $tmp), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)),
                (LLVMBitCast (GetVal $tmp), llvm_double_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)), llvm_i64_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_MIN_RTN_F32: D = MEM[addr]; MEM[addr] = min(as_float(D), as_float(data)) (32-bit float min with return).
def DS_MIN_RTN_F32_sem : InstSISemantic<DS_MIN_RTN_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOLT,
                  (LLVMBitCast (GetVal $old), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)),
                (LLVMBitCast (GetVal $old), llvm_float_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MAX_RTN_F32: D = MEM[addr]; MEM[addr] = max(as_float(D), as_float(data)) (32-bit float max with return).
def DS_MAX_RTN_F32_sem : InstSISemantic<DS_MAX_RTN_F32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOGT,
                  (LLVMBitCast (GetVal $old), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)),
                (LLVMBitCast (GetVal $old), llvm_float_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_float_ty)), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MIN_RTN_F64: D = MEM[addr]; MEM[addr] = min(as_double(D), as_double(data)) (64-bit float min with return).
def DS_MIN_RTN_F64_sem : InstSISemantic<DS_MIN_RTN_F64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOLT,
                  (LLVMBitCast (GetVal $old), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)),
                (LLVMBitCast (GetVal $old), llvm_double_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)), llvm_i64_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_MAX_RTN_F64: D = MEM[addr]; MEM[addr] = max(as_double(D), as_double(data)) (64-bit float max with return).
def DS_MAX_RTN_F64_sem : InstSISemantic<DS_MAX_RTN_F64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMSelect (LLVMFCmp LLVMFCmpOGT,
                  (LLVMBitCast (GetVal $old), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)),
                (LLVMBitCast (GetVal $old), llvm_double_ty),
                (LLVMBitCast (GetNamedOperand $data), llvm_double_ty)), llvm_i64_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS Exchange Instructions (64-bit and WRXCHG2)
//------------------------------------------------------------------------------

// DS_WRXCHG_RTN_B64: D = MEM[addr]; MEM[addr] = data (64-bit exchange).
def DS_WRXCHG_RTN_B64_sem : InstSISemantic<DS_WRXCHG_RTN_B64> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i64_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (GetNamedOperand $data),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_WRXCHG2_RTN_B32: Dual-offset 32-bit exchange.
// addr0 = addr + offset0 * 4; addr1 = addr + offset1 * 4
// old0 = MEM[addr0]; MEM[addr0] = data0; old1 = MEM[addr1]; MEM[addr1] = data1
// RETURN_DATA = {old1, old0}
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 231.
//   sea-islands-instruction-set-architecture_0.pdf, page 241.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-shader-instruction-set-architecture.pdf, page 187.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 189.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 179.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 177.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 429.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 476.
//   rdna-shader-instruction-set-architecture.pdf, page 208.
//   rdna2-shader-instruction-set-architecture.pdf, page 203.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 520.
//   rdna4-instruction-set-architecture.pdf, page 595 (as DS_STOREXCHG_2ADDR_RTN_B32).
def DS_WRXCHG2_RTN_B32_sem : InstSISemantic<DS_WRXCHG2_RTN_B32> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 4)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 4)))),
    (DefVal $old0, (LLVMLoad llvm_i32_ty, $addr0)),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (DefVal $old1, (LLVMLoad llvm_i32_ty, $addr1)),
    (LLVMStore (GetNamedOperand $data1), $addr1),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i32_ty), $old0, (ConstantInt llvm_i32_ty, 0)),
      $old1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

// DS_WRXCHG2_RTN_B64: Dual-offset 64-bit exchange.
// addr0 = addr + offset0 * 8; addr1 = addr + offset1 * 8
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 235.
//   sea-islands-instruction-set-architecture_0.pdf, page 244.
//   gcn3-instruction-set-architecture.pdf, page 270.
//   vega-shader-instruction-set-architecture.pdf, page 195.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 197.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 186.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 184.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 449.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 495.
//   rdna-shader-instruction-set-architecture.pdf, page 216.
//   rdna2-shader-instruction-set-architecture.pdf, page 210.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 535.
//   rdna4-instruction-set-architecture.pdf, page 610 (as DS_STOREXCHG_2ADDR_RTN_B64).
def DS_WRXCHG2_RTN_B64_sem : InstSISemantic<DS_WRXCHG2_RTN_B64> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 8)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 8)))),
    (DefVal $old0, (LLVMLoad llvm_i64_ty, $addr0)),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (DefVal $old1, (LLVMLoad llvm_i64_ty, $addr1)),
    (LLVMStore (GetNamedOperand $data1), $addr1),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i64_ty), $old0, (ConstantInt llvm_i32_ty, 0)),
      $old1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

// DS_WRXCHG2ST64_RTN_B32: Dual-offset stride-64 32-bit exchange.
// addr0 = addr + offset0 * 256; addr1 = addr + offset1 * 256
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 231.
//   sea-islands-instruction-set-architecture_0.pdf, page 241.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-shader-instruction-set-architecture.pdf, page 187.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 189.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 179.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 177.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 429.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 476.
//   rdna-shader-instruction-set-architecture.pdf, page 208.
//   rdna2-shader-instruction-set-architecture.pdf, page 203.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 520.
//   rdna4-instruction-set-architecture.pdf, page 595 (as DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32).
def DS_WRXCHG2ST64_RTN_B32_sem : InstSISemantic<DS_WRXCHG2ST64_RTN_B32> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 256)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 256)))),
    (DefVal $old0, (LLVMLoad llvm_i32_ty, $addr0)),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (DefVal $old1, (LLVMLoad llvm_i32_ty, $addr1)),
    (LLVMStore (GetNamedOperand $data1), $addr1),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i32_ty), $old0, (ConstantInt llvm_i32_ty, 0)),
      $old1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

// DS_WRXCHG2ST64_RTN_B64: Dual-offset stride-64 64-bit exchange.
// addr0 = addr + offset0 * 512; addr1 = addr + offset1 * 512
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 235.
//   sea-islands-instruction-set-architecture_0.pdf, page 244.
//   gcn3-instruction-set-architecture.pdf, page 270.
//   vega-shader-instruction-set-architecture.pdf, page 195.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 197.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 186.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 184.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 449.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 496.
//   rdna-shader-instruction-set-architecture.pdf, page 217.
//   rdna2-shader-instruction-set-architecture.pdf, page 210.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 535.
//   rdna4-instruction-set-architecture.pdf, page 611 (as DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64).
def DS_WRXCHG2ST64_RTN_B64_sem : InstSISemantic<DS_WRXCHG2ST64_RTN_B64> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 512)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 512)))),
    (DefVal $old0, (LLVMLoad llvm_i64_ty, $addr0)),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (DefVal $old1, (LLVMLoad llvm_i64_ty, $addr1)),
    (LLVMStore (GetNamedOperand $data1), $addr1),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i64_ty), $old0, (ConstantInt llvm_i32_ty, 0)),
      $old1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

//------------------------------------------------------------------------------
// DS Conditional Subtract and Clamp Instructions
//------------------------------------------------------------------------------

// DS_COND_SUB_U32: tmp = MEM[addr]; MEM[addr] = (tmp >= data) ? tmp - data : tmp.
def DS_COND_SUB_U32_sem : InstSISemantic<DS_COND_SUB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $data)),
                            (LLVMSub (GetVal $tmp), (GetNamedOperand $data)),
                            (GetVal $tmp)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_COND_SUB_RTN_U32: D = MEM[addr]; MEM[addr] = (D >= data) ? D - data : D; return old.
def DS_COND_SUB_RTN_U32_sem : InstSISemantic<DS_COND_SUB_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $old), (GetNamedOperand $data)),
                            (LLVMSub (GetVal $old), (GetNamedOperand $data)),
                            (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_SUB_CLAMP_U32: tmp = MEM[addr]; MEM[addr] = (tmp < data) ? 0 : tmp - data.
def DS_SUB_CLAMP_U32_sem : InstSISemantic<DS_SUB_CLAMP_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $tmp), (GetNamedOperand $data)),
                            (ConstantInt llvm_i32_ty, 0),
                            (LLVMSub (GetVal $tmp), (GetNamedOperand $data))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_SUB_CLAMP_RTN_U32: D = MEM[addr]; MEM[addr] = (D < data) ? 0 : D - data; return old.
def DS_SUB_CLAMP_RTN_U32_sem : InstSISemantic<DS_SUB_CLAMP_RTN_U32> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $old), (GetNamedOperand $data)),
                            (ConstantInt llvm_i32_ty, 0),
                            (LLVMSub (GetVal $old), (GetNamedOperand $data))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS D16 Read/Write Variants
//------------------------------------------------------------------------------

// DS_READ_U8_D16: Load u8, zero-extend, pack into low 16 bits of vdst, preserve high 16.
def DS_READ_U8_D16_sem : InstSISemantic<DS_READ_U8_D16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $lo, (LLVMZExt (GetVal $tmp), llvm_i32_ty)),
    (DefVal $hi, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_U8_D16_HI: Load u8, zero-extend, pack into high 16 bits of vdst, preserve low 16.
def DS_READ_U8_D16_HI_sem : InstSISemantic<DS_READ_U8_D16_HI> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $hi, (LLVMShl (LLVMZExt (GetVal $tmp), llvm_i32_ty), (ConstantInt llvm_i32_ty, 16))),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_I8_D16: Load i8, sign-extend to i16, pack into low 16 bits of vdst, preserve high 16.
def DS_READ_I8_D16_sem : InstSISemantic<DS_READ_I8_D16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $sext16, (LLVMSExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $lo, (LLVMZExt (GetVal $sext16), llvm_i32_ty)),
    (DefVal $hi, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_I8_D16_HI: Load i8, sign-extend to i16, pack into high 16 bits of vdst, preserve low 16.
def DS_READ_I8_D16_HI_sem : InstSISemantic<DS_READ_I8_D16_HI> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $sext16, (LLVMSExt (GetVal $tmp), llvm_i16_ty)),
    (DefVal $hi, (LLVMShl (LLVMZExt (GetVal $sext16), llvm_i32_ty), (ConstantInt llvm_i32_ty, 16))),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_U16_D16: Load u16, pack into low 16 bits of vdst, preserve high 16.
def DS_READ_U16_D16_sem : InstSISemantic<DS_READ_U16_D16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $lo, (LLVMZExt (GetVal $tmp), llvm_i32_ty)),
    (DefVal $hi, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_U16_D16_HI: Load u16, pack into high 16 bits of vdst, preserve low 16.
def DS_READ_U16_D16_HI_sem : InstSISemantic<DS_READ_U16_D16_HI> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i16_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $hi, (LLVMShl (LLVMZExt (GetVal $tmp), llvm_i32_ty), (ConstantInt llvm_i32_ty, 16))),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_WRITE_B8_D16_HI: Store bits [23:16] of data as byte.
def DS_WRITE_B8_D16_HI_sem : InstSISemantic<DS_WRITE_B8_D16_HI> {
  let Semantic = [
    (LLVMStore (LLVMTrunc (LLVMLShr (GetNamedOperand $data0), (ConstantInt llvm_i32_ty, 16)), llvm_i8_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_WRITE_B16_D16_HI: Store bits [31:16] of data as halfword.
def DS_WRITE_B16_D16_HI_sem : InstSISemantic<DS_WRITE_B16_D16_HI> {
  let Semantic = [
    (LLVMStore (LLVMTrunc (LLVMLShr (GetNamedOperand $data0), (ConstantInt llvm_i32_ty, 16)), llvm_i16_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

//------------------------------------------------------------------------------
// DS Packed Float Add (PK_ADD)
//------------------------------------------------------------------------------

// DS_PK_ADD_F16: Packed 2xf16 add (no return).
def DS_PK_ADD_F16_sem : InstSISemantic<DS_PK_ADD_F16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $mem_v2f16, (LLVMBitCast (GetVal $tmp), llvm_v2f16_ty)),
    (DefVal $data_v2f16, (LLVMBitCast (GetNamedOperand $data), llvm_v2f16_ty)),
    (DefVal $result, (LLVMFAdd (GetVal $mem_v2f16), (GetVal $data_v2f16))),
    (LLVMStore (LLVMBitCast (GetVal $result), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_PK_ADD_RTN_F16: Packed 2xf16 add with return.
def DS_PK_ADD_RTN_F16_sem : InstSISemantic<DS_PK_ADD_RTN_F16> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $mem_v2f16, (LLVMBitCast (GetVal $old), llvm_v2f16_ty)),
    (DefVal $data_v2f16, (LLVMBitCast (GetNamedOperand $data), llvm_v2f16_ty)),
    (DefVal $result, (LLVMFAdd (GetVal $mem_v2f16), (GetVal $data_v2f16))),
    (LLVMStore (LLVMBitCast (GetVal $result), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

// DS_PK_ADD_BF16: Packed 2xbf16 add (no return).
def DS_PK_ADD_BF16_sem : InstSISemantic<DS_PK_ADD_BF16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $mem_v2bf16, (LLVMBitCast (GetVal $tmp), llvm_v2bf16_ty)),
    (DefVal $data_v2bf16, (LLVMBitCast (GetNamedOperand $data), llvm_v2bf16_ty)),
    (DefVal $result, (LLVMFAdd (GetVal $mem_v2bf16), (GetVal $data_v2bf16))),
    (LLVMStore (LLVMBitCast (GetVal $result), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_PK_ADD_RTN_BF16: Packed 2xbf16 add with return.
def DS_PK_ADD_RTN_BF16_sem : InstSISemantic<DS_PK_ADD_RTN_BF16> {
  let Semantic = [
    (DefVal $old, (LLVMLoad llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $mem_v2bf16, (LLVMBitCast (GetVal $old), llvm_v2bf16_ty)),
    (DefVal $data_v2bf16, (LLVMBitCast (GetNamedOperand $data), llvm_v2bf16_ty)),
    (DefVal $result, (LLVMFAdd (GetVal $mem_v2bf16), (GetVal $data_v2bf16))),
    (LLVMStore (LLVMBitCast (GetVal $result), llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ];
}

//------------------------------------------------------------------------------
// DS READ2/WRITE2 Instructions (inline asm — dual offset)
//------------------------------------------------------------------------------

// DS_READ2_B32: Dual-offset 32-bit read.
// RETURN_DATA[31:0] = MEM[addr + offset0 * 4]; RETURN_DATA[63:32] = MEM[addr + offset1 * 4]
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 234.
//   sea-islands-instruction-set-architecture_0.pdf, page 243.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-shader-instruction-set-architecture.pdf, page 188.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 190.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 178.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 432.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 479.
//   rdna-shader-instruction-set-architecture.pdf, page 209.
//   rdna2-shader-instruction-set-architecture.pdf, page 204.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 524.
//   rdna4-instruction-set-architecture.pdf, page 599 (as DS_LOAD_2ADDR_B32).
def DS_READ2_B32_sem : InstSISemantic<DS_READ2_B32> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 4)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 4)))),
    (DefVal $val0, (LLVMLoad llvm_i32_ty, $addr0)),
    (DefVal $val1, (LLVMLoad llvm_i32_ty, $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i32_ty), $val0, (ConstantInt llvm_i32_ty, 0)),
      $val1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

// DS_READ2_B64: Dual-offset 64-bit read.
// RETURN_DATA[63:0] = MEM[addr + offset0 * 8]; RETURN_DATA[127:64] = MEM[addr + offset1 * 8]
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 235.
//   sea-islands-instruction-set-architecture_0.pdf, page 244.
//   gcn3-instruction-set-architecture.pdf, page 270.
//   vega-shader-instruction-set-architecture.pdf, page 196.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 198.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 187.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 185.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 451.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 498.
//   rdna-shader-instruction-set-architecture.pdf, page 217.
//   rdna2-shader-instruction-set-architecture.pdf, page 211.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 537.
//   rdna4-instruction-set-architecture.pdf, page 612 (as DS_LOAD_2ADDR_B64).
def DS_READ2_B64_sem : InstSISemantic<DS_READ2_B64> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 8)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 8)))),
    (DefVal $val0, (LLVMLoad llvm_i64_ty, $addr0)),
    (DefVal $val1, (LLVMLoad llvm_i64_ty, $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i64_ty), $val0, (ConstantInt llvm_i32_ty, 0)),
      $val1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

// DS_READ2ST64_B32: Dual-offset stride-64 32-bit read.
// RETURN_DATA[31:0] = MEM[addr + offset0 * 256]; RETURN_DATA[63:32] = MEM[addr + offset1 * 256]
// Sources (present in all architectures, same pages as DS_READ2_B32):
//   southern-islands-instruction-set-architecture.pdf, page 234.
//   sea-islands-instruction-set-architecture_0.pdf, page 243.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-shader-instruction-set-architecture.pdf, page 188.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 190.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 178.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 432.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 479.
//   rdna-shader-instruction-set-architecture.pdf, page 209.
//   rdna2-shader-instruction-set-architecture.pdf, page 204.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 524.
//   rdna4-instruction-set-architecture.pdf, page 599 (as DS_LOAD_2ADDR_STRIDE64_B32).
def DS_READ2ST64_B32_sem : InstSISemantic<DS_READ2ST64_B32> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 256)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 256)))),
    (DefVal $val0, (LLVMLoad llvm_i32_ty, $addr0)),
    (DefVal $val1, (LLVMLoad llvm_i32_ty, $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i32_ty), $val0, (ConstantInt llvm_i32_ty, 0)),
      $val1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

// DS_READ2ST64_B64: Dual-offset stride-64 64-bit read.
// RETURN_DATA[63:0] = MEM[addr + offset0 * 512]; RETURN_DATA[127:64] = MEM[addr + offset1 * 512]
// Sources (present in all architectures, same pages as DS_READ2_B64):
//   southern-islands-instruction-set-architecture.pdf, page 235.
//   sea-islands-instruction-set-architecture_0.pdf, page 244.
//   gcn3-instruction-set-architecture.pdf, page 270.
//   vega-shader-instruction-set-architecture.pdf, page 196.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 198.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 187.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 185.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 451.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 498.
//   rdna-shader-instruction-set-architecture.pdf, page 217.
//   rdna2-shader-instruction-set-architecture.pdf, page 211.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 537.
//   rdna4-instruction-set-architecture.pdf, page 613 (as DS_LOAD_2ADDR_STRIDE64_B64).
def DS_READ2ST64_B64_sem : InstSISemantic<DS_READ2ST64_B64> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 512)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 512)))),
    (DefVal $val0, (LLVMLoad llvm_i64_ty, $addr0)),
    (DefVal $val1, (LLVMLoad llvm_i64_ty, $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal llvm_v2i64_ty), $val0, (ConstantInt llvm_i32_ty, 0)),
      $val1, (ConstantInt llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, $result)
  ];
}

// DS_WRITE2_B32: Dual-offset 32-bit write.
// MEM[addr + offset0 * 4] = data0; MEM[addr + offset1 * 4] = data1
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 230.
//   sea-islands-instruction-set-architecture_0.pdf, page 240.
//   gcn3-instruction-set-architecture.pdf, page 267.
//   vega-shader-instruction-set-architecture.pdf, page 184.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 176.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 175.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 422.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 468.
//   rdna-shader-instruction-set-architecture.pdf, page 203.
//   rdna2-shader-instruction-set-architecture.pdf, page 197.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 514.
//   rdna4-instruction-set-architecture.pdf, page 588 (as DS_STORE_2ADDR_B32).
def DS_WRITE2_B32_sem : InstSISemantic<DS_WRITE2_B32> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 4)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 4)))),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (LLVMStore (GetNamedOperand $data1), $addr1)
  ];
}

// DS_WRITE2_B64: Dual-offset 64-bit write.
// MEM[addr + offset0 * 8] = data0; MEM[addr + offset1 * 8] = data1
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 235.
//   sea-islands-instruction-set-architecture_0.pdf, page 244.
//   gcn3-instruction-set-architecture.pdf, page 269.
//   vega-shader-instruction-set-architecture.pdf, page 192.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 194.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 184.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 182.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 441.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 488.
//   rdna-shader-instruction-set-architecture.pdf, page 214.
//   rdna2-shader-instruction-set-architecture.pdf, page 208.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 530.
//   rdna4-instruction-set-architecture.pdf, page 605 (as DS_STORE_2ADDR_B64).
def DS_WRITE2_B64_sem : InstSISemantic<DS_WRITE2_B64> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 8)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 8)))),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (LLVMStore (GetNamedOperand $data1), $addr1)
  ];
}

// DS_WRITE2ST64_B32: Dual-offset stride-64 32-bit write.
// MEM[addr + offset0 * 256] = data0; MEM[addr + offset1 * 256] = data1
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 231.
//   sea-islands-instruction-set-architecture_0.pdf, page 241.
//   gcn3-instruction-set-architecture.pdf, page 268.
//   vega-shader-instruction-set-architecture.pdf, page 184.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 186.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 176.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 175.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 422.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 469.
//   rdna-shader-instruction-set-architecture.pdf, page 203.
//   rdna2-shader-instruction-set-architecture.pdf, page 197.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 514.
//   rdna4-instruction-set-architecture.pdf, page 589 (as DS_STORE_2ADDR_STRIDE64_B32).
def DS_WRITE2ST64_B32_sem : InstSISemantic<DS_WRITE2ST64_B32> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 256)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 256)))),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (LLVMStore (GetNamedOperand $data1), $addr1)
  ];
}

// DS_WRITE2ST64_B64: Dual-offset stride-64 64-bit write.
// MEM[addr + offset0 * 512] = data0; MEM[addr + offset1 * 512] = data1
// Sources (present in all architectures):
//   southern-islands-instruction-set-architecture.pdf, page 235.
//   sea-islands-instruction-set-architecture_0.pdf, page 244.
//   gcn3-instruction-set-architecture.pdf, page 269.
//   vega-shader-instruction-set-architecture.pdf, page 192.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 194.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 184.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 182.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 441.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 488.
//   rdna-shader-instruction-set-architecture.pdf, page 214.
//   rdna2-shader-instruction-set-architecture.pdf, page 208.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 530.
//   rdna4-instruction-set-architecture.pdf, page 605 (as DS_STORE_2ADDR_STRIDE64_B64).
def DS_WRITE2ST64_B64_sem : InstSISemantic<DS_WRITE2ST64_B64> {
  let Semantic = [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset0), (ConstantInt llvm_i32_ty, 512)))),
    (DefVal $addr1, (LLVMPtrAdd $base, (LLVMMul (GetNamedOperand $offset1), (ConstantInt llvm_i32_ty, 512)))),
    (LLVMStore (GetNamedOperand $data0), $addr0),
    (LLVMStore (GetNamedOperand $data1), $addr1)
  ];
}

//------------------------------------------------------------------------------
// DS SRC2 Instructions (inline asm — implicit second source from LDS)
//------------------------------------------------------------------------------

// DS_ADD_SRC2_U32
def DS_ADD_SRC2_U32_sem : InstSISemantic<DS_ADD_SRC2_U32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_add_src2_u32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_ADD_SRC2_U64
def DS_ADD_SRC2_U64_sem : InstSISemantic<DS_ADD_SRC2_U64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_add_src2_u64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_ADD_SRC2_F32
def DS_ADD_SRC2_F32_sem : InstSISemantic<DS_ADD_SRC2_F32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_add_src2_f32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_SUB_SRC2_U32
def DS_SUB_SRC2_U32_sem : InstSISemantic<DS_SUB_SRC2_U32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_sub_src2_u32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_SUB_SRC2_U64
def DS_SUB_SRC2_U64_sem : InstSISemantic<DS_SUB_SRC2_U64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_sub_src2_u64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_RSUB_SRC2_U32
def DS_RSUB_SRC2_U32_sem : InstSISemantic<DS_RSUB_SRC2_U32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_rsub_src2_u32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_RSUB_SRC2_U64
def DS_RSUB_SRC2_U64_sem : InstSISemantic<DS_RSUB_SRC2_U64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_rsub_src2_u64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_INC_SRC2_U32
def DS_INC_SRC2_U32_sem : InstSISemantic<DS_INC_SRC2_U32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_inc_src2_u32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_INC_SRC2_U64
def DS_INC_SRC2_U64_sem : InstSISemantic<DS_INC_SRC2_U64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_inc_src2_u64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_DEC_SRC2_U32
def DS_DEC_SRC2_U32_sem : InstSISemantic<DS_DEC_SRC2_U32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_dec_src2_u32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_DEC_SRC2_U64
def DS_DEC_SRC2_U64_sem : InstSISemantic<DS_DEC_SRC2_U64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_dec_src2_u64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MIN_SRC2_I32
def DS_MIN_SRC2_I32_sem : InstSISemantic<DS_MIN_SRC2_I32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_min_src2_i32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MIN_SRC2_I64
def DS_MIN_SRC2_I64_sem : InstSISemantic<DS_MIN_SRC2_I64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_min_src2_i64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MIN_SRC2_U32
def DS_MIN_SRC2_U32_sem : InstSISemantic<DS_MIN_SRC2_U32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_min_src2_u32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MIN_SRC2_U64
def DS_MIN_SRC2_U64_sem : InstSISemantic<DS_MIN_SRC2_U64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_min_src2_u64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MIN_SRC2_F32
def DS_MIN_SRC2_F32_sem : InstSISemantic<DS_MIN_SRC2_F32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_min_src2_f32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MIN_SRC2_F64
def DS_MIN_SRC2_F64_sem : InstSISemantic<DS_MIN_SRC2_F64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_min_src2_f64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MAX_SRC2_I32
def DS_MAX_SRC2_I32_sem : InstSISemantic<DS_MAX_SRC2_I32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_max_src2_i32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MAX_SRC2_I64
def DS_MAX_SRC2_I64_sem : InstSISemantic<DS_MAX_SRC2_I64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_max_src2_i64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MAX_SRC2_U32
def DS_MAX_SRC2_U32_sem : InstSISemantic<DS_MAX_SRC2_U32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_max_src2_u32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MAX_SRC2_U64
def DS_MAX_SRC2_U64_sem : InstSISemantic<DS_MAX_SRC2_U64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_max_src2_u64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MAX_SRC2_F32
def DS_MAX_SRC2_F32_sem : InstSISemantic<DS_MAX_SRC2_F32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_max_src2_f32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_MAX_SRC2_F64
def DS_MAX_SRC2_F64_sem : InstSISemantic<DS_MAX_SRC2_F64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_max_src2_f64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_AND_SRC2_B32
def DS_AND_SRC2_B32_sem : InstSISemantic<DS_AND_SRC2_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_and_src2_b32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_AND_SRC2_B64
def DS_AND_SRC2_B64_sem : InstSISemantic<DS_AND_SRC2_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_and_src2_b64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_OR_SRC2_B32
def DS_OR_SRC2_B32_sem : InstSISemantic<DS_OR_SRC2_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_or_src2_b32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_OR_SRC2_B64
def DS_OR_SRC2_B64_sem : InstSISemantic<DS_OR_SRC2_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_or_src2_b64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_XOR_SRC2_B32
def DS_XOR_SRC2_B32_sem : InstSISemantic<DS_XOR_SRC2_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_xor_src2_b32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_XOR_SRC2_B64
def DS_XOR_SRC2_B64_sem : InstSISemantic<DS_XOR_SRC2_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_xor_src2_b64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_WRITE_SRC2_B32
def DS_WRITE_SRC2_B32_sem : InstSISemantic<DS_WRITE_SRC2_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_write_src2_b32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_WRITE_SRC2_B64
def DS_WRITE_SRC2_B64_sem : InstSISemantic<DS_WRITE_SRC2_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_write_src2_b64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

//------------------------------------------------------------------------------
// DS GWS Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_GWS_INIT: Initialize GWS resource.
def DS_GWS_INIT_sem : InstSISemantic<DS_GWS_INIT> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_gws_init $0 offset:$1 gds",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}

// DS_GWS_BARRIER: GWS barrier.
def DS_GWS_BARRIER_sem : InstSISemantic<DS_GWS_BARRIER> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_gws_barrier $0 offset:$1 gds",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}

// DS_GWS_SEMA_BR: GWS semaphore broadcast release.
def DS_GWS_SEMA_BR_sem : InstSISemantic<DS_GWS_SEMA_BR> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_gws_sema_br $0 offset:$1 gds",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}

// DS_GWS_SEMA_P: GWS semaphore P (wait).
def DS_GWS_SEMA_P_sem : InstSISemantic<DS_GWS_SEMA_P> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_gws_sema_p offset:$0 gds",
      "i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $offset))
  ];
}

// DS_GWS_SEMA_V: GWS semaphore V (signal).
def DS_GWS_SEMA_V_sem : InstSISemantic<DS_GWS_SEMA_V> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_gws_sema_v offset:$0 gds",
      "i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $offset))
  ];
}

//------------------------------------------------------------------------------
// DS Ordered Count (inline asm)
//------------------------------------------------------------------------------

// DS_ORDERED_COUNT: Ordered count operation (GDS).
def DS_ORDERED_COUNT_sem : InstSISemantic<DS_ORDERED_COUNT> {
  let Semantic = [
    (LLVMInlineAsm llvm_i32_ty, "ds_ordered_count $0, $1 offset:$2 gds",
      "=v,v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}

//------------------------------------------------------------------------------
// DS BVH Stack Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_BVH_STACK_RTN_B32: BVH stack pop/push.
def DS_BVH_STACK_RTN_B32_sem : InstSISemantic<DS_BVH_STACK_RTN_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_i32_ty, "ds_bvh_stack_rtn_b32 $0, $1, $2, $3",
      "=v,v,v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $data0), (GetNamedOperand $data1))
  ];
}

// DS_BVH_STACK_PUSH8_POP1_RTN_B32: BVH stack push 8 pop 1.
def DS_BVH_STACK_PUSH8_POP1_RTN_B32_sem : InstSISemantic<DS_BVH_STACK_PUSH8_POP1_RTN_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_i32_ty, "ds_bvh_stack_push8_pop1_rtn_b32 $0, $1, $2, $3",
      "=v,v,v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $data0), (GetNamedOperand $data1))
  ];
}

// DS_BVH_STACK_PUSH8_POP2_RTN_B64: BVH stack push 8 pop 2.
def DS_BVH_STACK_PUSH8_POP2_RTN_B64_sem : InstSISemantic<DS_BVH_STACK_PUSH8_POP2_RTN_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_bvh_stack_push8_pop2_rtn_b64 $0, $1, $2, $3",
      "=v,v,v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $data0), (GetNamedOperand $data1))
  ];
}

//------------------------------------------------------------------------------
// DS LOAD_TR / READ_TR Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_LOAD_TR4_B64: Load transpose 4 bytes to 64 bits.
def DS_LOAD_TR4_B64_sem : InstSISemantic<DS_LOAD_TR4_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_load_tr4_b64 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

// DS_LOAD_TR6_B96: Load transpose 6 bytes to 96 bits.
def DS_LOAD_TR6_B96_sem : InstSISemantic<DS_LOAD_TR6_B96> {
  let Semantic = [
    (LLVMInlineAsm llvm_v3i32_ty, "ds_load_tr6_b96 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

// DS_LOAD_TR8_B64: Load transpose 8 bytes to 64 bits.
def DS_LOAD_TR8_B64_sem : InstSISemantic<DS_LOAD_TR8_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_load_tr8_b64 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

// DS_LOAD_TR16_B128: Load transpose 16 bytes to 128 bits.
def DS_LOAD_TR16_B128_sem : InstSISemantic<DS_LOAD_TR16_B128> {
  let Semantic = [
    (LLVMInlineAsm llvm_v4i32_ty, "ds_load_tr16_b128 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

// DS_READ_B64_TR_B4: Read 64-bit transpose from 4-byte element.
def DS_READ_B64_TR_B4_sem : InstSISemantic<DS_READ_B64_TR_B4> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_read_b64_tr_b4 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

// DS_READ_B64_TR_B8: Read 64-bit transpose from 8-byte element.
def DS_READ_B64_TR_B8_sem : InstSISemantic<DS_READ_B64_TR_B8> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_read_b64_tr_b8 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

// DS_READ_B64_TR_B16: Read 64-bit transpose from 16-byte element.
def DS_READ_B64_TR_B16_sem : InstSISemantic<DS_READ_B64_TR_B16> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_read_b64_tr_b16 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

// DS_READ_B96_TR_B6: Read 96-bit transpose from 6-byte element.
def DS_READ_B96_TR_B6_sem : InstSISemantic<DS_READ_B96_TR_B6> {
  let Semantic = [
    (LLVMInlineAsm llvm_v3i32_ty, "ds_read_b96_tr_b6 $0, $1",
      "=v,v", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr))
  ];
}

//------------------------------------------------------------------------------
// DS ADDTID Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_READ_ADDTID_B32: Read with thread-ID-based address offset.
def DS_READ_ADDTID_B32_sem : InstSISemantic<DS_READ_ADDTID_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_i32_ty, "ds_read_addtid_b32 $0 offset:$1",
      "=v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $offset))
  ];
}

// DS_WRITE_ADDTID_B32: Write with thread-ID-based address offset.
def DS_WRITE_ADDTID_B32_sem : InstSISemantic<DS_WRITE_ADDTID_B32> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_write_addtid_b32 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}

//------------------------------------------------------------------------------
// DS GS_REG Instructions (inline asm, RDNA3+)
//------------------------------------------------------------------------------

// DS_ADD_GS_REG_RTN: Add to GS register with return.
def DS_ADD_GS_REG_RTN_sem : InstSISemantic<DS_ADD_GS_REG_RTN> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_add_gs_reg_rtn $0, $1 offset:$2 gds",
      "=v,v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}

// DS_SUB_GS_REG_RTN: Subtract from GS register with return.
def DS_SUB_GS_REG_RTN_sem : InstSISemantic<DS_SUB_GS_REG_RTN> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_sub_gs_reg_rtn $0, $1 offset:$2 gds",
      "=v,v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}

//------------------------------------------------------------------------------
// DS Permute / Intrinsic Instructions
//------------------------------------------------------------------------------

// DS_BPERMUTE_FI_B32: Backward permute with fetch-inactive.
def DS_BPERMUTE_FI_B32_sem : InstSIIntrinsic<DS_BPERMUTE_FI_B32, int_amdgcn_ds_bpermute_fi_b32>;

//------------------------------------------------------------------------------
// DS Atomic Barrier Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64: Async barrier arrive (undocumented).
def DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64_sem : InstSISemantic<DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_void_ty, "ds_atomic_async_barrier_arrive_b64 $0 offset:$1",
      "v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

// DS_ATOMIC_BARRIER_ARRIVE_RTN_B64: Barrier arrive with return (undocumented).
def DS_ATOMIC_BARRIER_ARRIVE_RTN_B64_sem : InstSISemantic<DS_ATOMIC_BARRIER_ARRIVE_RTN_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_i64_ty, "ds_atomic_barrier_arrive_rtn_b64 $0, $1 offset:$2",
      "=v,v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $offset))
  ];
}

//------------------------------------------------------------------------------
// DS CONDXCHG32_RTN_B64 (inline asm — complex conditional dual-dword exchange)
//------------------------------------------------------------------------------

// DS_CONDXCHG32_RTN_B64: Conditional 32-bit exchange on 64-bit pair.
def DS_CONDXCHG32_RTN_B64_sem : InstSISemantic<DS_CONDXCHG32_RTN_B64> {
  let Semantic = [
    (LLVMInlineAsm llvm_v2i32_ty, "ds_condxchg32_rtn_b64 $0, $1, $2 offset:$3",
      "=v,v,v,i", /*hasSideEffects=*/1, /*isAlignStack=*/0,
      (GetNamedOperand $addr), (GetNamedOperand $data0), (GetNamedOperand $offset))
  ];
}
