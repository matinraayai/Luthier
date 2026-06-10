//===- DSSem.td ----------------------------------------------*- tblgen -*-===//
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
/// \p localOrRegionPtr is a dag selecting between local_ptr_ty (LDS, AS 3)
/// and region_ptr_ty (GDS, AS 2) — typically via SelectOperand on the gds
/// modifier.
class CalculateSingleOffsetDSAddr<dag addr, dag offset,
                                  dag localOrRegionPtr> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr addr, localOrRegionPtr), offset);
}

/// GFX6-GFX8 only: byte address is clamped against M0[16:0] (the LDS segment
/// size set by the kernel prolog) before pointer conversion. The kernel
/// prolog initializes M0 to at least the allocated LDS size, so in-bounds
/// accesses are unaffected; this expression models the hardware clamp for
/// accesses that may go out of bounds.
/// See AMDGPUUsage.rst LLVM doc, section "amdgpu-amdhsa-kernel-prolog-m0"
class CalculateClampedSingleOffsetDSAddr<dag addr, dag offset,
                                          dag localOrRegionPtr> {
  dag ret = (LLVMIntToPtr
              (LLVMBinaryIntrinsic int_umin,
                (LLVMAdd addr, offset),
                (LLVMAnd (ImplicitUse M0),
                          (ConstantInt si_llvm_i32_ty, 0x1FFFF))),
              localOrRegionPtr);
}

/// DS address calculation with both pointer-type and clamp-form selection:
///   - GDS vs LDS pointer type from the \c gds modifier;
///   - GFX9+ uses the unclamped form (HW does not use M0 for LDS clamping);
///     GFX6-GFX8 clamps the byte address against M0[16:0].
class DSAddr<dag addr, dag offset> {
  dag ret = (SelectOperand (hasFeature $GFX9Insts),
              CalculateSingleOffsetDSAddr<addr, offset,
                (SelectOperand (hasModifierSet $gds),
                                region_ptr_ty, local_ptr_ty)>.ret,
              CalculateClampedSingleOffsetDSAddr<addr, offset,
                (SelectOperand (hasModifierSet $gds),
                                region_ptr_ty, local_ptr_ty)>.ret);
}

//------------------------------------------------------------------------------
// DS 32-bit Atomic Operations (no return)
// tmp = MEM[addr]; MEM[addr] = OP(tmp, data)
//------------------------------------------------------------------------------














//------------------------------------------------------------------------------
// DS Float Atomic Operations (no return)
//------------------------------------------------------------------------------




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
    (LLVMStore (LLVMTrunc (GetNamedOperand $data0), si_llvm_i8_ty),
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
    (LLVMStore (LLVMTrunc (GetNamedOperand $data0), si_llvm_i16_ty),
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
    (SetNamedOperand $vdst, (LLVMLoad si_llvm_i32_ty,
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
    (SetNamedOperand $vdst, (LLVMLoad si_llvm_i64_ty,
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
    (SetNamedOperand $vdst, (LLVMZExt (LLVMLoad si_llvm_i8_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), si_llvm_i32_ty))
  ];
}

// DS_READ_I8: D = sext(MEM[addr][7:0]).
def DS_READ_I8_sem : InstSISemantic<DS_READ_I8> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMSExt (LLVMLoad si_llvm_i8_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), si_llvm_i32_ty))
  ];
}

// DS_READ_U16: D = zext(MEM[addr][15:0]).
def DS_READ_U16_sem : InstSISemantic<DS_READ_U16> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMZExt (LLVMLoad si_llvm_i16_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), si_llvm_i32_ty))
  ];
}

// DS_READ_I16: D = sext(MEM[addr][15:0]).
def DS_READ_I16_sem : InstSISemantic<DS_READ_I16> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMSExt (LLVMLoad si_llvm_i16_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret), si_llvm_i32_ty))
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














//------------------------------------------------------------------------------
// DS Compare-and-Store Instructions
//------------------------------------------------------------------------------





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
def DS_SWIZZLE_B32_sem : InstSISemantic<DS_SWIZZLE_B32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ds_swizzle, [(GetNamedOperand $addr), (GetNamedOperand $offset)]))
  ];
}

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
foreach inst = [DS_PERMUTE_B32, DS_PERMUTE_B32_agpr] in
  def : InstSISemantic<inst, [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ds_permute, [(GetNamedOperand $addr), (GetNamedOperand $data0)]))
  ]>;

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
foreach inst = [DS_BPERMUTE_B32, DS_BPERMUTE_B32_agpr] in
  def : InstSISemantic<inst, [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ds_bpermute, [(GetNamedOperand $addr), (GetNamedOperand $data0)]))
  ]>;

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
// DS_APPEND has no $addr operand (LLVM's DS_0A_RET sets has_addr = 0).
// The append-counter address is the instruction offset interpreted as a
// pointer into LDS (or GDS, when the gds modifier is set; the GDS form
// implicitly adds M0.base, which the intrinsic handles internally).
def DS_APPEND_sem : InstSISemantic<DS_APPEND> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ds_append,
        [(LLVMIntToPtr (GetNamedOperand $offset),
                       (SelectOperand (hasModifierSet $gds),
                                       region_ptr_ty, local_ptr_ty)),
         (ConstantInt si_llvm_i1_ty, 0)]))
  ];
}

// DS_CONSUME: Consume a dword from a buffer. GDS_BASE -= 4; D = MEM[GDS_BASE].
// Sources: same as DS_APPEND (same pages or adjacent).
def DS_CONSUME_sem : InstSISemantic<DS_CONSUME> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ds_consume,
        [(LLVMIntToPtr (GetNamedOperand $offset),
                       (SelectOperand (hasModifierSet $gds),
                                       region_ptr_ty, local_ptr_ty)),
         (ConstantInt si_llvm_i1_ty, 0)]))
  ];
}

//------------------------------------------------------------------------------
// CI+ DS Instructions
//------------------------------------------------------------------------------

// DS_GWS_SEMA_RELEASE_ALL: GWS semaphore release all (CI+).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
// DS_WRAP_RTN_B32: D = MEM[A]; MEM[A] = (D >= data0) ? D - data0 : D + data1.
// Designed for ring-buffer wraparound subtraction.
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
//   gcn3-instruction-set-architecture.pdf, page 267.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 430.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 468.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 528.
foreach inst = [DS_WRAP_RTN_B32, DS_WRAP_RTN_B32_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $old, (LLVMLoad si_llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $old), (GetNamedOperand $data0)),
                            (LLVMSub (GetVal $old), (GetNamedOperand $data0)),
                            (LLVMAdd (GetVal $old), (GetNamedOperand $data1))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ]>;

// DS_CONDXCHG32_RTN_B64: Conditional pair-dword exchange (CI+).
//   addr0 = (addr + offset) & ~7  (8-byte aligned)
//   addr1 = addr0 + 4
//   old0 = MEM[addr0]; if data[31]: MEM[addr0] = data[30:0]
//   old1 = MEM[addr1]; if data[63]: MEM[addr1] = data[62:32]
//   return {old1, old0}
// Note: address is constructed without DSAddr because the alignment mask must
// be applied to the integer byte address before pointer conversion. No M0
// clamping is modeled (gfx9+ assumption).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 490.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 542.
foreach inst = [DS_CONDXCHG32_RTN_B64, DS_CONDXCHG32_RTN_B64_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $addr_full, (LLVMAdd (GetNamedOperand $addr), (GetNamedOperand $offset))),
    (DefVal $addr0_int, (LLVMAnd (GetVal $addr_full), (ConstantInt si_llvm_i32_ty, 0xFFFFFFF8))),
    (DefVal $addr0, (LLVMIntToPtr (GetVal $addr0_int),
                     (SelectOperand (hasModifierSet $gds),
                                     region_ptr_ty, local_ptr_ty))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $addr0), (ConstantInt si_llvm_i32_ty, 4))),
    (DefVal $data_lo, (LLVMTrunc (GetNamedOperand $data0), si_llvm_i32_ty)),
    (DefVal $data_hi, (LLVMTrunc
                        (LLVMLShr (GetNamedOperand $data0),
                                  (ConstantInt si_llvm_i64_ty, 32)),
                        si_llvm_i32_ty)),
    (DefVal $old0, (LLVMLoad si_llvm_i32_ty, (GetVal $addr0))),
    (DefVal $old1, (LLVMLoad si_llvm_i32_ty, (GetVal $addr1))),
    (DefVal $cond_lo, (LLVMTrunc
                        (LLVMLShr (GetVal $data_lo),
                                  (ConstantInt si_llvm_i32_ty, 31)),
                        si_llvm_i1_ty)),
    (DefVal $cond_hi, (LLVMTrunc
                        (LLVMLShr (GetVal $data_hi),
                                  (ConstantInt si_llvm_i32_ty, 31)),
                        si_llvm_i1_ty)),
    (DefVal $cleared_lo, (LLVMAnd (GetVal $data_lo),
                                   (ConstantInt si_llvm_i32_ty, 0x7FFFFFFF))),
    (DefVal $cleared_hi, (LLVMAnd (GetVal $data_hi),
                                   (ConstantInt si_llvm_i32_ty, 0x7FFFFFFF))),
    (LLVMStore (LLVMSelect (GetVal $cond_lo),
                            (GetVal $cleared_lo), (GetVal $old0)),
               (GetVal $addr0)),
    (LLVMStore (LLVMSelect (GetVal $cond_hi),
                            (GetVal $cleared_hi), (GetVal $old1)),
               (GetVal $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i32_ty),
                          (GetVal $old0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $old1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
  ]>;

// DS_WRITE_B96: MEM[addr] = data (96-bit, 3 sequential dwords).
// Sources:
//   sea-islands-instruction-set-architecture_0.pdf, pages 240-246.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 492.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 491.
//   rdna4-instruction-set-architecture.pdf, page 610.
def DS_WRITE_B96_sem : InstSISemantic<DS_WRITE_B96> {
  let Semantic = [
    (LLVMStore (GetNamedOperand $data0),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_WRITE_B128: MEM[addr] = data (128-bit, 4 sequential dwords).
def DS_WRITE_B128_sem : InstSISemantic<DS_WRITE_B128> {
  let Semantic = [
    (LLVMStore (GetNamedOperand $data0),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_READ_B96: D = MEM[addr] (96-bit, 3 sequential dwords).
def DS_READ_B96_sem : InstSISemantic<DS_READ_B96> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMLoad (LLVMIntNTy 96),
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret))
  ];
}

// DS_READ_B128: D = MEM[addr] (128-bit, 4 sequential dwords).
def DS_READ_B128_sem : InstSISemantic<DS_READ_B128> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMLoad si_llvm_i128_ty,
      DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret))
  ];
}

//------------------------------------------------------------------------------
// DS 64-bit Atomic Operations (no return)
// tmp = MEM[addr]; MEM[addr] = OP(tmp, data)
//------------------------------------------------------------------------------














//------------------------------------------------------------------------------
// DS 64-bit Atomic Operations (with return)
// D = MEM[addr]; MEM[addr] = OP(D, data); return old value.
//------------------------------------------------------------------------------















//------------------------------------------------------------------------------
// DS 64-bit Compare-and-Store Instructions
//------------------------------------------------------------------------------





//------------------------------------------------------------------------------
// DS CMPSTORE variants (GFX12+ renamed CMPST)
// Identical semantics to their CMPST counterparts.
//------------------------------------------------------------------------------









//------------------------------------------------------------------------------
// DS Float Atomics (64-bit and return variants)
//------------------------------------------------------------------------------










//------------------------------------------------------------------------------
// DS Exchange Instructions (64-bit and WRXCHG2)
//------------------------------------------------------------------------------


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
foreach inst = [DS_WRXCHG2_RTN_B32, DS_WRXCHG2_RTN_B32_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 4)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 4)))),
    (DefVal $old0, (LLVMLoad si_llvm_i32_ty, (GetVal $addr0))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (DefVal $old1, (LLVMLoad si_llvm_i32_ty, (GetVal $addr1))),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i32_ty), (GetVal $old0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $old1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
  ]>;

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
foreach inst = [DS_WRXCHG2_RTN_B64, DS_WRXCHG2_RTN_B64_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 8)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 8)))),
    (DefVal $old0, (LLVMLoad si_llvm_i64_ty, (GetVal $addr0))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (DefVal $old1, (LLVMLoad si_llvm_i64_ty, (GetVal $addr1))),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i64_ty), (GetVal $old0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $old1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
  ]>;

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
foreach inst = [DS_WRXCHG2ST64_RTN_B32, DS_WRXCHG2ST64_RTN_B32_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 256)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 256)))),
    (DefVal $old0, (LLVMLoad si_llvm_i32_ty, (GetVal $addr0))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (DefVal $old1, (LLVMLoad si_llvm_i32_ty, (GetVal $addr1))),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i32_ty), (GetVal $old0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $old1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
  ]>;

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
foreach inst = [DS_WRXCHG2ST64_RTN_B64, DS_WRXCHG2ST64_RTN_B64_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 512)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 512)))),
    (DefVal $old0, (LLVMLoad si_llvm_i64_ty, (GetVal $addr0))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (DefVal $old1, (LLVMLoad si_llvm_i64_ty, (GetVal $addr1))),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1)),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i64_ty), (GetVal $old0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $old1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
  ]>;

//------------------------------------------------------------------------------
// DS Conditional Subtract and Clamp Instructions
//------------------------------------------------------------------------------

// DS_COND_SUB_U32: tmp = MEM[addr]; MEM[addr] = (tmp >= data) ? tmp - data : tmp.
def DS_COND_SUB_U32_sem : InstSISemantic<DS_COND_SUB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $data0)),
                            (LLVMSub (GetVal $tmp), (GetNamedOperand $data0)),
                            (GetVal $tmp)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_COND_SUB_RTN_U32: D = MEM[addr]; MEM[addr] = (D >= data) ? D - data : D; return old.
foreach inst = [DS_COND_SUB_RTN_U32, DS_COND_SUB_RTN_U32_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $old, (LLVMLoad si_llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $old), (GetNamedOperand $data0)),
                            (LLVMSub (GetVal $old), (GetNamedOperand $data0)),
                            (GetVal $old)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ]>;

// DS_SUB_CLAMP_U32: tmp = MEM[addr]; MEM[addr] = (tmp < data) ? 0 : tmp - data.
def DS_SUB_CLAMP_U32_sem : InstSISemantic<DS_SUB_CLAMP_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $tmp), (GetNamedOperand $data0)),
                            (ConstantInt si_llvm_i32_ty, 0),
                            (LLVMSub (GetVal $tmp), (GetNamedOperand $data0))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_SUB_CLAMP_RTN_U32: D = MEM[addr]; MEM[addr] = (D < data) ? 0 : D - data; return old.
foreach inst = [DS_SUB_CLAMP_RTN_U32, DS_SUB_CLAMP_RTN_U32_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $old, (LLVMLoad si_llvm_i32_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $old), (GetNamedOperand $data0)),
                            (ConstantInt si_llvm_i32_ty, 0),
                            (LLVMSub (GetVal $old), (GetNamedOperand $data0))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (GetVal $old))
  ]>;

//------------------------------------------------------------------------------
// DS D16 Read/Write Variants
//------------------------------------------------------------------------------

// DS_READ_U8_D16: Load u8, zero-extend, pack into low 16 bits of vdst, preserve high 16.
def DS_READ_U8_D16_sem : InstSISemantic<DS_READ_U8_D16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $lo, (LLVMZExt (GetVal $tmp), si_llvm_i32_ty)),
    (DefVal $hi, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt si_llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_U8_D16_HI: Load u8, zero-extend, pack into high 16 bits of vdst, preserve low 16.
def DS_READ_U8_D16_HI_sem : InstSISemantic<DS_READ_U8_D16_HI> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $hi, (LLVMShl (LLVMZExt (GetVal $tmp), si_llvm_i32_ty), (ConstantInt si_llvm_i32_ty, 16))),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt si_llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_I8_D16: Load i8, sign-extend to i16, pack into low 16 bits of vdst, preserve high 16.
def DS_READ_I8_D16_sem : InstSISemantic<DS_READ_I8_D16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $sext16, (LLVMSExt (GetVal $tmp), si_llvm_i16_ty)),
    (DefVal $lo, (LLVMZExt (GetVal $sext16), si_llvm_i32_ty)),
    (DefVal $hi, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt si_llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_I8_D16_HI: Load i8, sign-extend to i16, pack into high 16 bits of vdst, preserve low 16.
def DS_READ_I8_D16_HI_sem : InstSISemantic<DS_READ_I8_D16_HI> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i8_ty, DSAddr<(GetNamedOperand $addr),
                                                (GetNamedOperand $offset)>.ret)),
    (DefVal $sext16, (LLVMSExt (GetVal $tmp), si_llvm_i16_ty)),
    (DefVal $hi, (LLVMShl (LLVMZExt (GetVal $sext16), si_llvm_i32_ty), (ConstantInt si_llvm_i32_ty, 16))),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt si_llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_U16_D16: Load u16, pack into low 16 bits of vdst, preserve high 16.
def DS_READ_U16_D16_sem : InstSISemantic<DS_READ_U16_D16> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i16_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $lo, (LLVMZExt (GetVal $tmp), si_llvm_i32_ty)),
    (DefVal $hi, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt si_llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_READ_U16_D16_HI: Load u16, pack into high 16 bits of vdst, preserve low 16.
def DS_READ_U16_D16_HI_sem : InstSISemantic<DS_READ_U16_D16_HI> {
  let Semantic = [
    (DefVal $tmp, (LLVMLoad si_llvm_i16_ty, DSAddr<(GetNamedOperand $addr),
                                                 (GetNamedOperand $offset)>.ret)),
    (DefVal $hi, (LLVMShl (LLVMZExt (GetVal $tmp), si_llvm_i32_ty), (ConstantInt si_llvm_i32_ty, 16))),
    (DefVal $lo, (LLVMAnd (GetNamedOperand $vdst), (ConstantInt si_llvm_i32_ty, 0xFFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $hi), (GetVal $lo)))
  ];
}

// DS_WRITE_B8_D16_HI: Store bits [23:16] of data as byte.
def DS_WRITE_B8_D16_HI_sem : InstSISemantic<DS_WRITE_B8_D16_HI> {
  let Semantic = [
    (LLVMStore (LLVMTrunc (LLVMLShr (GetNamedOperand $data0), (ConstantInt si_llvm_i32_ty, 16)), si_llvm_i8_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

// DS_WRITE_B16_D16_HI: Store bits [31:16] of data as halfword.
def DS_WRITE_B16_D16_HI_sem : InstSISemantic<DS_WRITE_B16_D16_HI> {
  let Semantic = [
    (LLVMStore (LLVMTrunc (LLVMLShr (GetNamedOperand $data0), (ConstantInt si_llvm_i32_ty, 16)), si_llvm_i16_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ];
}

//------------------------------------------------------------------------------
// DS Packed Float Add (PK_ADD)
//------------------------------------------------------------------------------





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
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 4)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 4)))),
    (DefVal $val0, (LLVMLoad si_llvm_i32_ty, (GetVal $addr0))),
    (DefVal $val1, (LLVMLoad si_llvm_i32_ty, (GetVal $addr1))),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i32_ty), (GetVal $val0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $val1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
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
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 8)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 8)))),
    (DefVal $val0, (LLVMLoad si_llvm_i64_ty, (GetVal $addr0))),
    (DefVal $val1, (LLVMLoad si_llvm_i64_ty, (GetVal $addr1))),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i64_ty), (GetVal $val0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $val1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
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
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 256)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 256)))),
    (DefVal $val0, (LLVMLoad si_llvm_i32_ty, (GetVal $addr0))),
    (DefVal $val1, (LLVMLoad si_llvm_i32_ty, (GetVal $addr1))),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i32_ty), (GetVal $val0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $val1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
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
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 512)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 512)))),
    (DefVal $val0, (LLVMLoad si_llvm_i64_ty, (GetVal $addr0))),
    (DefVal $val1, (LLVMLoad si_llvm_i64_ty, (GetVal $addr1))),
    (DefVal $result, (LLVMInsertElement
      (LLVMInsertElement (PoisionVal si_llvm_v2i64_ty), (GetVal $val0), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $val1), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $result))
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
foreach inst = [DS_WRITE2_B32, DS_WRITE2_B32_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 4)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 4)))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1))
  ]>;

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
foreach inst = [DS_WRITE2_B64, DS_WRITE2_B64_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 8)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 8)))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1))
  ]>;

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
foreach inst = [DS_WRITE2ST64_B32, DS_WRITE2ST64_B32_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 256)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 256)))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1))
  ]>;

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
foreach inst = [DS_WRITE2ST64_B64, DS_WRITE2ST64_B64_agpr] in
  def : InstSISemantic<inst, [
    (DefVal $base, (LLVMIntToPtr (GetNamedOperand $addr), local_ptr_ty)),
    (DefVal $addr0, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset0), (ConstantInt si_llvm_i32_ty, 512)))),
    (DefVal $addr1, (LLVMPtrAdd (GetVal $base), (LLVMMul (GetNamedOperand $offset1), (ConstantInt si_llvm_i32_ty, 512)))),
    (LLVMStore (GetNamedOperand $data0), (GetVal $addr0)),
    (LLVMStore (GetNamedOperand $data1), (GetVal $addr1))
  ]>;

//------------------------------------------------------------------------------
// DS SRC2 Instructions (inline asm — implicit second source from LDS)
//------------------------------------------------------------------------------

// DS_ADD_SRC2_U32
// DS_ADD_SRC2_U64
// DS_ADD_SRC2_F32
// DS_SUB_SRC2_U32
// DS_SUB_SRC2_U64
// DS_RSUB_SRC2_U32
// DS_RSUB_SRC2_U64
// DS_INC_SRC2_U32
// DS_INC_SRC2_U64
// DS_DEC_SRC2_U32
// DS_DEC_SRC2_U64
// DS_MIN_SRC2_I32
// DS_MIN_SRC2_I64
// DS_MIN_SRC2_U32
// DS_MIN_SRC2_U64
// DS_MIN_SRC2_F32
// DS_MIN_SRC2_F64
// DS_MAX_SRC2_I32
// DS_MAX_SRC2_I64
// DS_MAX_SRC2_U32
// DS_MAX_SRC2_U64
// DS_MAX_SRC2_F32
// DS_MAX_SRC2_F64
// DS_AND_SRC2_B32
// DS_AND_SRC2_B64
// DS_OR_SRC2_B32
// DS_OR_SRC2_B64
// DS_XOR_SRC2_B32
// DS_XOR_SRC2_B64
// DS_WRITE_SRC2_B32
// DS_WRITE_SRC2_B64
//------------------------------------------------------------------------------
// DS GWS Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_GWS_INIT: Initialize GWS resource.
// DS_GWS_BARRIER: GWS barrier.
// DS_GWS_SEMA_BR: GWS semaphore broadcast release.
// DS_GWS_SEMA_P: GWS semaphore P (wait).
// DS_GWS_SEMA_V: GWS semaphore V (signal).
//------------------------------------------------------------------------------
// DS Ordered Count (inline asm)
//------------------------------------------------------------------------------

// DS_ORDERED_COUNT: Ordered count operation (GDS).
//------------------------------------------------------------------------------
// DS BVH Stack Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_BVH_STACK_RTN_B32: BVH stack pop/push.
// DS_BVH_STACK_PUSH8_POP1_RTN_B32: BVH stack push 8 pop 1.
// DS_BVH_STACK_PUSH8_POP2_RTN_B64: BVH stack push 8 pop 2.
//------------------------------------------------------------------------------
// DS LOAD_TR / READ_TR Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_LOAD_TR4_B64: Load transpose 4 bytes to 64 bits.
// DS_LOAD_TR6_B96: Load transpose 6 bytes to 96 bits.
// DS_LOAD_TR8_B64: Load transpose 8 bytes to 64 bits.
// DS_LOAD_TR16_B128: Load transpose 16 bytes to 128 bits.
// DS_READ_B64_TR_B4: Read 64-bit transpose from 4-byte element.
// DS_READ_B64_TR_B8: Read 64-bit transpose from 8-byte element.
// DS_READ_B64_TR_B16: Read 64-bit transpose from 16-byte element.
// DS_READ_B96_TR_B6: Read 96-bit transpose from 6-byte element.
//------------------------------------------------------------------------------
// DS ADDTID Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_READ_ADDTID_B32: Read with thread-ID-based address offset.
// DS_WRITE_ADDTID_B32: Write with thread-ID-based address offset.
//------------------------------------------------------------------------------
// DS GS_REG Instructions (inline asm, RDNA3+)
//------------------------------------------------------------------------------

// DS_ADD_GS_REG_RTN: Add to GS register with return.
// DS_SUB_GS_REG_RTN: Subtract from GS register with return.
//------------------------------------------------------------------------------
// DS Permute / Intrinsic Instructions
//------------------------------------------------------------------------------

// DS_BPERMUTE_FI_B32: Backward permute with fetch-inactive.
def DS_BPERMUTE_FI_B32_sem : InstSISemantic<DS_BPERMUTE_FI_B32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ds_bpermute_fi_b32, [(GetNamedOperand $addr), (GetNamedOperand $data0)]))
  ];
}

//------------------------------------------------------------------------------
// DS Atomic Barrier Instructions (inline asm)
//------------------------------------------------------------------------------

// DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64: Async barrier arrive (undocumented).
// DS_ATOMIC_BARRIER_ARRIVE_RTN_B64: Barrier arrive with return (undocumented).
//------------------------------------------------------------------------------
// DS CONDXCHG32_RTN_B64 (inline asm — complex conditional dual-dword exchange)
//------------------------------------------------------------------------------

// DS_CONDXCHG32_RTN_B64: Conditional 32-bit exchange on 64-bit pair.
//==============================================================================
// DS _agpr variants (Phase 5)
// AGPR destination forms of existing DS atomics. Same IR semantics as the
// VGPR variant — replicated verbatim from the corresponding base record.
//==============================================================================

//------------------------------------------------------------------------------
// Group B: integer binary atomics (ADD / SUB / RSUB / AND / OR / XOR)
// Each entry covers the NoRet record + the RTN record's base/agpr pair.
// RSUB is the outlier — operand order is reversed (data - tmp vs tmp - data).
//------------------------------------------------------------------------------

class DSBinAtomicDesc<LLVMOp op, SILLVMType ty, bit reverse,
                     InstSI noret, InstSI rtn, InstSI rtn_agpr> {
  LLVMOp Op = op;
  SILLVMType Ty = ty;
  bit Reverse = reverse;
  InstSI NoRet = noret;
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  DSBinAtomicDesc<LLVMAdd, si_llvm_i32_ty, 0, DS_ADD_U32,  DS_ADD_RTN_U32,  DS_ADD_RTN_U32_agpr>,
  DSBinAtomicDesc<LLVMSub, si_llvm_i32_ty, 0, DS_SUB_U32,  DS_SUB_RTN_U32,  DS_SUB_RTN_U32_agpr>,
  DSBinAtomicDesc<LLVMSub, si_llvm_i32_ty, 1, DS_RSUB_U32, DS_RSUB_RTN_U32, DS_RSUB_RTN_U32_agpr>,
  DSBinAtomicDesc<LLVMAnd, si_llvm_i32_ty, 0, DS_AND_B32,  DS_AND_RTN_B32,  DS_AND_RTN_B32_agpr>,
  DSBinAtomicDesc<LLVMOr,  si_llvm_i32_ty, 0, DS_OR_B32,   DS_OR_RTN_B32,   DS_OR_RTN_B32_agpr>,
  DSBinAtomicDesc<LLVMXor, si_llvm_i32_ty, 0, DS_XOR_B32,  DS_XOR_RTN_B32,  DS_XOR_RTN_B32_agpr>,
  DSBinAtomicDesc<LLVMAdd, si_llvm_i64_ty, 0, DS_ADD_U64,  DS_ADD_RTN_U64,  DS_ADD_RTN_U64_agpr>,
  DSBinAtomicDesc<LLVMSub, si_llvm_i64_ty, 0, DS_SUB_U64,  DS_SUB_RTN_U64,  DS_SUB_RTN_U64_agpr>,
  DSBinAtomicDesc<LLVMSub, si_llvm_i64_ty, 1, DS_RSUB_U64, DS_RSUB_RTN_U64, DS_RSUB_RTN_U64_agpr>,
  DSBinAtomicDesc<LLVMAnd, si_llvm_i64_ty, 0, DS_AND_B64,  DS_AND_RTN_B64,  DS_AND_RTN_B64_agpr>,
  DSBinAtomicDesc<LLVMOr,  si_llvm_i64_ty, 0, DS_OR_B64,   DS_OR_RTN_B64,   DS_OR_RTN_B64_agpr>,
  DSBinAtomicDesc<LLVMXor, si_llvm_i64_ty, 0, DS_XOR_B64,  DS_XOR_RTN_B64,  DS_XOR_RTN_B64_agpr>
] in {
  def : InstSISemantic<desc.NoRet, [
    (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
    (LLVMStore !if(desc.Reverse,
                   (desc.Op (GetNamedOperand $data0), (GetVal $tmp)),
                   (desc.Op (GetVal $tmp), (GetNamedOperand $data0))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ]>;
  foreach rtnInst = desc.Rtn in
    def : InstSISemantic<rtnInst, [
      (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore !if(desc.Reverse,
                     (desc.Op (GetNamedOperand $data0), (GetVal $tmp)),
                     (desc.Op (GetVal $tmp), (GetNamedOperand $data0))),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ]>;
}


//------------------------------------------------------------------------------
// Group C: integer MIN/MAX (signed and unsigned, 32/64-bit)
//   MEM = (cmp(tmp, data) ? tmp : data); RTN variant also writes vdst = old
// Differs only in the comparison predicate and the operand width.
//------------------------------------------------------------------------------

class DSMinMaxIDesc<LLVMCmpPredicate pred, SILLVMType ty,
                    InstSI noret, InstSI rtn, InstSI rtn_agpr> {
  LLVMCmpPredicate Pred = pred;
  SILLVMType Ty = ty;
  InstSI NoRet = noret;
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  DSMinMaxIDesc<LLVMICmpSLT, si_llvm_i32_ty, DS_MIN_I32, DS_MIN_RTN_I32, DS_MIN_RTN_I32_agpr>,
  DSMinMaxIDesc<LLVMICmpSGT, si_llvm_i32_ty, DS_MAX_I32, DS_MAX_RTN_I32, DS_MAX_RTN_I32_agpr>,
  DSMinMaxIDesc<LLVMICmpULT, si_llvm_i32_ty, DS_MIN_U32, DS_MIN_RTN_U32, DS_MIN_RTN_U32_agpr>,
  DSMinMaxIDesc<LLVMICmpUGT, si_llvm_i32_ty, DS_MAX_U32, DS_MAX_RTN_U32, DS_MAX_RTN_U32_agpr>,
  DSMinMaxIDesc<LLVMICmpSLT, si_llvm_i64_ty, DS_MIN_I64, DS_MIN_RTN_I64, DS_MIN_RTN_I64_agpr>,
  DSMinMaxIDesc<LLVMICmpSGT, si_llvm_i64_ty, DS_MAX_I64, DS_MAX_RTN_I64, DS_MAX_RTN_I64_agpr>,
  DSMinMaxIDesc<LLVMICmpULT, si_llvm_i64_ty, DS_MIN_U64, DS_MIN_RTN_U64, DS_MIN_RTN_U64_agpr>,
  DSMinMaxIDesc<LLVMICmpUGT, si_llvm_i64_ty, DS_MAX_U64, DS_MAX_RTN_U64, DS_MAX_RTN_U64_agpr>
] in {
  def : InstSISemantic<desc.NoRet, [
    (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp desc.Pred, (GetVal $tmp), (GetNamedOperand $data0)),
                            (GetVal $tmp), (GetNamedOperand $data0)),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ]>;
  foreach rtnInst = desc.Rtn in
    def : InstSISemantic<rtnInst, [
      (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (LLVMSelect (LLVMCmp desc.Pred, (GetVal $tmp), (GetNamedOperand $data0)),
                              (GetVal $tmp), (GetNamedOperand $data0)),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ]>;
}


//------------------------------------------------------------------------------
// Group D: FP atomic ADD / MIN / MAX (F32 / F64)
//   Load as iN, bitcast to float/double, apply op, bitcast back, store.
//   MIN/MAX use IEEE llvm.minnum / llvm.maxnum (post-fix #4 in audit memory).
//------------------------------------------------------------------------------

class DSFpAddDesc<SILLVMType iTy, SILLVMType fpTy,
                  InstSI noret, InstSI rtn, InstSI rtn_agpr> {
  SILLVMType ITy = iTy;
  SILLVMType FpTy = fpTy;
  InstSI NoRet = noret;
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  DSFpAddDesc<si_llvm_i32_ty, si_llvm_float_ty,  DS_ADD_F32, DS_ADD_RTN_F32, DS_ADD_RTN_F32_agpr>,
  DSFpAddDesc<si_llvm_i64_ty, si_llvm_double_ty, DS_ADD_F64, DS_ADD_RTN_F64, DS_ADD_RTN_F64_agpr>
] in {
  def : InstSISemantic<desc.NoRet, [
    (DefVal $tmp, (LLVMLoad desc.ITy, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMFAdd (LLVMBitCast (GetVal $tmp), desc.FpTy),
                                       (LLVMBitCast (GetNamedOperand $data0), desc.FpTy)),
                 desc.ITy),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ]>;
  foreach rtnInst = desc.Rtn in
    def : InstSISemantic<rtnInst, [
      (DefVal $tmp, (LLVMLoad desc.ITy, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (LLVMBitCast (LLVMFAdd (LLVMBitCast (GetVal $tmp), desc.FpTy),
                                         (LLVMBitCast (GetNamedOperand $data0), desc.FpTy)),
                   desc.ITy),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ]>;
}

class DSFpMinMaxDesc<Intrinsic intr, SILLVMType iTy, SILLVMType fpTy,
                     InstSI noret, InstSI rtn, InstSI rtn_agpr> {
  Intrinsic Intr = intr;
  SILLVMType ITy = iTy;
  SILLVMType FpTy = fpTy;
  InstSI NoRet = noret;
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  DSFpMinMaxDesc<int_minnum, si_llvm_i32_ty, si_llvm_float_ty,  DS_MIN_F32, DS_MIN_RTN_F32, DS_MIN_RTN_F32_agpr>,
  DSFpMinMaxDesc<int_maxnum, si_llvm_i32_ty, si_llvm_float_ty,  DS_MAX_F32, DS_MAX_RTN_F32, DS_MAX_RTN_F32_agpr>,
  DSFpMinMaxDesc<int_minnum, si_llvm_i64_ty, si_llvm_double_ty, DS_MIN_F64, DS_MIN_RTN_F64, DS_MIN_RTN_F64_agpr>,
  DSFpMinMaxDesc<int_maxnum, si_llvm_i64_ty, si_llvm_double_ty, DS_MAX_F64, DS_MAX_RTN_F64, DS_MAX_RTN_F64_agpr>
] in {
  def : InstSISemantic<desc.NoRet, [
    (DefVal $tmp, (LLVMLoad desc.ITy, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
    (LLVMStore (LLVMBitCast (LLVMBinaryIntrinsic desc.Intr,
                  (LLVMBitCast (GetVal $tmp), desc.FpTy),
                  (LLVMBitCast (GetNamedOperand $data0), desc.FpTy)),
                 desc.ITy),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ]>;
  foreach rtnInst = desc.Rtn in
    def : InstSISemantic<rtnInst, [
      (DefVal $tmp, (LLVMLoad desc.ITy, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (LLVMBitCast (LLVMBinaryIntrinsic desc.Intr,
                    (LLVMBitCast (GetVal $tmp), desc.FpTy),
                    (LLVMBitCast (GetNamedOperand $data0), desc.FpTy)),
                   desc.ITy),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ]>;
}


//------------------------------------------------------------------------------
// Group E: INC / DEC integer atomics (U32 / U64, NoRet / RTN)
//   INC: MEM = (tmp >= data) ? 0 : tmp + 1
//   DEC: MEM = (tmp == 0 || tmp > data) ? data : tmp - 1
//------------------------------------------------------------------------------

class DSIncDecDesc<bit isInc, SILLVMType ty,
                   InstSI noret, InstSI rtn, InstSI rtn_agpr> {
  bit IsInc = isInc;
  SILLVMType Ty = ty;
  InstSI NoRet = noret;
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  DSIncDecDesc<1, si_llvm_i32_ty, DS_INC_U32, DS_INC_RTN_U32, DS_INC_RTN_U32_agpr>,
  DSIncDecDesc<0, si_llvm_i32_ty, DS_DEC_U32, DS_DEC_RTN_U32, DS_DEC_RTN_U32_agpr>,
  DSIncDecDesc<1, si_llvm_i64_ty, DS_INC_U64, DS_INC_RTN_U64, DS_INC_RTN_U64_agpr>,
  DSIncDecDesc<0, si_llvm_i64_ty, DS_DEC_U64, DS_DEC_RTN_U64, DS_DEC_RTN_U64_agpr>
] in {
  def : InstSISemantic<desc.NoRet, [
    (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
    (LLVMStore !if(desc.IsInc,
                   (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $data0)),
                                (ConstantInt desc.Ty, 0),
                                (LLVMAdd (GetVal $tmp), (ConstantInt desc.Ty, 1))),
                   (LLVMSelect (LLVMOr (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (ConstantInt desc.Ty, 0)),
                                        (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $data0))),
                                (GetNamedOperand $data0),
                                (LLVMSub (GetVal $tmp), (ConstantInt desc.Ty, 1)))),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ]>;
  foreach rtnInst = desc.Rtn in
    def : InstSISemantic<rtnInst, [
      (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore !if(desc.IsInc,
                     (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp), (GetNamedOperand $data0)),
                                  (ConstantInt desc.Ty, 0),
                                  (LLVMAdd (GetVal $tmp), (ConstantInt desc.Ty, 1))),
                     (LLVMSelect (LLVMOr (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (ConstantInt desc.Ty, 0)),
                                          (LLVMCmp LLVMICmpUGT, (GetVal $tmp), (GetNamedOperand $data0))),
                                  (GetNamedOperand $data0),
                                  (LLVMSub (GetVal $tmp), (ConstantInt desc.Ty, 1)))),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ]>;
}


//------------------------------------------------------------------------------
// Group J: MSKOR — masked OR
//   MEM = (tmp & ~data0) | data1   (data0 = mask, data1 = bits to OR in)
// Implemented as (tmp & (data0 ^ -1)) | data1.
//------------------------------------------------------------------------------

class DSMskorDesc<SILLVMType ty,
                  InstSI noret, InstSI noret_agpr,
                  InstSI rtn, InstSI rtn_agpr> {
  SILLVMType Ty = ty;
  list<InstSI> NoRet = [noret, noret_agpr];
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  DSMskorDesc<si_llvm_i32_ty, DS_MSKOR_B32, DS_MSKOR_B32_agpr,
                              DS_MSKOR_RTN_B32, DS_MSKOR_RTN_B32_agpr>,
  DSMskorDesc<si_llvm_i64_ty, DS_MSKOR_B64, DS_MSKOR_B64_agpr,
                              DS_MSKOR_RTN_B64, DS_MSKOR_RTN_B64_agpr>
] in {
  foreach noretInst = desc.NoRet in
    def : InstSISemantic<noretInst, [
      (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (LLVMOr (LLVMAnd (GetVal $tmp),
                                    (LLVMXor (GetNamedOperand $data0), (ConstantSignedInt desc.Ty, -1))),
                          (GetNamedOperand $data1)),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
    ]>;
  // The base record (DS_MSKOR_B32) has no agpr alias; handle base + agpr alias for RTN.
  foreach rtnInst = desc.Rtn in
    def : InstSISemantic<rtnInst, [
      (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (LLVMOr (LLVMAnd (GetVal $tmp),
                                    (LLVMXor (GetNamedOperand $data0), (ConstantSignedInt desc.Ty, -1))),
                          (GetNamedOperand $data1)),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ]>;
}


//------------------------------------------------------------------------------
// Group K: DS_WRXCHG_RTN_B32 / DS_WRXCHG_RTN_B64 — single-address swap.
//   old = MEM[addr]; MEM[addr] = data0; vdst = old
//------------------------------------------------------------------------------

class DSWrxchgDesc<SILLVMType ty, InstSI base, InstSI agpr> {
  SILLVMType Ty = ty;
  list<InstSI> Insts = [base, agpr];
}

foreach desc = [
  DSWrxchgDesc<si_llvm_i32_ty, DS_WRXCHG_RTN_B32, DS_WRXCHG_RTN_B32_agpr>,
  DSWrxchgDesc<si_llvm_i64_ty, DS_WRXCHG_RTN_B64, DS_WRXCHG_RTN_B64_agpr>
] in
  foreach inst = desc.Insts in
    def : InstSISemantic<inst, [
      (DefVal $tmp, (LLVMLoad desc.Ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (GetNamedOperand $data0),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ]>;


//------------------------------------------------------------------------------
// Group M: DS_PK_ADD packed half/bfloat add (CDNA3+, RDNA4).
//   Memory and data are i32; reinterpret as <2 x f16> or <2 x bf16>, element
//   add, bitcast back to i32. RTN form also returns the pre-op memory value.
//------------------------------------------------------------------------------

class DSPkAddDesc<SILLVMType vTy, InstSI noret, InstSI rtn, InstSI rtn_agpr> {
  SILLVMType VTy = vTy;
  InstSI NoRet = noret;
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  DSPkAddDesc<si_llvm_v2f16_ty,  DS_PK_ADD_F16,  DS_PK_ADD_RTN_F16,  DS_PK_ADD_RTN_F16_agpr>,
  DSPkAddDesc<si_llvm_v2bf16_ty, DS_PK_ADD_BF16, DS_PK_ADD_RTN_BF16, DS_PK_ADD_RTN_BF16_agpr>
] in {
  def : InstSISemantic<desc.NoRet, [
    (DefVal $tmp, (LLVMLoad si_llvm_i32_ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
    (DefVal $mem_v, (LLVMBitCast (GetVal $tmp), desc.VTy)),
    (DefVal $data_v, (LLVMBitCast (GetNamedOperand $data0), desc.VTy)),
    (DefVal $result, (LLVMFAdd (GetVal $mem_v), (GetVal $data_v))),
    (LLVMStore (LLVMBitCast (GetVal $result), si_llvm_i32_ty),
               DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
  ]>;
  foreach rtnInst = desc.Rtn in
    def : InstSISemantic<rtnInst, [
      (DefVal $old, (LLVMLoad si_llvm_i32_ty, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (DefVal $mem_v, (LLVMBitCast (GetVal $old), desc.VTy)),
      (DefVal $data_v, (LLVMBitCast (GetNamedOperand $data0), desc.VTy)),
      (DefVal $result, (LLVMFAdd (GetVal $mem_v), (GetVal $data_v))),
      (LLVMStore (LLVMBitCast (GetVal $result), si_llvm_i32_ty),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $old))
    ]>;
}

//------------------------------------------------------------------------------
// DS transpose loads (CDNA4 DS_READ_B*_TR_*; RDNA4 DS_LOAD_TR*_*)
// Each lowers to its dedicated LLVM intrinsic on a local pointer. The result
// type is the lane-strided vector the intrinsic naturally returns, per LLVM's
// DSLoadTrPat selection patterns in DSInstructions.td.
//   DS_READ_B64_TR_B4  / DS_LOAD_TR4_B64    -> v2i32 via int_amdgcn_ds_(read|load)_tr4_b64
//   DS_READ_B64_TR_B8  / DS_LOAD_TR8_B64    -> v2i32 via int_amdgcn_ds_(read|load)_tr8_b64
//   DS_READ_B96_TR_B6  / DS_LOAD_TR6_B96    -> v3i32 via int_amdgcn_ds_(read|load)_tr6_b96
//   DS_READ_B64_TR_B16                      -> v4i16 via int_amdgcn_ds_read_tr16_b64
//   DS_LOAD_TR16_B128                       -> v8i16 via int_amdgcn_ds_load_tr16_b128
//------------------------------------------------------------------------------

class DSTrLoadDesc<InstSI inst, SILLVMType vTy, Intrinsic intr> {
  InstSI Inst = inst;
  SILLVMType VTy = vTy;
  Intrinsic Intr = intr;
}

foreach desc = [
  DSTrLoadDesc<DS_READ_B64_TR_B4,  si_llvm_v2i32_ty, int_amdgcn_ds_read_tr4_b64>,
  DSTrLoadDesc<DS_READ_B64_TR_B8,  si_llvm_v2i32_ty, int_amdgcn_ds_read_tr8_b64>,
  DSTrLoadDesc<DS_READ_B64_TR_B16, si_llvm_v4i16_ty, int_amdgcn_ds_read_tr16_b64>,
  DSTrLoadDesc<DS_READ_B96_TR_B6,  si_llvm_v3i32_ty, int_amdgcn_ds_read_tr6_b96>,
  DSTrLoadDesc<DS_LOAD_TR4_B64,    si_llvm_v2i32_ty, int_amdgcn_ds_load_tr4_b64>,
  DSTrLoadDesc<DS_LOAD_TR6_B96,    si_llvm_v3i32_ty, int_amdgcn_ds_load_tr6_b96>,
  DSTrLoadDesc<DS_LOAD_TR8_B64,    si_llvm_v2i32_ty, int_amdgcn_ds_load_tr8_b64>,
  DSTrLoadDesc<DS_LOAD_TR16_B128,  si_llvm_v8i16_ty, int_amdgcn_ds_load_tr16_b128>
] in
  def : InstSISemantic<desc.Inst, [
    (SetNamedOperand $vdst, (LLVMIntrinsic desc.VTy, desc.Intr,
      [DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret]))
  ]>;

//------------------------------------------------------------------------------
// DS BVH stack helpers (RDNA4 ray-tracing) — DS_BVH_STACK_*
// Each lowers to its dedicated LLVM intrinsic. The intrinsic returns a
// {vdst, i32 addr} struct: vdst is the popped/last node pointer, and the i32
// is the new stack-address-pointer value (the in-out $addr operand).
// See DSInstructions.td DS_BVH_STACK class and AMDGPUInstructionSelector.cpp.
//   DS_BVH_STACK_RTN_B32              -> int_amdgcn_ds_bvh_stack_rtn              ({i32, i32}, data1 <4xi32>)
//   DS_BVH_STACK_PUSH8_POP1_RTN_B32   -> int_amdgcn_ds_bvh_stack_push8_pop1_rtn   ({i32, i32}, data1 <8xi32>)
//   DS_BVH_STACK_PUSH8_POP2_RTN_B64   -> int_amdgcn_ds_bvh_stack_push8_pop2_rtn   ({i64, i32}, data1 <8xi32>)
// The intrinsic is non-overloaded, so CreateIntrinsic auto-infers the return
// type (use the 2-arg form by omitting the leading type argument).
//------------------------------------------------------------------------------

class DSBvhStackDesc<InstSI inst, Intrinsic intr, SILLVMType data1Ty> {
  InstSI Inst = inst;
  Intrinsic Intr = intr;
  SILLVMType Data1Ty = data1Ty;
}

foreach desc = [
  DSBvhStackDesc<DS_BVH_STACK_RTN_B32,
                 int_amdgcn_ds_bvh_stack_rtn,            si_llvm_v4i32_ty>,
  DSBvhStackDesc<DS_BVH_STACK_PUSH8_POP1_RTN_B32,
                 int_amdgcn_ds_bvh_stack_push8_pop1_rtn, si_llvm_v8i32_ty>,
  DSBvhStackDesc<DS_BVH_STACK_PUSH8_POP2_RTN_B64,
                 int_amdgcn_ds_bvh_stack_push8_pop2_rtn, si_llvm_v8i32_ty>
] in
  def : InstSISemantic<desc.Inst, [
    (DefVal $result, (LLVMIntrinsic desc.Intr,
      [(GetNamedOperand $addr_in, si_llvm_i32_ty),
       (GetNamedOperand $data0, si_llvm_i32_ty),
       (GetNamedOperand $data1, desc.Data1Ty),
       (GetNamedOperand $offset, si_llvm_i32_ty)])),
    (SetNamedOperand $vdst, (LLVMExtractValue (GetVal $result), [0])),
    (SetNamedOperand $addr, (LLVMExtractValue (GetVal $result), [1]))
  ]>;

//------------------------------------------------------------------------------
// DS atomic barrier-arrive helpers (RDNA4 gfx1250+) — selection patterns in
// DSInstructions.td:907-917 (GCNPat → DS_ATOMIC_*_ARRIVE_*).
//   DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64
//     -> int_amdgcn_ds_atomic_async_barrier_arrive_b64(local_ptr) -> void
//     (NoRet pseudo: no $vdst, no $data0; pointer constructed from DSAddr.)
//   DS_ATOMIC_BARRIER_ARRIVE_RTN_B64 (+ _agpr alias on CDNA)
//     -> int_amdgcn_ds_atomic_barrier_arrive_rtn_b64(local_ptr, i64) -> i64
//------------------------------------------------------------------------------

def DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64_sem
  : InstSISemantic<DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64, [
      (LLVMIntrinsic si_llvm_void_ty,
                     int_amdgcn_ds_atomic_async_barrier_arrive_b64,
                     [DSAddr<(GetNamedOperand $addr),
                             (GetNamedOperand $offset)>.ret])
    ]>;

foreach inst = [DS_ATOMIC_BARRIER_ARRIVE_RTN_B64,
                DS_ATOMIC_BARRIER_ARRIVE_RTN_B64_agpr] in
  def : InstSISemantic<inst, [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i64_ty,
                     int_amdgcn_ds_atomic_barrier_arrive_rtn_b64,
                     [DSAddr<(GetNamedOperand $addr),
                             (GetNamedOperand $offset)>.ret,
                      (GetNamedOperand $data0, si_llvm_i64_ty)]))
  ]>;

//------------------------------------------------------------------------------
// DS GWS (Global Wave Sync) helpers — GDS-only semaphore primitives.
//   Mapping is 1:1 between pseudo and intrinsic (see
//   AMDGPUInstructionSelector.cpp gwsIntrinToOpcode).
//   DS_GWS_1D variants (init, barrier, sema_br): (i32 data0, i32 offset) -> void
//   DS_GWS_0D variants (sema_v, sema_p, sema_release_all): (i32 offset) -> void
// Operands need explicit i32 typing because the Offset operand is 16-bit
// natively but the intrinsics expect i32.
//------------------------------------------------------------------------------

class DSGws1DDesc<InstSI inst, Intrinsic intr> {
  InstSI Inst = inst;
  Intrinsic Intr = intr;
}

foreach desc = [
  DSGws1DDesc<DS_GWS_INIT,    int_amdgcn_ds_gws_init>,
  DSGws1DDesc<DS_GWS_BARRIER, int_amdgcn_ds_gws_barrier>,
  DSGws1DDesc<DS_GWS_SEMA_BR, int_amdgcn_ds_gws_sema_br>
] in
  def : InstSISemantic<desc.Inst, [
    (LLVMIntrinsic si_llvm_void_ty, desc.Intr,
      [(GetNamedOperand $data0, si_llvm_i32_ty),
       (GetNamedOperand $offset, si_llvm_i32_ty)])
  ]>;

class DSGws0DDesc<InstSI inst, Intrinsic intr> {
  InstSI Inst = inst;
  Intrinsic Intr = intr;
}

foreach desc = [
  DSGws0DDesc<DS_GWS_SEMA_V,           int_amdgcn_ds_gws_sema_v>,
  DSGws0DDesc<DS_GWS_SEMA_P,           int_amdgcn_ds_gws_sema_p>,
  DSGws0DDesc<DS_GWS_SEMA_RELEASE_ALL, int_amdgcn_ds_gws_sema_release_all>
] in
  def : InstSISemantic<desc.Inst, [
    (LLVMIntrinsic si_llvm_void_ty, desc.Intr,
      [(GetNamedOperand $offset, si_llvm_i32_ty)])
  ]>;

//------------------------------------------------------------------------------
// DS GS_REG_RTN — atomic update of a GDS_STRMOUT_* register bank entry.
// Operands: $vdst (i64), $data0 (i32 data), $offset (i32 ImmArg selecting
// register slot). Intrinsic return is `llvm_anyint_ty` — overloaded by
// declared return type; the pseudo uses VGPROp_64 for $vdst, so we lock to
// i64 by passing it as the overload retTy. RDNA3 only.
//------------------------------------------------------------------------------

foreach desc = [
  DSGws1DDesc<DS_ADD_GS_REG_RTN, int_amdgcn_ds_add_gs_reg_rtn>,
  DSGws1DDesc<DS_SUB_GS_REG_RTN, int_amdgcn_ds_sub_gs_reg_rtn>
] in
  def : InstSISemantic<desc.Inst, [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i64_ty, desc.Intr,
        [(GetNamedOperand $data0, si_llvm_i32_ty),
         (GetNamedOperand $offset, si_llvm_i32_ty)]))
  ]>;

//------------------------------------------------------------------------------
// DS_{READ,WRITE}_ADDTID_B32 — per-lane base-plus-stride LDS access.
//   addr_b16 = ({OFFSET1, OFFSET0} + M0[15:0]) truncated to 16 bits
//   byte_addr = (i32)addr_b16 + laneID * 4
//   MEM[byte_addr].b32 = DATA0  (WRITE)
//   RETURN_DATA      = MEM[byte_addr].b32  (READ)
// laneID is built from int_amdgcn_mbcnt_lo/hi (works for wave32 and wave64).
// The 16-bit truncation of the (offset + M0_lo) sum is modeled with an
// explicit AND of 0xFFFF after the add.
//------------------------------------------------------------------------------

class DSAddTidAddr {
  dag ret = (LLVMIntToPtr
    (LLVMAdd
      (LLVMAnd
        (LLVMAdd (GetNamedOperand $offset, si_llvm_i32_ty),
                 (LLVMAnd (ImplicitUse M0),
                          (ConstantInt si_llvm_i32_ty, 0xFFFF))),
        (ConstantInt si_llvm_i32_ty, 0xFFFF)),
      (LLVMMul
        (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_hi,
          [(ConstantSignedInt si_llvm_i32_ty, -1),
           (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_lo,
             [(ConstantSignedInt si_llvm_i32_ty, -1),
              (ConstantInt si_llvm_i32_ty, 0)])]),
        (ConstantInt si_llvm_i32_ty, 4))),
    local_ptr_ty);
}

def DS_READ_ADDTID_B32_sem : InstSISemantic<DS_READ_ADDTID_B32, [
  (SetNamedOperand $vdst, (LLVMLoad si_llvm_i32_ty, DSAddTidAddr<>.ret))
]>;

def DS_WRITE_ADDTID_B32_sem : InstSISemantic<DS_WRITE_ADDTID_B32, [
  (LLVMStore (GetNamedOperand $data0, si_llvm_i32_ty), DSAddTidAddr<>.ret)
]>;

//------------------------------------------------------------------------------
// Group F: DS_CMPST_* (pre-GFX11) + DS_CMPSTORE_* (GFX11+) — compare-and-store
//
//   CMPST: cmp = data0, src = data1   "src and cmp are OPPOSITE of BUFFER_ATOMIC_CMPSWAP"
//   CMPSTORE: cmp = data1, src = data0   "src and cmp AGREE with BUFFER_ATOMIC_CMPSWAP"
//
// The GFX11 rename also swapped operand roles. Both:
//   MEM = (cmp(old, <cmp-operand>) ? <src-operand> : old)
//   RTN form additionally writes vdst = old.
// Float variants bitcast the i32/i64 container to float/double for the
// compare. Integer variants compare in the container type directly.
//------------------------------------------------------------------------------

class DSCmpStoreDesc<bit isFloat, SILLVMType iTy, SILLVMType fpTy,
                     bit cmpFirst,
                     InstSI noret, InstSI noret_agpr,
                     InstSI rtn, InstSI rtn_agpr> {
  bit IsFloat = isFloat;
  SILLVMType ITy = iTy;
  SILLVMType FpTy = fpTy;
  bit CmpFirst = cmpFirst;          // 1 = CMPST (cmp=data0), 0 = CMPSTORE (cmp=data1)
  list<InstSI> NoRet = [noret, noret_agpr];
  list<InstSI> Rtn = [rtn, rtn_agpr];
}

foreach desc = [
  // CMPST family (pre-GFX11): cmp=data0, src=data1
  DSCmpStoreDesc<0, si_llvm_i32_ty, si_llvm_float_ty,  1,
    DS_CMPST_B32, DS_CMPST_B32_agpr, DS_CMPST_RTN_B32, DS_CMPST_RTN_B32_agpr>,
  DSCmpStoreDesc<0, si_llvm_i64_ty, si_llvm_double_ty, 1,
    DS_CMPST_B64, DS_CMPST_B64_agpr, DS_CMPST_RTN_B64, DS_CMPST_RTN_B64_agpr>,
  DSCmpStoreDesc<1, si_llvm_i32_ty, si_llvm_float_ty,  1,
    DS_CMPST_F32, DS_CMPST_F32_agpr, DS_CMPST_RTN_F32, DS_CMPST_RTN_F32_agpr>,
  DSCmpStoreDesc<1, si_llvm_i64_ty, si_llvm_double_ty, 1,
    DS_CMPST_F64, DS_CMPST_F64_agpr, DS_CMPST_RTN_F64, DS_CMPST_RTN_F64_agpr>,
  // CMPSTORE family (GFX11+): cmp=data1, src=data0
  DSCmpStoreDesc<0, si_llvm_i32_ty, si_llvm_float_ty,  0,
    DS_CMPSTORE_B32, DS_CMPSTORE_B32_agpr, DS_CMPSTORE_RTN_B32, DS_CMPSTORE_RTN_B32_agpr>,
  DSCmpStoreDesc<0, si_llvm_i64_ty, si_llvm_double_ty, 0,
    DS_CMPSTORE_B64, DS_CMPSTORE_B64_agpr, DS_CMPSTORE_RTN_B64, DS_CMPSTORE_RTN_B64_agpr>,
  DSCmpStoreDesc<1, si_llvm_i32_ty, si_llvm_float_ty,  0,
    DS_CMPSTORE_F32, DS_CMPSTORE_F32_agpr, DS_CMPSTORE_RTN_F32, DS_CMPSTORE_RTN_F32_agpr>,
  DSCmpStoreDesc<1, si_llvm_i64_ty, si_llvm_double_ty, 0,
    DS_CMPSTORE_F64, DS_CMPSTORE_F64_agpr, DS_CMPSTORE_RTN_F64, DS_CMPSTORE_RTN_F64_agpr>
] in {
  foreach inst = desc.NoRet in
    def : InstSISemantic<inst, [
      (DefVal $old, (LLVMLoad desc.ITy, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (LLVMSelect
                   !if(desc.IsFloat,
                       (LLVMFCmp LLVMFCmpOEQ,
                         (LLVMBitCast (GetVal $old), desc.FpTy),
                         (LLVMBitCast !if(desc.CmpFirst, (GetNamedOperand $data0), (GetNamedOperand $data1)), desc.FpTy)),
                       (LLVMCmp LLVMICmpEQ,
                         (GetVal $old),
                         !if(desc.CmpFirst, (GetNamedOperand $data0), (GetNamedOperand $data1)))),
                   !if(desc.CmpFirst, (GetNamedOperand $data1), (GetNamedOperand $data0)),
                   (GetVal $old)),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)
    ]>;
  foreach inst = desc.Rtn in
    def : InstSISemantic<inst, [
      (DefVal $old, (LLVMLoad desc.ITy, DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret)),
      (LLVMStore (LLVMSelect
                   !if(desc.IsFloat,
                       (LLVMFCmp LLVMFCmpOEQ,
                         (LLVMBitCast (GetVal $old), desc.FpTy),
                         (LLVMBitCast !if(desc.CmpFirst, (GetNamedOperand $data0), (GetNamedOperand $data1)), desc.FpTy)),
                       (LLVMCmp LLVMICmpEQ,
                         (GetVal $old),
                         !if(desc.CmpFirst, (GetNamedOperand $data0), (GetNamedOperand $data1)))),
                   !if(desc.CmpFirst, (GetNamedOperand $data1), (GetNamedOperand $data0)),
                   (GetVal $old)),
                 DSAddr<(GetNamedOperand $addr), (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (GetVal $old))
    ]>;
}

//==============================================================================
// DS_*_SRC2_* family — pre-RDNA A-relative second-source addressing
//==============================================================================
// Source: gcn3-instruction-set-architecture.pdf, pages 254-259
//   (also in southern-islands/sea-islands; removed in RDNA).
//
// For each DS_*_SRC2_* opcode, the hardware reads/writes two LDS locations:
//   - A = ADDR VGPR (byte offset into LDS) — already produced by VGPR;
//         used directly as the byte address into A's slot.
//   - B = A + 4 * displ, where the 15-bit signed `displ` is selected by
//         offset1[7] (= bit 15 of the merged 16-bit $offset):
//             offset1[7] == 0:  displ = sign_extend({offset1[6:0], offset0})
//             offset1[7] == 1:  displ = sign_extend({A[31], A[31:17]})
//
// Encoded as sign-extend(15-bit) idioms in LLVM IR:
//   displ_imm = ($offset << 17) ashr 17   (widens bit 14 of offset to bit 31)
//   displ_A   = ($addr ashr 17)           (sign-extends from A[31:17])
//
// SRC2 is LDS-only in all generations (no GDS encoding). We use local_ptr_ty
// directly. M0 LDS-size clamping (gfx6-gfx8) would also apply here, but the
// pseudos retain `$offset` as a raw immediate and we model the byte address
// arithmetic directly — matching how the SRC2 spec itself describes the
// operation.

class DSAddrSrc2B<dag addr, dag offset> {
  // 15-bit signed displacement selected by offset[15]:
  dag displ = (LLVMSelect
    (LLVMTrunc (LLVMLShr offset, (ConstantInt si_llvm_i32_ty, 15)),
               si_llvm_i1_ty),
    (LLVMAShr addr, (ConstantInt si_llvm_i32_ty, 17)),
    (LLVMAShr (LLVMShl offset, (ConstantInt si_llvm_i32_ty, 17)),
              (ConstantInt si_llvm_i32_ty, 17)));
  // B = A + 4 * displ
  dag ret = (LLVMIntToPtr
              (LLVMAdd addr,
                (LLVMShl displ, (ConstantInt si_llvm_i32_ty, 2))),
              local_ptr_ty);
}

class DSAddrSrc2A<dag addr> {
  dag ret = (LLVMIntToPtr addr, local_ptr_ty);
}

//------------------------------------------------------------------------------
// SRC2 integer binary: DS[A] = OP(DS[A], DS[B]) (RSUB swaps the operands).
//------------------------------------------------------------------------------

class DSSrc2BinOp<LLVMOp op, SILLVMType ty, bit reverse, InstSI inst> {
  LLVMOp Op = op;
  SILLVMType Ty = ty;
  bit Reverse = reverse;
  InstSI Inst = inst;
}

foreach desc = [
  DSSrc2BinOp<LLVMAdd, si_llvm_i32_ty, 0, DS_ADD_SRC2_U32>,
  DSSrc2BinOp<LLVMSub, si_llvm_i32_ty, 0, DS_SUB_SRC2_U32>,
  DSSrc2BinOp<LLVMSub, si_llvm_i32_ty, 1, DS_RSUB_SRC2_U32>,
  DSSrc2BinOp<LLVMAnd, si_llvm_i32_ty, 0, DS_AND_SRC2_B32>,
  DSSrc2BinOp<LLVMOr,  si_llvm_i32_ty, 0, DS_OR_SRC2_B32>,
  DSSrc2BinOp<LLVMXor, si_llvm_i32_ty, 0, DS_XOR_SRC2_B32>,
  DSSrc2BinOp<LLVMAdd, si_llvm_i64_ty, 0, DS_ADD_SRC2_U64>,
  DSSrc2BinOp<LLVMSub, si_llvm_i64_ty, 0, DS_SUB_SRC2_U64>,
  DSSrc2BinOp<LLVMSub, si_llvm_i64_ty, 1, DS_RSUB_SRC2_U64>,
  DSSrc2BinOp<LLVMAnd, si_llvm_i64_ty, 0, DS_AND_SRC2_B64>,
  DSSrc2BinOp<LLVMOr,  si_llvm_i64_ty, 0, DS_OR_SRC2_B64>,
  DSSrc2BinOp<LLVMXor, si_llvm_i64_ty, 0, DS_XOR_SRC2_B64>
] in
  def : InstSISemantic<desc.Inst, [
    (DefVal $a, (LLVMLoad desc.Ty,
      DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)),
    (DefVal $b, (LLVMLoad desc.Ty,
      DSAddrSrc2B<(GetNamedOperand $addr, si_llvm_i32_ty),
                  (GetNamedOperand $offset, si_llvm_i32_ty)>.ret)),
    (LLVMStore !if(desc.Reverse,
                   (desc.Op (GetVal $b), (GetVal $a)),
                   (desc.Op (GetVal $a), (GetVal $b))),
               DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)
  ]>;

//------------------------------------------------------------------------------
// SRC2 integer MIN/MAX: DS[A] = select(cmp(DS[A], DS[B]), DS[A], DS[B]).
//------------------------------------------------------------------------------

class DSSrc2MinMaxIntDesc<LLVMCmpPredicate pred, SILLVMType ty, InstSI inst> {
  LLVMCmpPredicate Pred = pred;
  SILLVMType Ty = ty;
  InstSI Inst = inst;
}

foreach desc = [
  DSSrc2MinMaxIntDesc<LLVMICmpSLT, si_llvm_i32_ty, DS_MIN_SRC2_I32>,
  DSSrc2MinMaxIntDesc<LLVMICmpSGT, si_llvm_i32_ty, DS_MAX_SRC2_I32>,
  DSSrc2MinMaxIntDesc<LLVMICmpULT, si_llvm_i32_ty, DS_MIN_SRC2_U32>,
  DSSrc2MinMaxIntDesc<LLVMICmpUGT, si_llvm_i32_ty, DS_MAX_SRC2_U32>,
  DSSrc2MinMaxIntDesc<LLVMICmpSLT, si_llvm_i64_ty, DS_MIN_SRC2_I64>,
  DSSrc2MinMaxIntDesc<LLVMICmpSGT, si_llvm_i64_ty, DS_MAX_SRC2_I64>,
  DSSrc2MinMaxIntDesc<LLVMICmpULT, si_llvm_i64_ty, DS_MIN_SRC2_U64>,
  DSSrc2MinMaxIntDesc<LLVMICmpUGT, si_llvm_i64_ty, DS_MAX_SRC2_U64>
] in
  def : InstSISemantic<desc.Inst, [
    (DefVal $a, (LLVMLoad desc.Ty,
      DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)),
    (DefVal $b, (LLVMLoad desc.Ty,
      DSAddrSrc2B<(GetNamedOperand $addr, si_llvm_i32_ty),
                  (GetNamedOperand $offset, si_llvm_i32_ty)>.ret)),
    (LLVMStore (LLVMSelect (LLVMCmp desc.Pred, (GetVal $a), (GetVal $b)),
                            (GetVal $a), (GetVal $b)),
               DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)
  ]>;

//------------------------------------------------------------------------------
// SRC2 FP MIN/MAX (F32/F64): bitcast to FP, minnum/maxnum, bitcast back.
//------------------------------------------------------------------------------

class DSSrc2FpMinMaxDesc<Intrinsic intr, SILLVMType iTy, SILLVMType fpTy,
                          InstSI inst> {
  Intrinsic Intr = intr;
  SILLVMType ITy = iTy;
  SILLVMType FpTy = fpTy;
  InstSI Inst = inst;
}

foreach desc = [
  DSSrc2FpMinMaxDesc<int_minnum, si_llvm_i32_ty, si_llvm_float_ty,  DS_MIN_SRC2_F32>,
  DSSrc2FpMinMaxDesc<int_maxnum, si_llvm_i32_ty, si_llvm_float_ty,  DS_MAX_SRC2_F32>,
  DSSrc2FpMinMaxDesc<int_minnum, si_llvm_i64_ty, si_llvm_double_ty, DS_MIN_SRC2_F64>,
  DSSrc2FpMinMaxDesc<int_maxnum, si_llvm_i64_ty, si_llvm_double_ty, DS_MAX_SRC2_F64>
] in
  def : InstSISemantic<desc.Inst, [
    (DefVal $a, (LLVMLoad desc.ITy,
      DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)),
    (DefVal $b, (LLVMLoad desc.ITy,
      DSAddrSrc2B<(GetNamedOperand $addr, si_llvm_i32_ty),
                  (GetNamedOperand $offset, si_llvm_i32_ty)>.ret)),
    (LLVMStore (LLVMBitCast
                 (LLVMBinaryIntrinsic desc.Intr,
                   (LLVMBitCast (GetVal $a), desc.FpTy),
                   (LLVMBitCast (GetVal $b), desc.FpTy)),
                 desc.ITy),
               DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)
  ]>;

// DS_ADD_SRC2_F32 — only F32 (no F64 variant) per the GCN3 ISA tables.
def DS_ADD_SRC2_F32_sem : InstSISemantic<DS_ADD_SRC2_F32, [
  (DefVal $a, (LLVMLoad si_llvm_i32_ty,
    DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)),
  (DefVal $b, (LLVMLoad si_llvm_i32_ty,
    DSAddrSrc2B<(GetNamedOperand $addr, si_llvm_i32_ty),
                (GetNamedOperand $offset, si_llvm_i32_ty)>.ret)),
  (LLVMStore (LLVMBitCast
               (LLVMFAdd (LLVMBitCast (GetVal $a), si_llvm_float_ty),
                         (LLVMBitCast (GetVal $b), si_llvm_float_ty)),
               si_llvm_i32_ty),
             DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)
]>;

//------------------------------------------------------------------------------
// SRC2 INC / DEC:
//   INC: DS[A] = (DS[A] >= DS[B]) ? 0 : DS[A] + 1
//   DEC: DS[A] = (DS[A] == 0 || DS[A] > DS[B]) ? DS[B] : DS[A] - 1
//------------------------------------------------------------------------------

class DSSrc2IncDecDesc<bit isInc, SILLVMType ty, InstSI inst> {
  bit IsInc = isInc;
  SILLVMType Ty = ty;
  InstSI Inst = inst;
}

foreach desc = [
  DSSrc2IncDecDesc<1, si_llvm_i32_ty, DS_INC_SRC2_U32>,
  DSSrc2IncDecDesc<0, si_llvm_i32_ty, DS_DEC_SRC2_U32>,
  DSSrc2IncDecDesc<1, si_llvm_i64_ty, DS_INC_SRC2_U64>,
  DSSrc2IncDecDesc<0, si_llvm_i64_ty, DS_DEC_SRC2_U64>
] in
  def : InstSISemantic<desc.Inst, [
    (DefVal $a, (LLVMLoad desc.Ty,
      DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)),
    (DefVal $b, (LLVMLoad desc.Ty,
      DSAddrSrc2B<(GetNamedOperand $addr, si_llvm_i32_ty),
                  (GetNamedOperand $offset, si_llvm_i32_ty)>.ret)),
    (LLVMStore
      !if(desc.IsInc,
          (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $a), (GetVal $b)),
                       (ConstantInt desc.Ty, 0),
                       (LLVMAdd (GetVal $a), (ConstantInt desc.Ty, 1))),
          (LLVMSelect (LLVMOr (LLVMCmp LLVMICmpEQ, (GetVal $a), (ConstantInt desc.Ty, 0)),
                               (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b))),
                       (GetVal $b),
                       (LLVMSub (GetVal $a), (ConstantInt desc.Ty, 1)))),
      DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)
  ]>;

//------------------------------------------------------------------------------
// SRC2 WRITE: DS[A] = DS[B]  (no read of A).
//------------------------------------------------------------------------------

class DSSrc2WriteDesc<SILLVMType ty, InstSI inst> {
  SILLVMType Ty = ty;
  InstSI Inst = inst;
}

foreach desc = [
  DSSrc2WriteDesc<si_llvm_i32_ty, DS_WRITE_SRC2_B32>,
  DSSrc2WriteDesc<si_llvm_i64_ty, DS_WRITE_SRC2_B64>
] in
  def : InstSISemantic<desc.Inst, [
    (DefVal $b, (LLVMLoad desc.Ty,
      DSAddrSrc2B<(GetNamedOperand $addr, si_llvm_i32_ty),
                  (GetNamedOperand $offset, si_llvm_i32_ty)>.ret)),
    (LLVMStore (GetVal $b),
               DSAddrSrc2A<(GetNamedOperand $addr, si_llvm_i32_ty)>.ret)
  ]>;

//==============================================================================
// Unmodeled DS instructions — intentionally no InstSISemantic record
//==============================================================================
//
// DS_ORDERED_COUNT — one MIR pseudo maps back to either
// `int_amdgcn_ds_ordered_add` or `int_amdgcn_ds_ordered_swap` based on
// OFFSET1[5:4]. The intrinsic signature also expects M0 packed as a pointer
// + several ImmArgs that aren't recoverable from MIR alone. Falls through to
// MIInlineAsmEmitter for an opaque `asm sideeffect`.
