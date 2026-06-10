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
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), flat_ptr_ty),
                         (GetNamedOperand $offset));
}

// FLAT SADDR-mode address (gfx11+): byteAddr = saddr64 + zext_i64(voffset32) + immOffset.
// $saddr is a uniform 64-bit base, $vaddr is a 32-bit per-lane offset.
class CalculateFlatSAddr<dag saddr, dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd
    (LLVMIntToPtr (LLVMAdd saddr,
                            (LLVMZExt vaddr, si_llvm_i64_ty)),
                   flat_ptr_ty),
    offset);
}

class CalculateGlobalAddr<dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), global_ptr_ty),
                         (GetNamedOperand $offset));
}

class CalculateGlobalSAddr<dag saddr, dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (LLVMAdd (GetNamedOperand $saddr),
                                                 (LLVMZExt (GetNamedOperand $vaddr), si_llvm_i64_ty)),
                          global_ptr_ty),
                         (GetNamedOperand $offset));
}

class CalculateScratchAddr<dag vaddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr (GetNamedOperand $vaddr), scratch_ptr_ty),
                         (GetNamedOperand $offset));
}

//==============================================================================
// FLAT Instructions - Loads (base + SADDR + D16 / D16_HI + t16 + MONITOR)
//
// Sources: GCN3 pp.283-284; Vega pp.218-222; Vega7nm pp.220-224;
//          CDNA1 pp.207-209; CDNA2 pp.200-203; CDNA3 pp.205-210;
//          CDNA4 pp.210-215; RDNA1/2 pp.185-190; RDNA3 pp.280-285;
//          RDNA4 pp.290-295.
//
// Conventions:
//   - Output operand is $vdst.
//   - Address: vaddr + offset (base) or saddr + zext(vaddr) + offset (SADDR).
//   - Sub-DWORD (U/SBYTE, U/SSHORT) sign- or zero-extend to i32 before write.
//   - DWORD/DWORDX2/X3/X4 load into i32 / v2i32 / v3i32 / v4i32 directly.
//   - D16 forms read 8 or 16 bits, merge into the LOW (or HIGH for _D16_HI)
//     half of $vdst while preserving the OTHER half from $vdst_in (tied input).
//   - D16_t16 forms (RDNA4 true-16): destination is a 16-bit VGPR ($vdst is
//     i16). No half-preservation since the register is only 16 bits wide.
//   - MONITOR_B{32,64,128} (gfx12): a load with HW-level cache-line monitor
//     hint. Modeled via int_amdgcn_flat_load_monitor_b{32,64,128}.
//==============================================================================

//------------------------------------------------------------------------------
// A. Standard FLAT loads — base + _SADDR.
//------------------------------------------------------------------------------

class FLATLoadDesc<InstSI base, InstSI sad, SILLVMType memTy,
                   bit isSigned, bit needsExt> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
  bit NeedsExt = needsExt;  // true for U/SBYTE and U/SSHORT (ext to i32)
}

foreach desc = [
  FLATLoadDesc<FLAT_LOAD_UBYTE,    FLAT_LOAD_UBYTE_SADDR,    si_llvm_i8_ty,  0, 1>,
  FLATLoadDesc<FLAT_LOAD_SBYTE,    FLAT_LOAD_SBYTE_SADDR,    si_llvm_i8_ty,  1, 1>,
  FLATLoadDesc<FLAT_LOAD_USHORT,   FLAT_LOAD_USHORT_SADDR,   si_llvm_i16_ty, 0, 1>,
  FLATLoadDesc<FLAT_LOAD_SSHORT,   FLAT_LOAD_SSHORT_SADDR,   si_llvm_i16_ty, 1, 1>,
  FLATLoadDesc<FLAT_LOAD_DWORD,    FLAT_LOAD_DWORD_SADDR,    si_llvm_i32_ty,  0, 0>,
  FLATLoadDesc<FLAT_LOAD_DWORDX2,  FLAT_LOAD_DWORDX2_SADDR,  si_llvm_v2i32_ty, 0, 0>,
  FLATLoadDesc<FLAT_LOAD_DWORDX3,  FLAT_LOAD_DWORDX3_SADDR,  si_llvm_v3i32_ty, 0, 0>,
  FLATLoadDesc<FLAT_LOAD_DWORDX4,  FLAT_LOAD_DWORDX4_SADDR,  si_llvm_v4i32_ty, 0, 0>
] in {
  // Base (no SADDR).
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (DefVal $val, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(desc.NeedsExt,
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $val), si_llvm_i32_ty),
               (LLVMZExt (GetVal $val), si_llvm_i32_ty)),
           (GetVal $val)))
  ]>;
  // _SADDR form.
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (DefVal $val, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(desc.NeedsExt,
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $val), si_llvm_i32_ty),
               (LLVMZExt (GetVal $val), si_llvm_i32_ty)),
           (GetVal $val)))
  ]>;
}

//------------------------------------------------------------------------------
// B. D16 / D16_HI FLAT loads — load 8 or 16 bits, merge with tied $vdst_in.
//   Low form : vdst = (vdst_in & 0xFFFF0000) | zext_i32(ext_i16(load_memTy))
//   High form: vdst = (vdst_in & 0x0000FFFF) | (zext_i32(ext_i16(load_memTy)) << 16)
// For SHORT (memTy = i16) the inner ext_i16 is the identity.
// For (U|S)BYTE  (memTy = i8) we extend the byte to i16 first; sign- vs
// zero-extension depends on the opcode (U vs S).
//------------------------------------------------------------------------------

class FLATD16LoadDesc<InstSI base, InstSI sad, InstSI hi, InstSI hiSad,
                      SILLVMType memTy, bit isSigned> {
  InstSI Base  = base;
  InstSI Sad   = sad;
  InstSI Hi    = hi;
  InstSI HiSad = hiSad;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
}

foreach desc = [
  FLATD16LoadDesc<FLAT_LOAD_UBYTE_D16, FLAT_LOAD_UBYTE_D16_SADDR,
                  FLAT_LOAD_UBYTE_D16_HI, FLAT_LOAD_UBYTE_D16_HI_SADDR,
                  si_llvm_i8_ty,  0>,
  FLATD16LoadDesc<FLAT_LOAD_SBYTE_D16, FLAT_LOAD_SBYTE_D16_SADDR,
                  FLAT_LOAD_SBYTE_D16_HI, FLAT_LOAD_SBYTE_D16_HI_SADDR,
                  si_llvm_i8_ty,  1>,
  FLATD16LoadDesc<FLAT_LOAD_SHORT_D16, FLAT_LOAD_SHORT_D16_SADDR,
                  FLAT_LOAD_SHORT_D16_HI, FLAT_LOAD_SHORT_D16_HI_SADDR,
                  si_llvm_i16_ty, 0>
] in {
  // D16 forms have a tied-input $vdst_in (same physical register as $vdst);
  // the loaded half is merged into one half of $vdst while the other half is
  // preserved from $vdst_in.
  //   Low form : vdst = (vdst_in & 0xFFFF0000) | zext_i32(ext_i16(load))
  //   High form: vdst = (vdst_in & 0x0000FFFF) | (zext_i32(ext_i16(load)) << 16)

  // Low half — base.
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $upper, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $upper), (GetVal $half32)))
  ]>;
  // Low half — SADDR.
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $upper, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $upper), (GetVal $half32)))
  ]>;
  // High half — base.
  def : InstSISemantic<desc.Hi, [
    (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $shifted, (LLVMShl (GetVal $half32),
                                (ConstantInt si_llvm_i32_ty, 16))),
    (DefVal $lower, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0x0000FFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $lower), (GetVal $shifted)))
  ]>;
  // High half — SADDR.
  def : InstSISemantic<desc.HiSad, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $shifted, (LLVMShl (GetVal $half32),
                                (ConstantInt si_llvm_i32_ty, 16))),
    (DefVal $lower, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0x0000FFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $lower), (GetVal $shifted)))
  ]>;
}

//------------------------------------------------------------------------------
// C. D16_t16 FLAT loads (RDNA4 true-16) — destination is a 16-bit VGPR.
//   vdst = ext_i16(load_memTy)   (no merge with vdst_in; register is 16 bits)
//------------------------------------------------------------------------------

class FLATT16LoadDesc<InstSI base, InstSI sad, SILLVMType memTy, bit isSigned> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
}

foreach desc = [
  FLATT16LoadDesc<FLAT_LOAD_UBYTE_D16_t16, FLAT_LOAD_UBYTE_D16_t16_SADDR,
                  si_llvm_i8_ty,  0>,
  FLATT16LoadDesc<FLAT_LOAD_SBYTE_D16_t16, FLAT_LOAD_SBYTE_D16_t16_SADDR,
                  si_llvm_i8_ty,  1>,
  FLATT16LoadDesc<FLAT_LOAD_SHORT_D16_t16, FLAT_LOAD_SHORT_D16_t16_SADDR,
                  si_llvm_i16_ty, 0>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(!eq(desc.MemTy.ElemSize, 16),
           (GetVal $raw),
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
               (LLVMZExt (GetVal $raw), si_llvm_i16_ty))))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(!eq(desc.MemTy.ElemSize, 16),
           (GetVal $raw),
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
               (LLVMZExt (GetVal $raw), si_llvm_i16_ty))))
  ]>;
}

//------------------------------------------------------------------------------
// D. MONITOR FLAT loads (gfx12) — load + HW cache-line monitor hint.
//   Modeled via int_amdgcn_flat_load_monitor_b{32,64,128}.
//------------------------------------------------------------------------------

class FLATMonitorDesc<InstSI base, InstSI sad, SILLVMType retTy, Intrinsic intr> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType RetTy = retTy;
  Intrinsic Intr  = intr;
}

foreach desc = [
  FLATMonitorDesc<FLAT_LOAD_MONITOR_B32,  FLAT_LOAD_MONITOR_B32_SADDR,
                   si_llvm_i32_ty,   int_amdgcn_flat_load_monitor_b32>,
  FLATMonitorDesc<FLAT_LOAD_MONITOR_B64,  FLAT_LOAD_MONITOR_B64_SADDR,
                   si_llvm_i64_ty,   int_amdgcn_flat_load_monitor_b64>,
  FLATMonitorDesc<FLAT_LOAD_MONITOR_B128, FLAT_LOAD_MONITOR_B128_SADDR,
                   si_llvm_v4i32_ty, int_amdgcn_flat_load_monitor_b128>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr,
                       [(GetVal $addr),
                        (GetNamedOperand $cpol, si_llvm_i32_ty)]))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr,
                       [(GetVal $addr),
                        (GetNamedOperand $cpol, si_llvm_i32_ty)]))
  ]>;
}

//==============================================================================
// FLAT Instructions - Stores (base + SADDR + D16_HI)
//
// Sources: GCN3 pp.283-284; Vega pp.218-222; CDNA1-4 / RDNA1-4 (see citations
// in the Loads section above; per-arch availability matches loads).
//
// Conventions:
//   - Input data operand is $vdata.
//   - Sub-DWORD stores truncate $vdata to i8 / i16 then store.
//   - DWORD / DWORDX2/3/4 store $vdata directly.
//   - D16_HI stores extract the UPPER 16 bits of $vdata via lshr 16 + trunc,
//     then store at the requested sub-DWORD width. Used for packed-16
//     scatter idioms.
//   - SADDR forms compute address as saddr + zext(vaddr) + offset; the rest
//     is identical to the non-SADDR form.
//==============================================================================

class FLATStoreDesc<InstSI base, InstSI sad,
                    SILLVMType storeTy, bit needsTrunc, bit isD16Hi> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType StoreTy = storeTy;  // i8 / i16 / i32 / v2i32 / v3i32 / v4i32
  bit NeedsTrunc = needsTrunc;    // true for sub-DWORD (i8 / i16) stores
  bit IsD16Hi    = isD16Hi;       // upper-half extraction before truncate
}

foreach desc = [
  FLATStoreDesc<FLAT_STORE_BYTE,         FLAT_STORE_BYTE_SADDR,
                 si_llvm_i8_ty,   1, 0>,
  FLATStoreDesc<FLAT_STORE_BYTE_D16_HI,  FLAT_STORE_BYTE_D16_HI_SADDR,
                 si_llvm_i8_ty,   1, 1>,
  FLATStoreDesc<FLAT_STORE_SHORT,        FLAT_STORE_SHORT_SADDR,
                 si_llvm_i16_ty,  1, 0>,
  FLATStoreDesc<FLAT_STORE_SHORT_D16_HI, FLAT_STORE_SHORT_D16_HI_SADDR,
                 si_llvm_i16_ty,  1, 1>,
  FLATStoreDesc<FLAT_STORE_DWORD,        FLAT_STORE_DWORD_SADDR,
                 si_llvm_i32_ty,  0, 0>,
  FLATStoreDesc<FLAT_STORE_DWORDX2,      FLAT_STORE_DWORDX2_SADDR,
                 si_llvm_v2i32_ty, 0, 0>,
  FLATStoreDesc<FLAT_STORE_DWORDX3,      FLAT_STORE_DWORDX3_SADDR,
                 si_llvm_v3i32_ty, 0, 0>,
  FLATStoreDesc<FLAT_STORE_DWORDX4,      FLAT_STORE_DWORDX4_SADDR,
                 si_llvm_v4i32_ty, 0, 0>
] in {
  // Base (no SADDR).
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.IsD16Hi,
                   (LLVMTrunc (LLVMLShr (GetNamedOperand $vdata, si_llvm_i32_ty),
                                         (ConstantInt si_llvm_i32_ty, 16)),
                              desc.StoreTy),
                   !if(desc.NeedsTrunc,
                       (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i32_ty),
                                   desc.StoreTy),
                       (GetNamedOperand $vdata))),
                (GetVal $addr))
  ]>;
  // SADDR.
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.IsD16Hi,
                   (LLVMTrunc (LLVMLShr (GetNamedOperand $vdata, si_llvm_i32_ty),
                                         (ConstantInt si_llvm_i32_ty, 16)),
                              desc.StoreTy),
                   !if(desc.NeedsTrunc,
                       (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i32_ty),
                                   desc.StoreTy),
                       (GetNamedOperand $vdata))),
                (GetVal $addr))
  ]>;
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

//------------------------------------------------------------------------------
// Binary RMW atomics — uniform `atomicrmw <op>, addr, vdata` pattern.
//   NoRet form  → side-effect only.
//   RTN forms (incl _agpr) → write old loaded value to $vdst.
//   SADDR forms  → same body, SADDR address-calc helper.
// Covers SWAP/ADD/SUB/AND/OR/XOR/SMIN/UMIN/SMAX/UMAX/INC/DEC × {i32, i64}.
// INC/DEC use UIncWrap/UDecWrap which model the AMDGPU "increment wrapped"
// and "decrement wrapped" semantics directly.
//------------------------------------------------------------------------------

class FLATBinAtomicDesc<AtomicRMWOp op,
                         InstSI noret, InstSI rtn, InstSI rtnAgpr,
                         InstSI sad, InstSI sadRtn> {
  AtomicRMWOp Op = op;
  InstSI NoRet   = noret;
  InstSI Rtn     = rtn;
  InstSI RtnAgpr = rtnAgpr;
  InstSI Sad     = sad;
  InstSI SadRtn  = sadRtn;
}

foreach desc = [
  // 32-bit
  FLATBinAtomicDesc<AtomicRMWXchg,
    FLAT_ATOMIC_SWAP, FLAT_ATOMIC_SWAP_RTN, FLAT_ATOMIC_SWAP_RTN_agpr,
    FLAT_ATOMIC_SWAP_SADDR, FLAT_ATOMIC_SWAP_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWAdd,
    FLAT_ATOMIC_ADD, FLAT_ATOMIC_ADD_RTN, FLAT_ATOMIC_ADD_RTN_agpr,
    FLAT_ATOMIC_ADD_SADDR, FLAT_ATOMIC_ADD_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWSub,
    FLAT_ATOMIC_SUB, FLAT_ATOMIC_SUB_RTN, FLAT_ATOMIC_SUB_RTN_agpr,
    FLAT_ATOMIC_SUB_SADDR, FLAT_ATOMIC_SUB_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWAnd,
    FLAT_ATOMIC_AND, FLAT_ATOMIC_AND_RTN, FLAT_ATOMIC_AND_RTN_agpr,
    FLAT_ATOMIC_AND_SADDR, FLAT_ATOMIC_AND_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWOr,
    FLAT_ATOMIC_OR,  FLAT_ATOMIC_OR_RTN,  FLAT_ATOMIC_OR_RTN_agpr,
    FLAT_ATOMIC_OR_SADDR,  FLAT_ATOMIC_OR_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWXor,
    FLAT_ATOMIC_XOR, FLAT_ATOMIC_XOR_RTN, FLAT_ATOMIC_XOR_RTN_agpr,
    FLAT_ATOMIC_XOR_SADDR, FLAT_ATOMIC_XOR_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWMin,
    FLAT_ATOMIC_SMIN, FLAT_ATOMIC_SMIN_RTN, FLAT_ATOMIC_SMIN_RTN_agpr,
    FLAT_ATOMIC_SMIN_SADDR, FLAT_ATOMIC_SMIN_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUMin,
    FLAT_ATOMIC_UMIN, FLAT_ATOMIC_UMIN_RTN, FLAT_ATOMIC_UMIN_RTN_agpr,
    FLAT_ATOMIC_UMIN_SADDR, FLAT_ATOMIC_UMIN_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWMax,
    FLAT_ATOMIC_SMAX, FLAT_ATOMIC_SMAX_RTN, FLAT_ATOMIC_SMAX_RTN_agpr,
    FLAT_ATOMIC_SMAX_SADDR, FLAT_ATOMIC_SMAX_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUMax,
    FLAT_ATOMIC_UMAX, FLAT_ATOMIC_UMAX_RTN, FLAT_ATOMIC_UMAX_RTN_agpr,
    FLAT_ATOMIC_UMAX_SADDR, FLAT_ATOMIC_UMAX_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUIncWrap,
    FLAT_ATOMIC_INC, FLAT_ATOMIC_INC_RTN, FLAT_ATOMIC_INC_RTN_agpr,
    FLAT_ATOMIC_INC_SADDR, FLAT_ATOMIC_INC_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUDecWrap,
    FLAT_ATOMIC_DEC, FLAT_ATOMIC_DEC_RTN, FLAT_ATOMIC_DEC_RTN_agpr,
    FLAT_ATOMIC_DEC_SADDR, FLAT_ATOMIC_DEC_SADDR_RTN>,
  // 64-bit (X2)
  FLATBinAtomicDesc<AtomicRMWXchg,
    FLAT_ATOMIC_SWAP_X2, FLAT_ATOMIC_SWAP_X2_RTN, FLAT_ATOMIC_SWAP_X2_RTN_agpr,
    FLAT_ATOMIC_SWAP_X2_SADDR, FLAT_ATOMIC_SWAP_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWAdd,
    FLAT_ATOMIC_ADD_X2, FLAT_ATOMIC_ADD_X2_RTN, FLAT_ATOMIC_ADD_X2_RTN_agpr,
    FLAT_ATOMIC_ADD_X2_SADDR, FLAT_ATOMIC_ADD_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWSub,
    FLAT_ATOMIC_SUB_X2, FLAT_ATOMIC_SUB_X2_RTN, FLAT_ATOMIC_SUB_X2_RTN_agpr,
    FLAT_ATOMIC_SUB_X2_SADDR, FLAT_ATOMIC_SUB_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWAnd,
    FLAT_ATOMIC_AND_X2, FLAT_ATOMIC_AND_X2_RTN, FLAT_ATOMIC_AND_X2_RTN_agpr,
    FLAT_ATOMIC_AND_X2_SADDR, FLAT_ATOMIC_AND_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWOr,
    FLAT_ATOMIC_OR_X2,  FLAT_ATOMIC_OR_X2_RTN,  FLAT_ATOMIC_OR_X2_RTN_agpr,
    FLAT_ATOMIC_OR_X2_SADDR,  FLAT_ATOMIC_OR_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWXor,
    FLAT_ATOMIC_XOR_X2, FLAT_ATOMIC_XOR_X2_RTN, FLAT_ATOMIC_XOR_X2_RTN_agpr,
    FLAT_ATOMIC_XOR_X2_SADDR, FLAT_ATOMIC_XOR_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWMin,
    FLAT_ATOMIC_SMIN_X2, FLAT_ATOMIC_SMIN_X2_RTN, FLAT_ATOMIC_SMIN_X2_RTN_agpr,
    FLAT_ATOMIC_SMIN_X2_SADDR, FLAT_ATOMIC_SMIN_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUMin,
    FLAT_ATOMIC_UMIN_X2, FLAT_ATOMIC_UMIN_X2_RTN, FLAT_ATOMIC_UMIN_X2_RTN_agpr,
    FLAT_ATOMIC_UMIN_X2_SADDR, FLAT_ATOMIC_UMIN_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWMax,
    FLAT_ATOMIC_SMAX_X2, FLAT_ATOMIC_SMAX_X2_RTN, FLAT_ATOMIC_SMAX_X2_RTN_agpr,
    FLAT_ATOMIC_SMAX_X2_SADDR, FLAT_ATOMIC_SMAX_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUMax,
    FLAT_ATOMIC_UMAX_X2, FLAT_ATOMIC_UMAX_X2_RTN, FLAT_ATOMIC_UMAX_X2_RTN_agpr,
    FLAT_ATOMIC_UMAX_X2_SADDR, FLAT_ATOMIC_UMAX_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUIncWrap,
    FLAT_ATOMIC_INC_X2, FLAT_ATOMIC_INC_X2_RTN, FLAT_ATOMIC_INC_X2_RTN_agpr,
    FLAT_ATOMIC_INC_X2_SADDR, FLAT_ATOMIC_INC_X2_SADDR_RTN>,
  FLATBinAtomicDesc<AtomicRMWUDecWrap,
    FLAT_ATOMIC_DEC_X2, FLAT_ATOMIC_DEC_X2_RTN, FLAT_ATOMIC_DEC_X2_RTN_agpr,
    FLAT_ATOMIC_DEC_X2_SADDR, FLAT_ATOMIC_DEC_X2_SADDR_RTN>
] in {
  // NoRet (base): side-effecting atomicrmw, no SSA result store.
  def : InstSISemantic<desc.NoRet, [
    (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (LLVMAtomicRMW desc.Op, (GetVal $addr), (GetNamedOperand $vdata),
                    AlignNone, (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty)))
  ]>;
  // NoRet (SADDR).
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (LLVMAtomicRMW desc.Op, (GetVal $addr), (GetNamedOperand $vdata),
                    AlignNone, (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty)))
  ]>;
  // RTN (base + _agpr): write old loaded value to $vdst.
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, [
      (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (LLVMAtomicRMW desc.Op, (GetVal $addr),
                                              (GetNamedOperand $vdata),
                                              AlignNone,
                                              (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty))))
    ]>;
  // RTN (SADDR).
  def : InstSISemantic<desc.SadRtn, [
    (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (LLVMAtomicRMW desc.Op, (GetVal $addr),
                                            (GetNamedOperand $vdata),
                                            AlignNone, (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty))))
  ]>;
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

//==============================================================================
// GLOBAL Instructions - Loads (base + SADDR + D16/D16_HI)
//
// Sources: Vega+ (see GLOBAL atomics citations; loads share the same
// chapters in each per-arch PDF — page ranges given there).
//
// Same patterns as FLAT loads but using CalculateGlobalAddr / GlobalSAddr.
// addrspace(1) (global) ptr type.
//==============================================================================

class GLOBALLoadDesc<InstSI base, InstSI sad, SILLVMType memTy,
                     bit isSigned, bit needsExt> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
  bit NeedsExt = needsExt;
}

foreach desc = [
  GLOBALLoadDesc<GLOBAL_LOAD_UBYTE,    GLOBAL_LOAD_UBYTE_SADDR,    si_llvm_i8_ty,  0, 1>,
  GLOBALLoadDesc<GLOBAL_LOAD_SBYTE,    GLOBAL_LOAD_SBYTE_SADDR,    si_llvm_i8_ty,  1, 1>,
  GLOBALLoadDesc<GLOBAL_LOAD_USHORT,   GLOBAL_LOAD_USHORT_SADDR,   si_llvm_i16_ty, 0, 1>,
  GLOBALLoadDesc<GLOBAL_LOAD_SSHORT,   GLOBAL_LOAD_SSHORT_SADDR,   si_llvm_i16_ty, 1, 1>,
  GLOBALLoadDesc<GLOBAL_LOAD_DWORD,    GLOBAL_LOAD_DWORD_SADDR,    si_llvm_i32_ty,   0, 0>,
  GLOBALLoadDesc<GLOBAL_LOAD_DWORDX2,  GLOBAL_LOAD_DWORDX2_SADDR,  si_llvm_v2i32_ty, 0, 0>,
  GLOBALLoadDesc<GLOBAL_LOAD_DWORDX3,  GLOBAL_LOAD_DWORDX3_SADDR,  si_llvm_v3i32_ty, 0, 0>,
  GLOBALLoadDesc<GLOBAL_LOAD_DWORDX4,  GLOBAL_LOAD_DWORDX4_SADDR,  si_llvm_v4i32_ty, 0, 0>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (DefVal $val, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(desc.NeedsExt,
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $val), si_llvm_i32_ty),
               (LLVMZExt (GetVal $val), si_llvm_i32_ty)),
           (GetVal $val)))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (DefVal $val, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(desc.NeedsExt,
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $val), si_llvm_i32_ty),
               (LLVMZExt (GetVal $val), si_llvm_i32_ty)),
           (GetVal $val)))
  ]>;
}

// D16 / D16_HI GLOBAL loads — same pattern as FLAT. See FLAT D16 comment for
// the half-preserve workaround (read $vdst as tied source).
class GLOBALD16LoadDesc<InstSI base, InstSI sad, InstSI hi, InstSI hiSad,
                        SILLVMType memTy, bit isSigned> {
  InstSI Base  = base;
  InstSI Sad   = sad;
  InstSI Hi    = hi;
  InstSI HiSad = hiSad;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
}

foreach desc = [
  GLOBALD16LoadDesc<GLOBAL_LOAD_UBYTE_D16, GLOBAL_LOAD_UBYTE_D16_SADDR,
                    GLOBAL_LOAD_UBYTE_D16_HI, GLOBAL_LOAD_UBYTE_D16_HI_SADDR,
                    si_llvm_i8_ty,  0>,
  GLOBALD16LoadDesc<GLOBAL_LOAD_SBYTE_D16, GLOBAL_LOAD_SBYTE_D16_SADDR,
                    GLOBAL_LOAD_SBYTE_D16_HI, GLOBAL_LOAD_SBYTE_D16_HI_SADDR,
                    si_llvm_i8_ty,  1>,
  GLOBALD16LoadDesc<GLOBAL_LOAD_SHORT_D16, GLOBAL_LOAD_SHORT_D16_SADDR,
                    GLOBAL_LOAD_SHORT_D16_HI, GLOBAL_LOAD_SHORT_D16_HI_SADDR,
                    si_llvm_i16_ty, 0>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $upper, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $upper), (GetVal $half32)))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $upper, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0xFFFF0000))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $upper), (GetVal $half32)))
  ]>;
  def : InstSISemantic<desc.Hi, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $shifted, (LLVMShl (GetVal $half32),
                                (ConstantInt si_llvm_i32_ty, 16))),
    (DefVal $lower, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0x0000FFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $lower), (GetVal $shifted)))
  ]>;
  def : InstSISemantic<desc.HiSad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (DefVal $half, !if(!eq(desc.MemTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(desc.IsSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    (DefVal $shifted, (LLVMShl (GetVal $half32),
                                (ConstantInt si_llvm_i32_ty, 16))),
    (DefVal $lower, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty, 0x0000FFFF))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $lower), (GetVal $shifted)))
  ]>;
}

//==============================================================================
// GLOBAL Instructions - D16 Loads
// Sources: Same as GLOBAL Loads above.

//==============================================================================

//==============================================================================
// GLOBAL Instructions - Stores
// Sources: Same as GLOBAL Loads above.

//==============================================================================

class GLOBALStoreDesc<InstSI base, InstSI sad,
                      SILLVMType storeTy, bit needsTrunc, bit isD16Hi> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType StoreTy = storeTy;
  bit NeedsTrunc = needsTrunc;
  bit IsD16Hi    = isD16Hi;
}

foreach desc = [
  GLOBALStoreDesc<GLOBAL_STORE_BYTE,         GLOBAL_STORE_BYTE_SADDR,
                   si_llvm_i8_ty,   1, 0>,
  GLOBALStoreDesc<GLOBAL_STORE_BYTE_D16_HI,  GLOBAL_STORE_BYTE_D16_HI_SADDR,
                   si_llvm_i8_ty,   1, 1>,
  GLOBALStoreDesc<GLOBAL_STORE_SHORT,        GLOBAL_STORE_SHORT_SADDR,
                   si_llvm_i16_ty,  1, 0>,
  GLOBALStoreDesc<GLOBAL_STORE_SHORT_D16_HI, GLOBAL_STORE_SHORT_D16_HI_SADDR,
                   si_llvm_i16_ty,  1, 1>,
  GLOBALStoreDesc<GLOBAL_STORE_DWORD,        GLOBAL_STORE_DWORD_SADDR,
                   si_llvm_i32_ty,  0, 0>,
  GLOBALStoreDesc<GLOBAL_STORE_DWORDX2,      GLOBAL_STORE_DWORDX2_SADDR,
                   si_llvm_v2i32_ty, 0, 0>,
  GLOBALStoreDesc<GLOBAL_STORE_DWORDX3,      GLOBAL_STORE_DWORDX3_SADDR,
                   si_llvm_v3i32_ty, 0, 0>,
  GLOBALStoreDesc<GLOBAL_STORE_DWORDX4,      GLOBAL_STORE_DWORDX4_SADDR,
                   si_llvm_v4i32_ty, 0, 0>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.IsD16Hi,
                   (LLVMTrunc (LLVMLShr (GetNamedOperand $vdata, si_llvm_i32_ty),
                                         (ConstantInt si_llvm_i32_ty, 16)),
                              desc.StoreTy),
                   !if(desc.NeedsTrunc,
                       (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i32_ty),
                                   desc.StoreTy),
                       (GetNamedOperand $vdata))),
                (GetVal $addr))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.IsD16Hi,
                   (LLVMTrunc (LLVMLShr (GetNamedOperand $vdata, si_llvm_i32_ty),
                                         (ConstantInt si_llvm_i32_ty, 16)),
                              desc.StoreTy),
                   !if(desc.NeedsTrunc,
                       (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i32_ty),
                                   desc.StoreTy),
                       (GetNamedOperand $vdata))),
                (GetVal $addr))
  ]>;
}

//==============================================================================
// GLOBAL Instructions - _SADDR Load/Store variants (GFX9+)
// Same semantics as base GLOBAL loads/stores but with SADDR addressing mode.
// Uses scalar SGPR base address + VGPR offset instead of just VGPR.
// Sources: Same as GLOBAL Loads/Stores above.

//==============================================================================

// GLOBAL Load SADDR variants - empty semantics (same load behavior, different addressing)


// GLOBAL Store SADDR variants


//==============================================================================
// GLOBAL Instructions - Atomics
// Sources: see FLAT atomics citations; GLOBAL ATOMICS appear in the same
// per-arch chapters (gfx9+ Vega introduced GLOBAL_ATOMIC_*).
//==============================================================================

//------------------------------------------------------------------------------
// Binary RMW GLOBAL atomics — same uniform pattern as FLAT atomics but with
// CalculateGlobalAddr / CalculateGlobalSAddr address helpers.
//------------------------------------------------------------------------------

class GLOBALBinAtomicDesc<AtomicRMWOp op,
                           InstSI noret, InstSI rtn, InstSI rtnAgpr,
                           InstSI sad, InstSI sadRtn> {
  AtomicRMWOp Op = op;
  InstSI NoRet   = noret;
  InstSI Rtn     = rtn;
  InstSI RtnAgpr = rtnAgpr;
  InstSI Sad     = sad;
  InstSI SadRtn  = sadRtn;
}

foreach desc = [
  // 32-bit
  GLOBALBinAtomicDesc<AtomicRMWXchg,
    GLOBAL_ATOMIC_SWAP, GLOBAL_ATOMIC_SWAP_RTN, GLOBAL_ATOMIC_SWAP_RTN_agpr,
    GLOBAL_ATOMIC_SWAP_SADDR, GLOBAL_ATOMIC_SWAP_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWAdd,
    GLOBAL_ATOMIC_ADD, GLOBAL_ATOMIC_ADD_RTN, GLOBAL_ATOMIC_ADD_RTN_agpr,
    GLOBAL_ATOMIC_ADD_SADDR, GLOBAL_ATOMIC_ADD_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWSub,
    GLOBAL_ATOMIC_SUB, GLOBAL_ATOMIC_SUB_RTN, GLOBAL_ATOMIC_SUB_RTN_agpr,
    GLOBAL_ATOMIC_SUB_SADDR, GLOBAL_ATOMIC_SUB_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWAnd,
    GLOBAL_ATOMIC_AND, GLOBAL_ATOMIC_AND_RTN, GLOBAL_ATOMIC_AND_RTN_agpr,
    GLOBAL_ATOMIC_AND_SADDR, GLOBAL_ATOMIC_AND_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWOr,
    GLOBAL_ATOMIC_OR,  GLOBAL_ATOMIC_OR_RTN,  GLOBAL_ATOMIC_OR_RTN_agpr,
    GLOBAL_ATOMIC_OR_SADDR,  GLOBAL_ATOMIC_OR_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWXor,
    GLOBAL_ATOMIC_XOR, GLOBAL_ATOMIC_XOR_RTN, GLOBAL_ATOMIC_XOR_RTN_agpr,
    GLOBAL_ATOMIC_XOR_SADDR, GLOBAL_ATOMIC_XOR_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWMin,
    GLOBAL_ATOMIC_SMIN, GLOBAL_ATOMIC_SMIN_RTN, GLOBAL_ATOMIC_SMIN_RTN_agpr,
    GLOBAL_ATOMIC_SMIN_SADDR, GLOBAL_ATOMIC_SMIN_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUMin,
    GLOBAL_ATOMIC_UMIN, GLOBAL_ATOMIC_UMIN_RTN, GLOBAL_ATOMIC_UMIN_RTN_agpr,
    GLOBAL_ATOMIC_UMIN_SADDR, GLOBAL_ATOMIC_UMIN_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWMax,
    GLOBAL_ATOMIC_SMAX, GLOBAL_ATOMIC_SMAX_RTN, GLOBAL_ATOMIC_SMAX_RTN_agpr,
    GLOBAL_ATOMIC_SMAX_SADDR, GLOBAL_ATOMIC_SMAX_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUMax,
    GLOBAL_ATOMIC_UMAX, GLOBAL_ATOMIC_UMAX_RTN, GLOBAL_ATOMIC_UMAX_RTN_agpr,
    GLOBAL_ATOMIC_UMAX_SADDR, GLOBAL_ATOMIC_UMAX_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUIncWrap,
    GLOBAL_ATOMIC_INC, GLOBAL_ATOMIC_INC_RTN, GLOBAL_ATOMIC_INC_RTN_agpr,
    GLOBAL_ATOMIC_INC_SADDR, GLOBAL_ATOMIC_INC_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUDecWrap,
    GLOBAL_ATOMIC_DEC, GLOBAL_ATOMIC_DEC_RTN, GLOBAL_ATOMIC_DEC_RTN_agpr,
    GLOBAL_ATOMIC_DEC_SADDR, GLOBAL_ATOMIC_DEC_SADDR_RTN>,
  // 64-bit (X2)
  GLOBALBinAtomicDesc<AtomicRMWXchg,
    GLOBAL_ATOMIC_SWAP_X2, GLOBAL_ATOMIC_SWAP_X2_RTN, GLOBAL_ATOMIC_SWAP_X2_RTN_agpr,
    GLOBAL_ATOMIC_SWAP_X2_SADDR, GLOBAL_ATOMIC_SWAP_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWAdd,
    GLOBAL_ATOMIC_ADD_X2, GLOBAL_ATOMIC_ADD_X2_RTN, GLOBAL_ATOMIC_ADD_X2_RTN_agpr,
    GLOBAL_ATOMIC_ADD_X2_SADDR, GLOBAL_ATOMIC_ADD_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWSub,
    GLOBAL_ATOMIC_SUB_X2, GLOBAL_ATOMIC_SUB_X2_RTN, GLOBAL_ATOMIC_SUB_X2_RTN_agpr,
    GLOBAL_ATOMIC_SUB_X2_SADDR, GLOBAL_ATOMIC_SUB_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWAnd,
    GLOBAL_ATOMIC_AND_X2, GLOBAL_ATOMIC_AND_X2_RTN, GLOBAL_ATOMIC_AND_X2_RTN_agpr,
    GLOBAL_ATOMIC_AND_X2_SADDR, GLOBAL_ATOMIC_AND_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWOr,
    GLOBAL_ATOMIC_OR_X2,  GLOBAL_ATOMIC_OR_X2_RTN,  GLOBAL_ATOMIC_OR_X2_RTN_agpr,
    GLOBAL_ATOMIC_OR_X2_SADDR,  GLOBAL_ATOMIC_OR_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWXor,
    GLOBAL_ATOMIC_XOR_X2, GLOBAL_ATOMIC_XOR_X2_RTN, GLOBAL_ATOMIC_XOR_X2_RTN_agpr,
    GLOBAL_ATOMIC_XOR_X2_SADDR, GLOBAL_ATOMIC_XOR_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWMin,
    GLOBAL_ATOMIC_SMIN_X2, GLOBAL_ATOMIC_SMIN_X2_RTN, GLOBAL_ATOMIC_SMIN_X2_RTN_agpr,
    GLOBAL_ATOMIC_SMIN_X2_SADDR, GLOBAL_ATOMIC_SMIN_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUMin,
    GLOBAL_ATOMIC_UMIN_X2, GLOBAL_ATOMIC_UMIN_X2_RTN, GLOBAL_ATOMIC_UMIN_X2_RTN_agpr,
    GLOBAL_ATOMIC_UMIN_X2_SADDR, GLOBAL_ATOMIC_UMIN_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWMax,
    GLOBAL_ATOMIC_SMAX_X2, GLOBAL_ATOMIC_SMAX_X2_RTN, GLOBAL_ATOMIC_SMAX_X2_RTN_agpr,
    GLOBAL_ATOMIC_SMAX_X2_SADDR, GLOBAL_ATOMIC_SMAX_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUMax,
    GLOBAL_ATOMIC_UMAX_X2, GLOBAL_ATOMIC_UMAX_X2_RTN, GLOBAL_ATOMIC_UMAX_X2_RTN_agpr,
    GLOBAL_ATOMIC_UMAX_X2_SADDR, GLOBAL_ATOMIC_UMAX_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUIncWrap,
    GLOBAL_ATOMIC_INC_X2, GLOBAL_ATOMIC_INC_X2_RTN, GLOBAL_ATOMIC_INC_X2_RTN_agpr,
    GLOBAL_ATOMIC_INC_X2_SADDR, GLOBAL_ATOMIC_INC_X2_SADDR_RTN>,
  GLOBALBinAtomicDesc<AtomicRMWUDecWrap,
    GLOBAL_ATOMIC_DEC_X2, GLOBAL_ATOMIC_DEC_X2_RTN, GLOBAL_ATOMIC_DEC_X2_RTN_agpr,
    GLOBAL_ATOMIC_DEC_X2_SADDR, GLOBAL_ATOMIC_DEC_X2_SADDR_RTN>
] in {
  def : InstSISemantic<desc.NoRet, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (LLVMAtomicRMW desc.Op, (GetVal $addr), (GetNamedOperand $vdata),
                    AlignNone, (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty)))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (LLVMAtomicRMW desc.Op, (GetVal $addr), (GetNamedOperand $vdata),
                    AlignNone, (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty)))
  ]>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, [
      (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                          (GetNamedOperand $offset)>.ret),
      (SetNamedOperand $vdst, (LLVMAtomicRMW desc.Op, (GetVal $addr),
                                              (GetNamedOperand $vdata),
                                              AlignNone,
                                              (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty))))
    ]>;
  def : InstSISemantic<desc.SadRtn, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (LLVMAtomicRMW desc.Op, (GetVal $addr),
                                            (GetNamedOperand $vdata),
                                            AlignNone, (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty))))
  ]>;
}

//------------------------------------------------------------------------------
// GLOBAL binary RMW — _SADDR_RTN_agpr forms (CDNA2+ AGPR-destination
// variants of the SADDR_RTN form). Same semantic body, just a different
// opcode list.
//------------------------------------------------------------------------------

class GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWOp op, InstSI inst> {
  AtomicRMWOp Op = op;
  InstSI Inst = inst;
}

foreach desc = [
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWXchg,     GLOBAL_ATOMIC_SWAP_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWAdd,      GLOBAL_ATOMIC_ADD_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWSub,      GLOBAL_ATOMIC_SUB_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWAnd,      GLOBAL_ATOMIC_AND_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWOr,       GLOBAL_ATOMIC_OR_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWXor,      GLOBAL_ATOMIC_XOR_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWMin,      GLOBAL_ATOMIC_SMIN_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUMin,     GLOBAL_ATOMIC_UMIN_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWMax,      GLOBAL_ATOMIC_SMAX_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUMax,     GLOBAL_ATOMIC_UMAX_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUIncWrap, GLOBAL_ATOMIC_INC_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUDecWrap, GLOBAL_ATOMIC_DEC_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWXchg,     GLOBAL_ATOMIC_SWAP_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWAdd,      GLOBAL_ATOMIC_ADD_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWSub,      GLOBAL_ATOMIC_SUB_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWAnd,      GLOBAL_ATOMIC_AND_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWOr,       GLOBAL_ATOMIC_OR_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWXor,      GLOBAL_ATOMIC_XOR_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWMin,      GLOBAL_ATOMIC_SMIN_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUMin,     GLOBAL_ATOMIC_UMIN_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWMax,      GLOBAL_ATOMIC_SMAX_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUMax,     GLOBAL_ATOMIC_UMAX_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUIncWrap, GLOBAL_ATOMIC_INC_X2_SADDR_RTN_agpr>,
  GLOBALBinAtomicSadRtnAgprDesc<AtomicRMWUDecWrap, GLOBAL_ATOMIC_DEC_X2_SADDR_RTN_agpr>
] in
  def : InstSISemantic<desc.Inst, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst, (LLVMAtomicRMW desc.Op, (GetVal $addr),
                                            (GetNamedOperand $vdata),
                                            AlignNone, (GetOrdering (GetNamedOperand $cpol, si_llvm_i32_ty)), (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty))))
  ]>;

// GLOBAL_ATOMIC_ORDERED_ADD_B64 — inter-wavefront serialized 64-bit add,
// ordered by an HW sequencer. Modeled via int_amdgcn_global_atomic_ordered_add_b64
// rather than plain atomicrmw so downstream passes don't reorder/optimize
// it as a regular memory atomic.
class OrderedAddDesc<InstSI inst, bit hasOut> {
  InstSI Inst = inst;
  bit HasOut = hasOut;
}

foreach desc = [
  OrderedAddDesc<GLOBAL_ATOMIC_ORDERED_ADD_B64,           0>,
  OrderedAddDesc<GLOBAL_ATOMIC_ORDERED_ADD_B64_RTN,       1>,
  OrderedAddDesc<GLOBAL_ATOMIC_ORDERED_ADD_B64_RTN_agpr,  1>
] in {
  def : InstSISemantic<desc.Inst, !if(desc.HasOut,
    [(DefVal $a, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
     (SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_i64_ty,
                       int_amdgcn_global_atomic_ordered_add_b64,
                       [(GetVal $a),
                        (GetNamedOperand $vdata, si_llvm_i64_ty)]))],
    [(DefVal $a, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
     (LLVMIntrinsic si_llvm_i64_ty,
                    int_amdgcn_global_atomic_ordered_add_b64,
                    [(GetVal $a),
                     (GetNamedOperand $vdata, si_llvm_i64_ty)])])>;
}

foreach desc = [
  OrderedAddDesc<GLOBAL_ATOMIC_ORDERED_ADD_B64_SADDR,           0>,
  OrderedAddDesc<GLOBAL_ATOMIC_ORDERED_ADD_B64_SADDR_RTN,       1>,
  OrderedAddDesc<GLOBAL_ATOMIC_ORDERED_ADD_B64_SADDR_RTN_agpr,  1>
] in {
  def : InstSISemantic<desc.Inst, !if(desc.HasOut,
    [(DefVal $a, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
     (SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_i64_ty,
                       int_amdgcn_global_atomic_ordered_add_b64,
                       [(GetVal $a),
                        (GetNamedOperand $vdata, si_llvm_i64_ty)]))],
    [(DefVal $a, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
     (LLVMIntrinsic si_llvm_i64_ty,
                    int_amdgcn_global_atomic_ordered_add_b64,
                    [(GetVal $a),
                     (GetNamedOperand $vdata, si_llvm_i64_ty)])])>;
}

//==============================================================================
// GLOBAL Instructions - X2 (64-bit) Atomic variants
// Same atomic operations as 32-bit base but operating on 64-bit data.
// Sources: Same as GLOBAL Atomics above.

//==============================================================================



//==============================================================================
// GLOBAL Instructions - _SADDR Atomic variants
// Sources: Same as GLOBAL Atomics above.

//==============================================================================



//==============================================================================
// GLOBAL Instructions - _RTN Atomic variants
// Sources: Same as GLOBAL Atomics above.

//==============================================================================



//==============================================================================
// GLOBAL Instructions - _SADDR_RTN Atomic variants
// Sources: Same as GLOBAL Atomics above.

//==============================================================================



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



//==============================================================================
// GLOBAL Instructions - ADDTID variants (GFX10+)
// Load/store with thread-ID offset added to the address.

//==============================================================================



//==============================================================================
// GLOBAL Instructions - Block Load/Store (GFX12+)
// Large block transfer operations.

//==============================================================================



//==============================================================================
// GLOBAL Instructions - Cache Management (GFX12+)

//==============================================================================

// GLOBAL_INV / GLOBAL_WB / GLOBAL_WBINV — gfx12 scope-parameterized cache
// fences. The scope (bits[4:3] of cpol) selects CU / SE / device / system
// level, decoded by Translator.getSyncScope. Semantics:
//   GLOBAL_INV   → acquire fence (memory becomes visible to this thread)
//   GLOBAL_WB    → release fence (this thread's writes become visible)
//   GLOBAL_WBINV → acquire-release fence (both directions)
def GLOBAL_INV_sem : InstSISemantic<GLOBAL_INV, [
  (LLVMFence AtomicOrderingAcquire,
             (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty)))
]>;

def GLOBAL_WB_sem : InstSISemantic<GLOBAL_WB, [
  (LLVMFence AtomicOrderingRelease,
             (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty)))
]>;

def GLOBAL_WBINV_sem : InstSISemantic<GLOBAL_WBINV, [
  (LLVMFence AtomicOrderingAcqRel,
             (GetSyncScope (GetNamedOperand $cpol, si_llvm_i32_ty)))
]>;

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

//==============================================================================
// SCRATCH Instructions - D16 Loads
// Sources: Same as SCRATCH Loads above.

//==============================================================================

//==============================================================================
// SCRATCH Instructions - Stores
// Sources: Same as SCRATCH Loads above.

//==============================================================================

//==============================================================================
// SCRATCH Instructions - _SADDR, _SVS, and _ST variants
// _SADDR: Uses scalar SGPR address + offset (no VGPR base)
// _SVS: Stack + VGPR + SGPR addressing (GFX12+)
// _ST: Stack-relative addressing, no VGPR (GFX12+)
// Same load/store semantics as base, different addressing modes.
// Sources: Same as SCRATCH Loads/Stores above.

//==============================================================================

// SCRATCH Load _SADDR variants


// SCRATCH Load _SVS variants


// SCRATCH Load _ST variants


// SCRATCH Store _SADDR variants


// SCRATCH Store _SVS variants


// SCRATCH Store _ST variants


//==============================================================================
// SCRATCH Instructions - Load to LDS
// Load from scratch memory directly into LDS, bypassing VGPRs.
// Each has base, _SADDR, _SVS, and _ST variants.
// Sources: Same as SCRATCH Loads above.

//==============================================================================



//==============================================================================
// SCRATCH Instructions - Block Load/Store (GFX12+)
// Large block transfer operations on scratch memory.
// Each has base, _SADDR, _SVS, and _ST variants.

//==============================================================================



//==============================================================================
// FLAT _agpr variants (Phase 5) — replicated from base records.
//==============================================================================

// _agpr alias of FLAT_ATOMIC_ADD_F32_RTN
// _agpr alias of FLAT_ATOMIC_ADD_F64_RTN
// _agpr alias of FLAT_ATOMIC_ADD_RTN
// _agpr alias of FLAT_ATOMIC_ADD_X2_RTN
// _agpr alias of FLAT_ATOMIC_AND_RTN
// _agpr alias of FLAT_ATOMIC_AND_X2_RTN
// _agpr alias of FLAT_ATOMIC_CMPSWAP_RTN
// _agpr alias of FLAT_ATOMIC_CMPSWAP_X2_RTN
// _agpr alias of FLAT_ATOMIC_COND_SUB_U32_RTN
// _agpr alias of FLAT_ATOMIC_CSUB_U32_RTN
// _agpr alias of FLAT_ATOMIC_DEC_RTN
// _agpr alias of FLAT_ATOMIC_DEC_X2_RTN
// _agpr alias of FLAT_ATOMIC_FCMPSWAP_RTN
// _agpr alias of FLAT_ATOMIC_FCMPSWAP_X2_RTN
// _agpr alias of FLAT_ATOMIC_FMAX_RTN
// _agpr alias of FLAT_ATOMIC_FMIN_RTN
// _agpr alias of FLAT_ATOMIC_INC_RTN
// _agpr alias of FLAT_ATOMIC_INC_X2_RTN
// _agpr alias of FLAT_ATOMIC_MAX_F64_RTN
// _agpr alias of FLAT_ATOMIC_MIN_F64_RTN
// _agpr alias of FLAT_ATOMIC_OR_RTN
// _agpr alias of FLAT_ATOMIC_OR_X2_RTN
// _agpr alias of FLAT_ATOMIC_PK_ADD_BF16_RTN
// _agpr alias of FLAT_ATOMIC_PK_ADD_F16_RTN
// _agpr alias of FLAT_ATOMIC_SMAX_RTN
// _agpr alias of FLAT_ATOMIC_SMAX_X2_RTN
// _agpr alias of FLAT_ATOMIC_SMIN_RTN
// _agpr alias of FLAT_ATOMIC_SMIN_X2_RTN
// _agpr alias of FLAT_ATOMIC_SUB_RTN
// _agpr alias of FLAT_ATOMIC_SUB_X2_RTN
// _agpr alias of FLAT_ATOMIC_SWAP_RTN
// _agpr alias of FLAT_ATOMIC_SWAP_X2_RTN
// _agpr alias of FLAT_ATOMIC_UMAX_RTN
// _agpr alias of FLAT_ATOMIC_UMAX_X2_RTN
// _agpr alias of FLAT_ATOMIC_UMIN_RTN
// _agpr alias of FLAT_ATOMIC_UMIN_X2_RTN
// _agpr alias of FLAT_ATOMIC_XOR_RTN
// _agpr alias of FLAT_ATOMIC_XOR_X2_RTN
//==============================================================================
// GLOBAL _agpr variants (Phase 5) — replicated from base records.
//==============================================================================



//==============================================================================
// SCRATCH _agpr variants (Phase 5) — replicated from base records.
//==============================================================================



//==============================================================================
// GLOBAL atomic variants (Phase 6) — _RTN / _SADDR / _SADDR_RTN / _agpr
// Derived from existing base GLOBAL_ATOMIC_X semantics:
//   _RTN:   writeback target switched from $vdata to $vdst
//   _SADDR: address calc switched from CalculateGlobalAddr to CalculateGlobalSAddr
//   _agpr:  AGPR destination form; same semantic as VGPR form
//==============================================================================

//==============================================================================
// GLOBAL_ATOMIC X2 (64-bit) base variants (Phase 6) — cloned from FLAT analogs
//==============================================================================


//==============================================================================
// GLOBAL atomic variants (Phase 6) — _RTN / _SADDR / _SADDR_RTN / _agpr
// Derived from existing base GLOBAL_ATOMIC_X semantics:
//   _RTN:   writeback target switched from $vdata to $vdst
//   _SADDR: address calc switched from CalculateGlobalAddr to CalculateGlobalSAddr
//   _agpr:  AGPR destination form; same semantic as VGPR form
//==============================================================================


//==============================================================================
// SCRATCH _SADDR / _ST / _SVS variants (Phase 7)
// _SADDR: scalar base + offset.  _ST: offset only (no base).  _SVS: scalar base + vector index + offset.
//==============================================================================

// _SADDR variant of SCRATCH_LOAD_DWORDX2
// _ST variant of SCRATCH_LOAD_DWORDX2
// _SVS variant of SCRATCH_LOAD_DWORDX2
// _SADDR variant of SCRATCH_LOAD_DWORDX3
// _ST variant of SCRATCH_LOAD_DWORDX3
// _SVS variant of SCRATCH_LOAD_DWORDX3
// _SADDR variant of SCRATCH_LOAD_DWORDX4
// _ST variant of SCRATCH_LOAD_DWORDX4
// _SVS variant of SCRATCH_LOAD_DWORDX4
// _SADDR variant of SCRATCH_LOAD_DWORD
// _ST variant of SCRATCH_LOAD_DWORD
// _SVS variant of SCRATCH_LOAD_DWORD
// _SADDR variant of SCRATCH_LOAD_SBYTE_D16_HI
// _ST variant of SCRATCH_LOAD_SBYTE_D16_HI
// _SVS variant of SCRATCH_LOAD_SBYTE_D16_HI
// _SADDR variant of SCRATCH_LOAD_SBYTE_D16
// _ST variant of SCRATCH_LOAD_SBYTE_D16
// _SVS variant of SCRATCH_LOAD_SBYTE_D16
// _SADDR variant of SCRATCH_LOAD_SBYTE
// _ST variant of SCRATCH_LOAD_SBYTE
// _SVS variant of SCRATCH_LOAD_SBYTE
// _SADDR variant of SCRATCH_LOAD_SHORT_D16_HI
// _ST variant of SCRATCH_LOAD_SHORT_D16_HI
// _SVS variant of SCRATCH_LOAD_SHORT_D16_HI
// _SADDR variant of SCRATCH_LOAD_SHORT_D16
// _ST variant of SCRATCH_LOAD_SHORT_D16
// _SVS variant of SCRATCH_LOAD_SHORT_D16
// _SADDR variant of SCRATCH_LOAD_SSHORT
// _ST variant of SCRATCH_LOAD_SSHORT
// _SVS variant of SCRATCH_LOAD_SSHORT
// _SADDR variant of SCRATCH_LOAD_UBYTE_D16_HI
// _ST variant of SCRATCH_LOAD_UBYTE_D16_HI
// _SVS variant of SCRATCH_LOAD_UBYTE_D16_HI
// _SADDR variant of SCRATCH_LOAD_UBYTE_D16
// _ST variant of SCRATCH_LOAD_UBYTE_D16
// _SVS variant of SCRATCH_LOAD_UBYTE_D16
// _SADDR variant of SCRATCH_LOAD_UBYTE
// _ST variant of SCRATCH_LOAD_UBYTE
// _SVS variant of SCRATCH_LOAD_UBYTE
// _SADDR variant of SCRATCH_LOAD_USHORT
// _ST variant of SCRATCH_LOAD_USHORT
// _SVS variant of SCRATCH_LOAD_USHORT
// _SADDR variant of SCRATCH_STORE_BYTE_D16_HI
// _ST variant of SCRATCH_STORE_BYTE_D16_HI
// _SVS variant of SCRATCH_STORE_BYTE_D16_HI
// _SADDR variant of SCRATCH_STORE_BYTE
// _ST variant of SCRATCH_STORE_BYTE
// _SVS variant of SCRATCH_STORE_BYTE
// _SADDR variant of SCRATCH_STORE_DWORDX2
// _ST variant of SCRATCH_STORE_DWORDX2
// _SVS variant of SCRATCH_STORE_DWORDX2
// _SADDR variant of SCRATCH_STORE_DWORDX3
// _ST variant of SCRATCH_STORE_DWORDX3
// _SVS variant of SCRATCH_STORE_DWORDX3
// _SADDR variant of SCRATCH_STORE_DWORDX4
// _ST variant of SCRATCH_STORE_DWORDX4
// _SVS variant of SCRATCH_STORE_DWORDX4
// _SADDR variant of SCRATCH_STORE_DWORD
// _ST variant of SCRATCH_STORE_DWORD
// _SVS variant of SCRATCH_STORE_DWORD
// _SADDR variant of SCRATCH_STORE_SHORT_D16_HI
// _ST variant of SCRATCH_STORE_SHORT_D16_HI
// _SVS variant of SCRATCH_STORE_SHORT_D16_HI
// _SADDR variant of SCRATCH_STORE_SHORT
// _ST variant of SCRATCH_STORE_SHORT
// _SVS variant of SCRATCH_STORE_SHORT
//==============================================================================
// GLOBAL_ cross-product alias variants (Phase 8 generic)
//==============================================================================

// alias of GLOBAL_LOAD_DWORD
// alias of GLOBAL_LOAD_DWORDX3
// alias of GLOBAL_LOAD_DWORDX3_SADDR
// alias of GLOBAL_LOAD_DWORDX4
// alias of GLOBAL_LOAD_DWORDX4_SADDR
// alias of GLOBAL_LOAD_DWORD_SADDR
// alias of GLOBAL_LOAD_SBYTE
// alias of GLOBAL_LOAD_SBYTE_SADDR
// alias of GLOBAL_LOAD_SSHORT
// alias of GLOBAL_LOAD_SSHORT_SADDR
// alias of GLOBAL_LOAD_UBYTE
// alias of GLOBAL_LOAD_UBYTE_SADDR
// alias of GLOBAL_LOAD_USHORT
// alias of GLOBAL_LOAD_USHORT_SADDR
//==============================================================================
// SCRATCH_ cross-product alias variants (Phase 8 generic)
//==============================================================================

// alias of SCRATCH_LOAD_DWORD
// alias of SCRATCH_LOAD_DWORD_SADDR
// alias of SCRATCH_LOAD_DWORD_ST
// alias of SCRATCH_LOAD_DWORD_SVS
// alias of SCRATCH_LOAD_SBYTE
// alias of SCRATCH_LOAD_SBYTE_SADDR
// alias of SCRATCH_LOAD_SBYTE_ST
// alias of SCRATCH_LOAD_SBYTE_SVS
// alias of SCRATCH_LOAD_SSHORT
// alias of SCRATCH_LOAD_SSHORT_SADDR
// alias of SCRATCH_LOAD_SSHORT_ST
// alias of SCRATCH_LOAD_SSHORT_SVS
// alias of SCRATCH_LOAD_UBYTE
// alias of SCRATCH_LOAD_UBYTE_SADDR
// alias of SCRATCH_LOAD_UBYTE_ST
// alias of SCRATCH_LOAD_UBYTE_SVS
// alias of SCRATCH_LOAD_USHORT
// alias of SCRATCH_LOAD_USHORT_SADDR
// alias of SCRATCH_LOAD_USHORT_ST
// alias of SCRATCH_LOAD_USHORT_SVS
//==============================================================================
// FLAT atomic variants (Phase 9) — _RTN / _SADDR / _SADDR_RTN / _RTN_agpr
// Generated from base FLAT_ATOMIC_X semantics: _RTN swaps writeback target
// $vdata→$vdst; _SADDR substitutes addr calc to use $saddr + zext($vaddr).
//==============================================================================


//==============================================================================
// gfx12 specialized FLAT/GLOBAL ops — modeled via dedicated intrinsics.
//
// These opcodes previously emitted a `s_waitcnt(0)` fence stub. The fence
// was wrong for analysis purposes; it advertised a memory-fence side-effect
// the actual instructions don't have. Per the DSSem.td precedent, the stubs
// were dropped and proper intrinsic-based semantics added where available.
//==============================================================================

//------------------------------------------------------------------------------
// FLAT / GLOBAL PREFETCH — int_amdgcn_flat_prefetch / int_amdgcn_global_prefetch
// Signature: void (ptr, i32 cpol_imm).
//------------------------------------------------------------------------------

def FLAT_PREFETCH_B8_sem : InstSISemantic<FLAT_PREFETCH_B8, [
  (DefVal $addr, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                    (GetNamedOperand $offset)>.ret),
  (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_flat_prefetch,
                 [(GetVal $addr),
                  (GetNamedOperand $cpol, si_llvm_i32_ty)])
]>;

def FLAT_PREFETCH_B8_SADDR_sem : InstSISemantic<FLAT_PREFETCH_B8_SADDR, [
  (DefVal $addr, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                     (GetNamedOperand $vaddr),
                                     (GetNamedOperand $offset)>.ret),
  (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_flat_prefetch,
                 [(GetVal $addr),
                  (GetNamedOperand $cpol, si_llvm_i32_ty)])
]>;

def GLOBAL_PREFETCH_B8_sem : InstSISemantic<GLOBAL_PREFETCH_B8, [
  (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
  (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_global_prefetch,
                 [(GetVal $addr),
                  (GetNamedOperand $cpol, si_llvm_i32_ty)])
]>;

def GLOBAL_PREFETCH_B8_SADDR_sem : InstSISemantic<GLOBAL_PREFETCH_B8_SADDR, [
  (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                       (GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret),
  (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_global_prefetch,
                 [(GetVal $addr),
                  (GetNamedOperand $cpol, si_llvm_i32_ty)])
]>;

//------------------------------------------------------------------------------
// GLOBAL_LOAD_MONITOR_B{32,64,128} (+ SADDR) — cache-line monitor hint load.
// Intrinsic: int_amdgcn_global_load_monitor_b{N}(ptr addrspace(1), i32 cpol).
//------------------------------------------------------------------------------

class GlobalMonitorDesc<InstSI base, InstSI sad, SILLVMType retTy, Intrinsic intr> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType RetTy = retTy;
  Intrinsic Intr  = intr;
}

foreach desc = [
  GlobalMonitorDesc<GLOBAL_LOAD_MONITOR_B32,  GLOBAL_LOAD_MONITOR_B32_SADDR,
                     si_llvm_i32_ty,   int_amdgcn_global_load_monitor_b32>,
  GlobalMonitorDesc<GLOBAL_LOAD_MONITOR_B64,  GLOBAL_LOAD_MONITOR_B64_SADDR,
                     si_llvm_i64_ty,   int_amdgcn_global_load_monitor_b64>,
  GlobalMonitorDesc<GLOBAL_LOAD_MONITOR_B128, GLOBAL_LOAD_MONITOR_B128_SADDR,
                     si_llvm_v4i32_ty, int_amdgcn_global_load_monitor_b128>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr,
                       [(GetVal $addr),
                        (GetNamedOperand $cpol, si_llvm_i32_ty)]))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr,
                       [(GetVal $addr),
                        (GetNamedOperand $cpol, si_llvm_i32_ty)]))
  ]>;
}

//------------------------------------------------------------------------------
// GLOBAL_LOAD_TR{4,6}_B{64,96} (+ SADDR) — WMMA transpose loads (RDNA4).
// Intrinsics: int_amdgcn_global_load_tr4_b64 (returns i32), tr6_b96 (v3i32).
// TR_B64_w{32,64} and TR_B128_w{32,64} are SADDR-only multi-arg overloads
// of int_amdgcn_global_load_tr_b{64,128} — deferred (LLVM intrinsic is
// overloaded by return type which requires extra emitter support).
//------------------------------------------------------------------------------

class GlobalTRLoadDesc<InstSI base, InstSI sad, SILLVMType retTy, Intrinsic intr> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType RetTy = retTy;
  Intrinsic Intr  = intr;
}

foreach desc = [
  GlobalTRLoadDesc<GLOBAL_LOAD_TR4_B64, GLOBAL_LOAD_TR4_B64_SADDR,
                    si_llvm_i32_ty,   int_amdgcn_global_load_tr4_b64>,
  GlobalTRLoadDesc<GLOBAL_LOAD_TR6_B96, GLOBAL_LOAD_TR6_B96_SADDR,
                    si_llvm_v3i32_ty, int_amdgcn_global_load_tr6_b96>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr, [(GetVal $addr)]))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr, [(GetVal $addr)]))
  ]>;
}

//------------------------------------------------------------------------------
// GLOBAL_LOAD_ASYNC_TO_LDS_B{8,32,64,128} (+ SADDR) — async DRAM→LDS copy.
// Intrinsic: int_amdgcn_global_load_async_to_lds_b{N}(
//                global_ptr, local_ptr, i32 offset_imm, i32 cpol_imm).
// MIR (per FLAT_Global_Load_LDS_Pseudo IsAsync=1): vdst slot holds the LDS
// base register (it's an input register, just named vdst by the pseudo).
//------------------------------------------------------------------------------

class GlobalAsyncToLdsDesc<InstSI base, InstSI sad, Intrinsic intr> {
  InstSI Base = base;
  InstSI Sad  = sad;
  Intrinsic Intr = intr;
}

foreach desc = [
  GlobalAsyncToLdsDesc<GLOBAL_LOAD_ASYNC_TO_LDS_B8,
                        GLOBAL_LOAD_ASYNC_TO_LDS_B8_SADDR,
                        int_amdgcn_global_load_async_to_lds_b8>,
  GlobalAsyncToLdsDesc<GLOBAL_LOAD_ASYNC_TO_LDS_B32,
                        GLOBAL_LOAD_ASYNC_TO_LDS_B32_SADDR,
                        int_amdgcn_global_load_async_to_lds_b32>,
  GlobalAsyncToLdsDesc<GLOBAL_LOAD_ASYNC_TO_LDS_B64,
                        GLOBAL_LOAD_ASYNC_TO_LDS_B64_SADDR,
                        int_amdgcn_global_load_async_to_lds_b64>,
  GlobalAsyncToLdsDesc<GLOBAL_LOAD_ASYNC_TO_LDS_B128,
                        GLOBAL_LOAD_ASYNC_TO_LDS_B128_SADDR,
                        int_amdgcn_global_load_async_to_lds_b128>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $g, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                     (GetNamedOperand $offset)>.ret),
    (DefVal $l, (LLVMIntToPtr (GetNamedOperand $vdst, si_llvm_i32_ty),
                                local_ptr_ty)),
    (LLVMIntrinsic si_llvm_void_ty, desc.Intr,
                   [(GetVal $g), (GetVal $l),
                    (ConstantInt si_llvm_i32_ty, 0),
                    (GetNamedOperand $cpol, si_llvm_i32_ty)])
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $g, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                      (GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (DefVal $l, (LLVMIntToPtr (GetNamedOperand $vdst, si_llvm_i32_ty),
                                local_ptr_ty)),
    (LLVMIntrinsic si_llvm_void_ty, desc.Intr,
                   [(GetVal $g), (GetVal $l),
                    (ConstantInt si_llvm_i32_ty, 0),
                    (GetNamedOperand $cpol, si_llvm_i32_ty)])
  ]>;
}

//------------------------------------------------------------------------------
// GLOBAL_STORE_ASYNC_FROM_LDS_B{8,32,64,128} (+ SADDR) — async LDS→DRAM copy.
// Intrinsic: int_amdgcn_global_store_async_from_lds_b{N}(
//                global_ptr, local_ptr, i32 offset_imm, i32 cpol_imm).
//------------------------------------------------------------------------------

class GlobalAsyncFromLdsDesc<InstSI base, InstSI sad, Intrinsic intr> {
  InstSI Base = base;
  InstSI Sad  = sad;
  Intrinsic Intr = intr;
}

foreach desc = [
  GlobalAsyncFromLdsDesc<GLOBAL_STORE_ASYNC_FROM_LDS_B8,
                          GLOBAL_STORE_ASYNC_FROM_LDS_B8_SADDR,
                          int_amdgcn_global_store_async_from_lds_b8>,
  GlobalAsyncFromLdsDesc<GLOBAL_STORE_ASYNC_FROM_LDS_B32,
                          GLOBAL_STORE_ASYNC_FROM_LDS_B32_SADDR,
                          int_amdgcn_global_store_async_from_lds_b32>,
  GlobalAsyncFromLdsDesc<GLOBAL_STORE_ASYNC_FROM_LDS_B64,
                          GLOBAL_STORE_ASYNC_FROM_LDS_B64_SADDR,
                          int_amdgcn_global_store_async_from_lds_b64>,
  GlobalAsyncFromLdsDesc<GLOBAL_STORE_ASYNC_FROM_LDS_B128,
                          GLOBAL_STORE_ASYNC_FROM_LDS_B128_SADDR,
                          int_amdgcn_global_store_async_from_lds_b128>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $g, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                     (GetNamedOperand $offset)>.ret),
    (DefVal $l, (LLVMIntToPtr (GetNamedOperand $vdata, si_llvm_i32_ty),
                                local_ptr_ty)),
    (LLVMIntrinsic si_llvm_void_ty, desc.Intr,
                   [(GetVal $g), (GetVal $l),
                    (ConstantInt si_llvm_i32_ty, 0),
                    (GetNamedOperand $cpol, si_llvm_i32_ty)])
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $g, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                      (GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (DefVal $l, (LLVMIntToPtr (GetNamedOperand $vdata, si_llvm_i32_ty),
                                local_ptr_ty)),
    (LLVMIntrinsic si_llvm_void_ty, desc.Intr,
                   [(GetVal $g), (GetVal $l),
                    (ConstantInt si_llvm_i32_ty, 0),
                    (GetNamedOperand $cpol, si_llvm_i32_ty)])
  ]>;
}

//==============================================================================
// Unmodeled remaining ops — fall through to MIInlineAsmEmitter:
//   - GLOBAL_LOAD/STORE_BLOCK (+ SADDR)
//   - SCRATCH_LOAD/STORE_BLOCK (+ _ST / _SVS / SADDR)
//   - GLOBAL_LOAD_DWORD_ADDTID (+ SADDR), GLOBAL_STORE_DWORD_ADDTID (+ SADDR)
//   - CLUSTER_LOAD_B{32,64,128} (+ SADDR), CLUSTER_LOAD_ASYNC_TO_LDS_*
//   - GLOBAL_LOAD_TR_B{64,128}_w{32,64}_SADDR (overloaded-return intrinsic)
//   - TENSOR_SAVE, TENSOR_STOP
// These need M0 / mask broadcast handling, dedicated BLOCK semantics, or
// overloaded-return-type intrinsic emitter support — not in scope here.
//==============================================================================


//==============================================================================
// FP / CMPSWAP / specialty atomics (FLAT + GLOBAL) — foreach compression.
//
// FLAT has 5 form types per op:  NoRet, RTN, RTN_agpr, SADDR, SADDR_RTN.
// GLOBAL has 6 form types per op: + SADDR_RTN_agpr (CDNA2+).
//==============================================================================

//------------------------------------------------------------------------------
// CMPSWAP / CMPSWAP_X2 — integer compare-and-swap.
//   $vdata is a v2-typed register: element 0 = src (new value),
//                                  element 1 = cmp (compare value).
//   if MEM[A] == cmp: MEM[A] = src
//   RTN: $vdst = old MEM[A]
//------------------------------------------------------------------------------

class CmpSwapForeachDesc<SILLVMType scalarTy, SILLVMType vecTy,
                          InstSI base, list<InstSI> hasAddrForms> {
  SILLVMType ScalarTy = scalarTy;
  SILLVMType VecTy = vecTy;
  InstSI Base = base;            // the NoRet form
  list<InstSI> Forms = hasAddrForms;
}

// CMPSWAP semantic body — emitted once for each (segment, opcode, form).
// addr  : pointer expression (already computed)
// outOp : "no" → side-effect only; "yes" → SetNamedOperand $vdst, $tmp
class CmpSwapBody<SILLVMType scalarTy, SILLVMType vecTy, dag addr, bit hasOut> {
  list<dag> body = !if(hasOut,
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad scalarTy, (GetVal $a))),
      (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata, vecTy),
                                          (ConstantInt si_llvm_i32_ty, 1))),
      (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata, vecTy),
                                          (ConstantInt si_llvm_i32_ty, 0))),
      (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)),
                              (GetVal $src), (GetVal $tmp)),
                  (GetVal $a)),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ],
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad scalarTy, (GetVal $a))),
      (DefVal $cmp, (LLVMExtractElement (GetNamedOperand $vdata, vecTy),
                                          (ConstantInt si_llvm_i32_ty, 1))),
      (DefVal $src, (LLVMExtractElement (GetNamedOperand $vdata, vecTy),
                                          (ConstantInt si_llvm_i32_ty, 0))),
      (LLVMStore (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $tmp), (GetVal $cmp)),
                              (GetVal $src), (GetVal $tmp)),
                  (GetVal $a))
    ]);
}

// FLAT CMPSWAP / CMPSWAP_X2 — 5 forms each (no SADDR_RTN_agpr).
foreach desc = [
  // 32-bit
  [FLAT_ATOMIC_CMPSWAP, FLAT_ATOMIC_CMPSWAP_RTN, FLAT_ATOMIC_CMPSWAP_RTN_agpr,
   FLAT_ATOMIC_CMPSWAP_SADDR, FLAT_ATOMIC_CMPSWAP_SADDR_RTN],
  // 64-bit (X2)
  [FLAT_ATOMIC_CMPSWAP_X2, FLAT_ATOMIC_CMPSWAP_X2_RTN, FLAT_ATOMIC_CMPSWAP_X2_RTN_agpr,
   FLAT_ATOMIC_CMPSWAP_X2_SADDR, FLAT_ATOMIC_CMPSWAP_X2_SADDR_RTN]
] in {
  defvar isX2 = !ne(!find(!cast<string>(desc[0]), "_X2"), -1);
  defvar scalarTy = !if(isX2, si_llvm_i64_ty, si_llvm_i32_ty);
  defvar vecTy = !if(isX2, si_llvm_v2i64_ty, si_llvm_v2i32_ty);
  // NoRet
  def : InstSISemantic<desc[0],
    CmpSwapBody<scalarTy, vecTy,
                CalculateFlatAddr<(GetNamedOperand $vaddr),
                                   (GetNamedOperand $offset)>.ret, 0>.body>;
  // RTN, RTN_agpr (base)
  foreach rtnInst = [desc[1], desc[2]] in
    def : InstSISemantic<rtnInst,
      CmpSwapBody<scalarTy, vecTy,
                  CalculateFlatAddr<(GetNamedOperand $vaddr),
                                     (GetNamedOperand $offset)>.ret, 1>.body>;
  // SADDR NoRet
  def : InstSISemantic<desc[3],
    CmpSwapBody<scalarTy, vecTy,
                CalculateFlatSAddr<(GetNamedOperand $saddr),
                                    (GetNamedOperand $vaddr),
                                    (GetNamedOperand $offset)>.ret, 0>.body>;
  // SADDR RTN
  def : InstSISemantic<desc[4],
    CmpSwapBody<scalarTy, vecTy,
                CalculateFlatSAddr<(GetNamedOperand $saddr),
                                    (GetNamedOperand $vaddr),
                                    (GetNamedOperand $offset)>.ret, 1>.body>;
}

// GLOBAL CMPSWAP / CMPSWAP_X2 — 6 forms each (incl SADDR_RTN_agpr).
foreach desc = [
  [GLOBAL_ATOMIC_CMPSWAP, GLOBAL_ATOMIC_CMPSWAP_RTN, GLOBAL_ATOMIC_CMPSWAP_RTN_agpr,
   GLOBAL_ATOMIC_CMPSWAP_SADDR, GLOBAL_ATOMIC_CMPSWAP_SADDR_RTN,
   GLOBAL_ATOMIC_CMPSWAP_SADDR_RTN_agpr],
  [GLOBAL_ATOMIC_CMPSWAP_X2, GLOBAL_ATOMIC_CMPSWAP_X2_RTN,
   GLOBAL_ATOMIC_CMPSWAP_X2_RTN_agpr,
   GLOBAL_ATOMIC_CMPSWAP_X2_SADDR, GLOBAL_ATOMIC_CMPSWAP_X2_SADDR_RTN,
   GLOBAL_ATOMIC_CMPSWAP_X2_SADDR_RTN_agpr]
] in {
  defvar isX2 = !ne(!find(!cast<string>(desc[0]), "_X2"), -1);
  defvar scalarTy = !if(isX2, si_llvm_i64_ty, si_llvm_i32_ty);
  defvar vecTy = !if(isX2, si_llvm_v2i64_ty, si_llvm_v2i32_ty);
  def : InstSISemantic<desc[0],
    CmpSwapBody<scalarTy, vecTy,
                CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                     (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc[1], desc[2]] in
    def : InstSISemantic<rtnInst,
      CmpSwapBody<scalarTy, vecTy,
                  CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                       (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc[3],
    CmpSwapBody<scalarTy, vecTy,
                CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                      (GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc[4], desc[5]] in
    def : InstSISemantic<rtnInst,
      CmpSwapBody<scalarTy, vecTy,
                  CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                        (GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret, 1>.body>;
}

//------------------------------------------------------------------------------
// COND_SUB_U32 / CSUB_U32 / CSUB (GLOBAL) — conditional subtract.
//   COND_SUB_U32 : MEM[A] = (MEM[A] >= vdata) ? (MEM[A] - vdata) : MEM[A]
//   CSUB_U32     : MEM[A] = (MEM[A] >  vdata) ? (MEM[A] - vdata) : 0
//------------------------------------------------------------------------------

class CondSubBody<dag addr, bit isCsub, bit hasOut> {
  list<dag> body = !if(hasOut,
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad si_llvm_i32_ty, (GetVal $a))),
      (LLVMStore !if(isCsub,
                     (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $tmp),
                                  (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (LLVMSub (GetVal $tmp),
                                (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (ConstantInt si_llvm_i32_ty, 0)),
                     (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp),
                                  (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (LLVMSub (GetVal $tmp),
                                (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (GetVal $tmp))),
                  (GetVal $a)),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ],
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad si_llvm_i32_ty, (GetVal $a))),
      (LLVMStore !if(isCsub,
                     (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $tmp),
                                  (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (LLVMSub (GetVal $tmp),
                                (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (ConstantInt si_llvm_i32_ty, 0)),
                     (LLVMSelect (LLVMCmp LLVMICmpUGE, (GetVal $tmp),
                                  (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (LLVMSub (GetVal $tmp),
                                (GetNamedOperand $vdata, si_llvm_i32_ty)),
                       (GetVal $tmp))),
                  (GetVal $a))
    ]);
}

class CondSubDesc<bit isCsub, InstSI noret, InstSI rtn, InstSI rtnAgpr,
                   InstSI sad, InstSI sadRtn, InstSI sadRtnAgpr> {
  bit IsCsub = isCsub;
  InstSI NoRet = noret;
  InstSI Rtn = rtn;
  InstSI RtnAgpr = rtnAgpr;
  InstSI Sad = sad;
  InstSI SadRtn = sadRtn;
  InstSI SadRtnAgpr = sadRtnAgpr;  // ? for FLAT (which lacks this form)
}

// FLAT COND_SUB_U32 + CSUB_U32 — 5 forms each (no SADDR_RTN_agpr).
foreach desc = [
  CondSubDesc<0, FLAT_ATOMIC_COND_SUB_U32, FLAT_ATOMIC_COND_SUB_U32_RTN,
              FLAT_ATOMIC_COND_SUB_U32_RTN_agpr,
              FLAT_ATOMIC_COND_SUB_U32_SADDR, FLAT_ATOMIC_COND_SUB_U32_SADDR_RTN, ?>,
  CondSubDesc<1, FLAT_ATOMIC_CSUB_U32, FLAT_ATOMIC_CSUB_U32_RTN,
              FLAT_ATOMIC_CSUB_U32_RTN_agpr,
              FLAT_ATOMIC_CSUB_U32_SADDR, FLAT_ATOMIC_CSUB_U32_SADDR_RTN, ?>
] in {
  def : InstSISemantic<desc.NoRet, CondSubBody<
    CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret,
    desc.IsCsub, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, CondSubBody<
      CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret,
      desc.IsCsub, 1>.body>;
  def : InstSISemantic<desc.Sad, CondSubBody<
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret,
    desc.IsCsub, 0>.body>;
  def : InstSISemantic<desc.SadRtn, CondSubBody<
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret,
    desc.IsCsub, 1>.body>;
}

// GLOBAL COND_SUB_U32 + CSUB — 6 forms each (incl SADDR_RTN_agpr).
foreach desc = [
  CondSubDesc<0, GLOBAL_ATOMIC_COND_SUB_U32, GLOBAL_ATOMIC_COND_SUB_U32_RTN,
              GLOBAL_ATOMIC_COND_SUB_U32_RTN_agpr,
              GLOBAL_ATOMIC_COND_SUB_U32_SADDR, GLOBAL_ATOMIC_COND_SUB_U32_SADDR_RTN,
              GLOBAL_ATOMIC_COND_SUB_U32_SADDR_RTN_agpr>,
  CondSubDesc<1, GLOBAL_ATOMIC_CSUB, GLOBAL_ATOMIC_CSUB_RTN,
              GLOBAL_ATOMIC_CSUB_RTN_agpr,
              GLOBAL_ATOMIC_CSUB_SADDR, GLOBAL_ATOMIC_CSUB_SADDR_RTN,
              GLOBAL_ATOMIC_CSUB_SADDR_RTN_agpr>
] in {
  def : InstSISemantic<desc.NoRet, CondSubBody<
    CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret,
    desc.IsCsub, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, CondSubBody<
      CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret,
      desc.IsCsub, 1>.body>;
  def : InstSISemantic<desc.Sad, CondSubBody<
    CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret,
    desc.IsCsub, 0>.body>;
  foreach rtnInst = [desc.SadRtn, desc.SadRtnAgpr] in
    def : InstSISemantic<rtnInst, CondSubBody<
      CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                            (GetNamedOperand $offset)>.ret,
      desc.IsCsub, 1>.body>;
}

//------------------------------------------------------------------------------
// FP ADD (F32 / F64) — atomic floating-point add.
//   tmp = (fp) load (fp) MEM[A]
//   a   = bitcast vdata to fp
//   MEM[A] = (i) bitcast (fadd tmp, a) to (i)
//   RTN: vdst = bitcast tmp to (i)
//------------------------------------------------------------------------------

class FpAddBody<SILLVMType iTy, SILLVMType fpTy, dag addr, bit hasOut> {
  list<dag> body = !if(hasOut,
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad fpTy, (GetVal $a))),
      (DefVal $rhs, (LLVMBitCast (GetNamedOperand $vdata, iTy), fpTy)),
      (LLVMStore (LLVMBitCast (LLVMFAdd (GetVal $tmp), (GetVal $rhs)), iTy),
                  (GetVal $a)),
      (SetNamedOperand $vdst, (LLVMBitCast (GetVal $tmp), iTy))
    ],
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad fpTy, (GetVal $a))),
      (DefVal $rhs, (LLVMBitCast (GetNamedOperand $vdata, iTy), fpTy)),
      (LLVMStore (LLVMBitCast (LLVMFAdd (GetVal $tmp), (GetVal $rhs)), iTy),
                  (GetVal $a))
    ]);
}

class FpAddDesc<SILLVMType iTy, SILLVMType fpTy,
                 InstSI noret, InstSI rtn, InstSI rtnAgpr,
                 InstSI sad, InstSI sadRtn, InstSI sadRtnAgpr> {
  SILLVMType ITy = iTy;
  SILLVMType FpTy = fpTy;
  InstSI NoRet = noret;
  InstSI Rtn = rtn;
  InstSI RtnAgpr = rtnAgpr;
  InstSI Sad = sad;
  InstSI SadRtn = sadRtn;
  InstSI SadRtnAgpr = sadRtnAgpr;  // ? for FLAT
}

// FLAT FP-ADD F32/F64 — 5 forms each.
foreach desc = [
  FpAddDesc<si_llvm_i32_ty, si_llvm_float_ty,
            FLAT_ATOMIC_ADD_F32, FLAT_ATOMIC_ADD_F32_RTN, FLAT_ATOMIC_ADD_F32_RTN_agpr,
            FLAT_ATOMIC_ADD_F32_SADDR, FLAT_ATOMIC_ADD_F32_SADDR_RTN, ?>,
  FpAddDesc<si_llvm_i64_ty, si_llvm_double_ty,
            FLAT_ATOMIC_ADD_F64, FLAT_ATOMIC_ADD_F64_RTN, FLAT_ATOMIC_ADD_F64_RTN_agpr,
            FLAT_ATOMIC_ADD_F64_SADDR, FLAT_ATOMIC_ADD_F64_SADDR_RTN, ?>
] in {
  def : InstSISemantic<desc.NoRet, FpAddBody<desc.ITy, desc.FpTy,
    CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, FpAddBody<desc.ITy, desc.FpTy,
      CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, FpAddBody<desc.ITy, desc.FpTy,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 0>.body>;
  def : InstSISemantic<desc.SadRtn, FpAddBody<desc.ITy, desc.FpTy,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 1>.body>;
}

// GLOBAL FP-ADD F32/F64 — 6 forms each.
foreach desc = [
  FpAddDesc<si_llvm_i32_ty, si_llvm_float_ty,
            GLOBAL_ATOMIC_ADD_F32, GLOBAL_ATOMIC_ADD_F32_RTN, GLOBAL_ATOMIC_ADD_F32_RTN_agpr,
            GLOBAL_ATOMIC_ADD_F32_SADDR, GLOBAL_ATOMIC_ADD_F32_SADDR_RTN,
            GLOBAL_ATOMIC_ADD_F32_SADDR_RTN_agpr>,
  FpAddDesc<si_llvm_i64_ty, si_llvm_double_ty,
            GLOBAL_ATOMIC_ADD_F64, GLOBAL_ATOMIC_ADD_F64_RTN, GLOBAL_ATOMIC_ADD_F64_RTN_agpr,
            GLOBAL_ATOMIC_ADD_F64_SADDR, GLOBAL_ATOMIC_ADD_F64_SADDR_RTN,
            GLOBAL_ATOMIC_ADD_F64_SADDR_RTN_agpr>
] in {
  def : InstSISemantic<desc.NoRet, FpAddBody<desc.ITy, desc.FpTy,
    CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, FpAddBody<desc.ITy, desc.FpTy,
      CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, FpAddBody<desc.ITy, desc.FpTy,
    CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.SadRtn, desc.SadRtnAgpr] in
    def : InstSISemantic<rtnInst, FpAddBody<desc.ITy, desc.FpTy,
      CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                            (GetNamedOperand $offset)>.ret, 1>.body>;
}

//------------------------------------------------------------------------------
// FP MIN / MAX (FMIN, FMAX = F32; MIN_F64, MAX_F64) — atomic compare-select.
// Per AMDGPU ISA: result = (vdata cmp tmp) ? vdata : tmp  (no NaN handling;
// this matches what the HW does — fmin/fmax are FCMP-OLT/OGT then select,
// not LLVM's minnum/maxnum).
//------------------------------------------------------------------------------

class FpMinMaxBody<SILLVMType iTy, SILLVMType fpTy, LLVMCmpPredicate pred,
                    dag addr, bit hasOut> {
  list<dag> body = !if(hasOut,
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad fpTy, (GetVal $a))),
      (DefVal $rhs, (LLVMBitCast (GetNamedOperand $vdata, iTy), fpTy)),
      (LLVMStore (LLVMBitCast
                   (LLVMSelect (LLVMCmp pred, (GetVal $rhs), (GetVal $tmp)),
                                (GetVal $rhs), (GetVal $tmp)),
                   iTy),
                  (GetVal $a)),
      (SetNamedOperand $vdst, (LLVMBitCast (GetVal $tmp), iTy))
    ],
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad fpTy, (GetVal $a))),
      (DefVal $rhs, (LLVMBitCast (GetNamedOperand $vdata, iTy), fpTy)),
      (LLVMStore (LLVMBitCast
                   (LLVMSelect (LLVMCmp pred, (GetVal $rhs), (GetVal $tmp)),
                                (GetVal $rhs), (GetVal $tmp)),
                   iTy),
                  (GetVal $a))
    ]);
}

class FpMinMaxDesc<SILLVMType iTy, SILLVMType fpTy, LLVMCmpPredicate pred,
                    InstSI noret, InstSI rtn, InstSI rtnAgpr,
                    InstSI sad, InstSI sadRtn, InstSI sadRtnAgpr> {
  SILLVMType ITy = iTy;
  SILLVMType FpTy = fpTy;
  LLVMCmpPredicate Pred = pred;
  InstSI NoRet = noret;
  InstSI Rtn = rtn;
  InstSI RtnAgpr = rtnAgpr;
  InstSI Sad = sad;
  InstSI SadRtn = sadRtn;
  InstSI SadRtnAgpr = sadRtnAgpr;  // ? for FLAT
}

// FLAT FMIN/FMAX (F32) + MIN_F64/MAX_F64 — 5 forms each.
foreach desc = [
  FpMinMaxDesc<si_llvm_i32_ty, si_llvm_float_ty,  LLVMFCmpOLT,
               FLAT_ATOMIC_FMIN, FLAT_ATOMIC_FMIN_RTN, FLAT_ATOMIC_FMIN_RTN_agpr,
               FLAT_ATOMIC_FMIN_SADDR, FLAT_ATOMIC_FMIN_SADDR_RTN, ?>,
  FpMinMaxDesc<si_llvm_i32_ty, si_llvm_float_ty,  LLVMFCmpOGT,
               FLAT_ATOMIC_FMAX, FLAT_ATOMIC_FMAX_RTN, FLAT_ATOMIC_FMAX_RTN_agpr,
               FLAT_ATOMIC_FMAX_SADDR, FLAT_ATOMIC_FMAX_SADDR_RTN, ?>,
  FpMinMaxDesc<si_llvm_i64_ty, si_llvm_double_ty, LLVMFCmpOLT,
               FLAT_ATOMIC_MIN_F64, FLAT_ATOMIC_MIN_F64_RTN, FLAT_ATOMIC_MIN_F64_RTN_agpr,
               FLAT_ATOMIC_MIN_F64_SADDR, FLAT_ATOMIC_MIN_F64_SADDR_RTN, ?>,
  FpMinMaxDesc<si_llvm_i64_ty, si_llvm_double_ty, LLVMFCmpOGT,
               FLAT_ATOMIC_MAX_F64, FLAT_ATOMIC_MAX_F64_RTN, FLAT_ATOMIC_MAX_F64_RTN_agpr,
               FLAT_ATOMIC_MAX_F64_SADDR, FLAT_ATOMIC_MAX_F64_SADDR_RTN, ?>
] in {
  def : InstSISemantic<desc.NoRet, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
    CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
      CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 0>.body>;
  def : InstSISemantic<desc.SadRtn, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 1>.body>;
}

// GLOBAL FMIN/FMAX + MIN_F64/MAX_F64 — 6 forms each.
foreach desc = [
  FpMinMaxDesc<si_llvm_i32_ty, si_llvm_float_ty,  LLVMFCmpOLT,
               GLOBAL_ATOMIC_FMIN, GLOBAL_ATOMIC_FMIN_RTN, GLOBAL_ATOMIC_FMIN_RTN_agpr,
               GLOBAL_ATOMIC_FMIN_SADDR, GLOBAL_ATOMIC_FMIN_SADDR_RTN,
               GLOBAL_ATOMIC_FMIN_SADDR_RTN_agpr>,
  FpMinMaxDesc<si_llvm_i32_ty, si_llvm_float_ty,  LLVMFCmpOGT,
               GLOBAL_ATOMIC_FMAX, GLOBAL_ATOMIC_FMAX_RTN, GLOBAL_ATOMIC_FMAX_RTN_agpr,
               GLOBAL_ATOMIC_FMAX_SADDR, GLOBAL_ATOMIC_FMAX_SADDR_RTN,
               GLOBAL_ATOMIC_FMAX_SADDR_RTN_agpr>,
  FpMinMaxDesc<si_llvm_i64_ty, si_llvm_double_ty, LLVMFCmpOLT,
               GLOBAL_ATOMIC_MIN_F64, GLOBAL_ATOMIC_MIN_F64_RTN, GLOBAL_ATOMIC_MIN_F64_RTN_agpr,
               GLOBAL_ATOMIC_MIN_F64_SADDR, GLOBAL_ATOMIC_MIN_F64_SADDR_RTN,
               GLOBAL_ATOMIC_MIN_F64_SADDR_RTN_agpr>,
  FpMinMaxDesc<si_llvm_i64_ty, si_llvm_double_ty, LLVMFCmpOGT,
               GLOBAL_ATOMIC_MAX_F64, GLOBAL_ATOMIC_MAX_F64_RTN, GLOBAL_ATOMIC_MAX_F64_RTN_agpr,
               GLOBAL_ATOMIC_MAX_F64_SADDR, GLOBAL_ATOMIC_MAX_F64_SADDR_RTN,
               GLOBAL_ATOMIC_MAX_F64_SADDR_RTN_agpr>
] in {
  def : InstSISemantic<desc.NoRet, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
    CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
      CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
    CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.SadRtn, desc.SadRtnAgpr] in
    def : InstSISemantic<rtnInst, FpMinMaxBody<desc.ITy, desc.FpTy, desc.Pred,
      CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                            (GetNamedOperand $offset)>.ret, 1>.body>;
}

//------------------------------------------------------------------------------
// FCMPSWAP / FCMPSWAP_X2 — float compare-and-swap (FCmp OEQ).
//   $vdata is a v2-typed register: element 0 = src (new fp value),
//                                  element 1 = cmp (compare fp value).
//   if MEM[A] (fcmp oeq) cmp: MEM[A] = src
//------------------------------------------------------------------------------

class FCmpSwapBody<SILLVMType iTy, SILLVMType fpTy, SILLVMType iVecTy,
                    dag addr, bit hasOut> {
  list<dag> body = !if(hasOut,
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad fpTy, (GetVal $a))),
      (DefVal $cmpI, (LLVMExtractElement (GetNamedOperand $vdata, iVecTy),
                                          (ConstantInt si_llvm_i32_ty, 1))),
      (DefVal $srcI, (LLVMExtractElement (GetNamedOperand $vdata, iVecTy),
                                          (ConstantInt si_llvm_i32_ty, 0))),
      (DefVal $cmp, (LLVMBitCast (GetVal $cmpI), fpTy)),
      (DefVal $src, (LLVMBitCast (GetVal $srcI), fpTy)),
      (LLVMStore (LLVMBitCast
                   (LLVMSelect (LLVMCmp LLVMFCmpOEQ, (GetVal $tmp), (GetVal $cmp)),
                                (GetVal $src), (GetVal $tmp)),
                   iTy),
                  (GetVal $a)),
      (SetNamedOperand $vdst, (LLVMBitCast (GetVal $tmp), iTy))
    ],
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad fpTy, (GetVal $a))),
      (DefVal $cmpI, (LLVMExtractElement (GetNamedOperand $vdata, iVecTy),
                                          (ConstantInt si_llvm_i32_ty, 1))),
      (DefVal $srcI, (LLVMExtractElement (GetNamedOperand $vdata, iVecTy),
                                          (ConstantInt si_llvm_i32_ty, 0))),
      (DefVal $cmp, (LLVMBitCast (GetVal $cmpI), fpTy)),
      (DefVal $src, (LLVMBitCast (GetVal $srcI), fpTy)),
      (LLVMStore (LLVMBitCast
                   (LLVMSelect (LLVMCmp LLVMFCmpOEQ, (GetVal $tmp), (GetVal $cmp)),
                                (GetVal $src), (GetVal $tmp)),
                   iTy),
                  (GetVal $a))
    ]);
}

class FCmpSwapDesc<SILLVMType iTy, SILLVMType fpTy, SILLVMType iVecTy,
                    InstSI noret, InstSI rtn, InstSI rtnAgpr,
                    InstSI sad, InstSI sadRtn, InstSI sadRtnAgpr> {
  SILLVMType ITy = iTy;
  SILLVMType FpTy = fpTy;
  SILLVMType IVecTy = iVecTy;
  InstSI NoRet = noret;
  InstSI Rtn = rtn;
  InstSI RtnAgpr = rtnAgpr;
  InstSI Sad = sad;
  InstSI SadRtn = sadRtn;
  InstSI SadRtnAgpr = sadRtnAgpr;  // ? for FLAT
}

// FLAT FCMPSWAP / FCMPSWAP_X2 — 5 forms each.
foreach desc = [
  FCmpSwapDesc<si_llvm_i32_ty, si_llvm_float_ty,  si_llvm_v2i32_ty,
               FLAT_ATOMIC_FCMPSWAP, FLAT_ATOMIC_FCMPSWAP_RTN, FLAT_ATOMIC_FCMPSWAP_RTN_agpr,
               FLAT_ATOMIC_FCMPSWAP_SADDR, FLAT_ATOMIC_FCMPSWAP_SADDR_RTN, ?>,
  FCmpSwapDesc<si_llvm_i64_ty, si_llvm_double_ty, si_llvm_v2i64_ty,
               FLAT_ATOMIC_FCMPSWAP_X2, FLAT_ATOMIC_FCMPSWAP_X2_RTN, FLAT_ATOMIC_FCMPSWAP_X2_RTN_agpr,
               FLAT_ATOMIC_FCMPSWAP_X2_SADDR, FLAT_ATOMIC_FCMPSWAP_X2_SADDR_RTN, ?>
] in {
  def : InstSISemantic<desc.NoRet, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
    CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
      CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 0>.body>;
  def : InstSISemantic<desc.SadRtn, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 1>.body>;
}

// GLOBAL FCMPSWAP / FCMPSWAP_X2 — 6 forms each.
foreach desc = [
  FCmpSwapDesc<si_llvm_i32_ty, si_llvm_float_ty,  si_llvm_v2i32_ty,
               GLOBAL_ATOMIC_FCMPSWAP, GLOBAL_ATOMIC_FCMPSWAP_RTN, GLOBAL_ATOMIC_FCMPSWAP_RTN_agpr,
               GLOBAL_ATOMIC_FCMPSWAP_SADDR, GLOBAL_ATOMIC_FCMPSWAP_SADDR_RTN,
               GLOBAL_ATOMIC_FCMPSWAP_SADDR_RTN_agpr>,
  FCmpSwapDesc<si_llvm_i64_ty, si_llvm_double_ty, si_llvm_v2i64_ty,
               GLOBAL_ATOMIC_FCMPSWAP_X2, GLOBAL_ATOMIC_FCMPSWAP_X2_RTN, GLOBAL_ATOMIC_FCMPSWAP_X2_RTN_agpr,
               GLOBAL_ATOMIC_FCMPSWAP_X2_SADDR, GLOBAL_ATOMIC_FCMPSWAP_X2_SADDR_RTN,
               GLOBAL_ATOMIC_FCMPSWAP_X2_SADDR_RTN_agpr>
] in {
  def : InstSISemantic<desc.NoRet, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
    CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
      CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
    CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.SadRtn, desc.SadRtnAgpr] in
    def : InstSISemantic<rtnInst, FCmpSwapBody<desc.ITy, desc.FpTy, desc.IVecTy,
      CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                            (GetNamedOperand $offset)>.ret, 1>.body>;
}

//------------------------------------------------------------------------------
// PK_ADD (F16 / BF16) — packed 2 × 16-bit float atomic add.
//   $vdata is an i32 holding two packed FP16 (or BF16) elements.
//   tmp = (i32) load (i32) MEM[A]
//   result = bitcast (fadd <2 x fp16> bitcast(tmp), bitcast(vdata)) to i32
//   MEM[A] = result
//   RTN: vdst = tmp
//------------------------------------------------------------------------------

class PkAddBody<SILLVMType v2fpTy, dag addr, bit hasOut> {
  list<dag> body = !if(hasOut,
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad si_llvm_i32_ty, (GetVal $a))),
      (DefVal $lhsFp, (LLVMBitCast (GetVal $tmp), v2fpTy)),
      (DefVal $rhsFp, (LLVMBitCast (GetNamedOperand $vdata, si_llvm_i32_ty),
                                     v2fpTy)),
      (LLVMStore (LLVMBitCast (LLVMFAdd (GetVal $lhsFp), (GetVal $rhsFp)),
                                si_llvm_i32_ty),
                  (GetVal $a)),
      (SetNamedOperand $vdst, (GetVal $tmp))
    ],
    [
      (DefVal $a, addr),
      (DefVal $tmp, (LLVMLoad si_llvm_i32_ty, (GetVal $a))),
      (DefVal $lhsFp, (LLVMBitCast (GetVal $tmp), v2fpTy)),
      (DefVal $rhsFp, (LLVMBitCast (GetNamedOperand $vdata, si_llvm_i32_ty),
                                     v2fpTy)),
      (LLVMStore (LLVMBitCast (LLVMFAdd (GetVal $lhsFp), (GetVal $rhsFp)),
                                si_llvm_i32_ty),
                  (GetVal $a))
    ]);
}

class PkAddDesc<SILLVMType v2fpTy,
                 InstSI noret, InstSI rtn, InstSI rtnAgpr,
                 InstSI sad, InstSI sadRtn, InstSI sadRtnAgpr> {
  SILLVMType V2FpTy = v2fpTy;
  InstSI NoRet = noret;
  InstSI Rtn = rtn;
  InstSI RtnAgpr = rtnAgpr;
  InstSI Sad = sad;
  InstSI SadRtn = sadRtn;
  InstSI SadRtnAgpr = sadRtnAgpr;  // ? for FLAT
}

// FLAT PK_ADD F16 / BF16 — 5 forms each.
foreach desc = [
  PkAddDesc<si_llvm_v2f16_ty,
            FLAT_ATOMIC_PK_ADD_F16, FLAT_ATOMIC_PK_ADD_F16_RTN, FLAT_ATOMIC_PK_ADD_F16_RTN_agpr,
            FLAT_ATOMIC_PK_ADD_F16_SADDR, FLAT_ATOMIC_PK_ADD_F16_SADDR_RTN, ?>,
  PkAddDesc<si_llvm_v2bf16_ty,
            FLAT_ATOMIC_PK_ADD_BF16, FLAT_ATOMIC_PK_ADD_BF16_RTN, FLAT_ATOMIC_PK_ADD_BF16_RTN_agpr,
            FLAT_ATOMIC_PK_ADD_BF16_SADDR, FLAT_ATOMIC_PK_ADD_BF16_SADDR_RTN, ?>
] in {
  def : InstSISemantic<desc.NoRet, PkAddBody<desc.V2FpTy,
    CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, PkAddBody<desc.V2FpTy,
      CalculateFlatAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, PkAddBody<desc.V2FpTy,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 0>.body>;
  def : InstSISemantic<desc.SadRtn, PkAddBody<desc.V2FpTy,
    CalculateFlatSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                        (GetNamedOperand $offset)>.ret, 1>.body>;
}

// GLOBAL PK_ADD F16 / BF16 — 6 forms each.
foreach desc = [
  PkAddDesc<si_llvm_v2f16_ty,
            GLOBAL_ATOMIC_PK_ADD_F16, GLOBAL_ATOMIC_PK_ADD_F16_RTN, GLOBAL_ATOMIC_PK_ADD_F16_RTN_agpr,
            GLOBAL_ATOMIC_PK_ADD_F16_SADDR, GLOBAL_ATOMIC_PK_ADD_F16_SADDR_RTN,
            GLOBAL_ATOMIC_PK_ADD_F16_SADDR_RTN_agpr>,
  PkAddDesc<si_llvm_v2bf16_ty,
            GLOBAL_ATOMIC_PK_ADD_BF16, GLOBAL_ATOMIC_PK_ADD_BF16_RTN, GLOBAL_ATOMIC_PK_ADD_BF16_RTN_agpr,
            GLOBAL_ATOMIC_PK_ADD_BF16_SADDR, GLOBAL_ATOMIC_PK_ADD_BF16_SADDR_RTN,
            GLOBAL_ATOMIC_PK_ADD_BF16_SADDR_RTN_agpr>
] in {
  def : InstSISemantic<desc.NoRet, PkAddBody<desc.V2FpTy,
    CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.Rtn, desc.RtnAgpr] in
    def : InstSISemantic<rtnInst, PkAddBody<desc.V2FpTy,
      CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret, 1>.body>;
  def : InstSISemantic<desc.Sad, PkAddBody<desc.V2FpTy,
    CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret, 0>.body>;
  foreach rtnInst = [desc.SadRtn, desc.SadRtnAgpr] in
    def : InstSISemantic<rtnInst, PkAddBody<desc.V2FpTy,
      CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                            (GetNamedOperand $offset)>.ret, 1>.body>;
}

//==============================================================================
// SCRATCH address-calc helpers (4 modes: base, SADDR, ST, SVS).
//
// Per gfx9+ ISA:
//   base (no SADDR, no ST): addr = inttoptr(vaddr) + offset
//   _SADDR : uniform-base form; addr = inttoptr(saddr) + offset
//   _ST    : stack-only; no $vaddr; addr = inttoptr(offset)
//   _SVS   : saddr + voffset; addr = inttoptr(saddr + zext(vaddr)) + offset
//
// All scratch pointers are addrspace(5); the per-thread base is implicit in
// LLVM's addrspace-5 lowering, so we don't model it here.
//==============================================================================

class CalculateScratchSAddrAddr<dag saddr, dag offset> {
  dag ret = (LLVMPtrAdd (LLVMIntToPtr saddr, scratch_ptr_ty), offset);
}

class CalculateScratchSVSAddr<dag saddr, dag vaddr, dag offset> {
  // SCRATCH _SVS mode: saddr is SReg_32 (32-bit) and vaddr is VGPR_32. Both
  // 32 bits — no zero-extension needed. (Pre-refactor records had a
  // zext-to-i64 here that would have crashed CreateAdd on type mismatch
  // had any test exercised SVS.)
  dag ret = (LLVMPtrAdd
    (LLVMIntToPtr (LLVMAdd saddr, vaddr), scratch_ptr_ty),
    offset);
}

class CalculateScratchSTAddr<dag offset> {
  dag ret = (LLVMIntToPtr offset, scratch_ptr_ty);
}

//------------------------------------------------------------------------------
// SCRATCH Loads — 4 address forms × widths. Sub-DWORD widths ext to i32.
//------------------------------------------------------------------------------

class ScratchLoadDesc<InstSI base, InstSI sad, InstSI st, InstSI svs,
                      SILLVMType memTy, bit isSigned, bit needsExt> {
  InstSI Base = base;
  InstSI Sad  = sad;
  InstSI ST   = st;
  InstSI SVS  = svs;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
  bit NeedsExt = needsExt;
}

class ScratchLoadBody<SILLVMType memTy, bit isSigned, bit needsExt, dag addr> {
  list<dag> body = [
    (DefVal $a, addr),
    (DefVal $val, (LLVMLoad memTy, (GetVal $a))),
    (SetNamedOperand $vdst,
       !if(needsExt,
           !if(isSigned,
               (LLVMSExt (GetVal $val), si_llvm_i32_ty),
               (LLVMZExt (GetVal $val), si_llvm_i32_ty)),
           (GetVal $val)))
  ];
}

foreach desc = [
  ScratchLoadDesc<SCRATCH_LOAD_UBYTE,  SCRATCH_LOAD_UBYTE_SADDR,
                  SCRATCH_LOAD_UBYTE_ST, SCRATCH_LOAD_UBYTE_SVS,
                  si_llvm_i8_ty,   0, 1>,
  ScratchLoadDesc<SCRATCH_LOAD_SBYTE,  SCRATCH_LOAD_SBYTE_SADDR,
                  SCRATCH_LOAD_SBYTE_ST, SCRATCH_LOAD_SBYTE_SVS,
                  si_llvm_i8_ty,   1, 1>,
  ScratchLoadDesc<SCRATCH_LOAD_USHORT, SCRATCH_LOAD_USHORT_SADDR,
                  SCRATCH_LOAD_USHORT_ST, SCRATCH_LOAD_USHORT_SVS,
                  si_llvm_i16_ty,  0, 1>,
  ScratchLoadDesc<SCRATCH_LOAD_SSHORT, SCRATCH_LOAD_SSHORT_SADDR,
                  SCRATCH_LOAD_SSHORT_ST, SCRATCH_LOAD_SSHORT_SVS,
                  si_llvm_i16_ty,  1, 1>,
  ScratchLoadDesc<SCRATCH_LOAD_DWORD,   SCRATCH_LOAD_DWORD_SADDR,
                  SCRATCH_LOAD_DWORD_ST, SCRATCH_LOAD_DWORD_SVS,
                  si_llvm_i32_ty,   0, 0>,
  ScratchLoadDesc<SCRATCH_LOAD_DWORDX2, SCRATCH_LOAD_DWORDX2_SADDR,
                  SCRATCH_LOAD_DWORDX2_ST, SCRATCH_LOAD_DWORDX2_SVS,
                  si_llvm_v2i32_ty, 0, 0>,
  ScratchLoadDesc<SCRATCH_LOAD_DWORDX3, SCRATCH_LOAD_DWORDX3_SADDR,
                  SCRATCH_LOAD_DWORDX3_ST, SCRATCH_LOAD_DWORDX3_SVS,
                  si_llvm_v3i32_ty, 0, 0>,
  ScratchLoadDesc<SCRATCH_LOAD_DWORDX4, SCRATCH_LOAD_DWORDX4_SADDR,
                  SCRATCH_LOAD_DWORDX4_ST, SCRATCH_LOAD_DWORDX4_SVS,
                  si_llvm_v4i32_ty, 0, 0>
] in {
  def : InstSISemantic<desc.Base,
    ScratchLoadBody<desc.MemTy, desc.IsSigned, desc.NeedsExt,
      CalculateScratchAddr<(GetNamedOperand $vaddr),
                            (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.Sad,
    ScratchLoadBody<desc.MemTy, desc.IsSigned, desc.NeedsExt,
      CalculateScratchSAddrAddr<(GetNamedOperand $saddr),
                                 (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.ST,
    ScratchLoadBody<desc.MemTy, desc.IsSigned, desc.NeedsExt,
      CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.SVS,
    ScratchLoadBody<desc.MemTy, desc.IsSigned, desc.NeedsExt,
      CalculateScratchSVSAddr<(GetNamedOperand $saddr),
                               (GetNamedOperand $vaddr),
                               (GetNamedOperand $offset)>.ret>.body>;
}

//------------------------------------------------------------------------------
// SCRATCH Stores — 4 address forms × widths. Sub-DWORD widths trunc.
//------------------------------------------------------------------------------

class ScratchStoreDesc<InstSI base, InstSI sad, InstSI st, InstSI svs,
                       SILLVMType storeTy, bit needsTrunc> {
  InstSI Base = base;
  InstSI Sad  = sad;
  InstSI ST   = st;
  InstSI SVS  = svs;
  SILLVMType StoreTy = storeTy;
  bit NeedsTrunc = needsTrunc;
}

class ScratchStoreBody<SILLVMType storeTy, bit needsTrunc, dag addr> {
  list<dag> body = [
    (DefVal $a, addr),
    (LLVMStore !if(needsTrunc,
                   (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i32_ty), storeTy),
                   (GetNamedOperand $vdata)),
                (GetVal $a))
  ];
}

foreach desc = [
  ScratchStoreDesc<SCRATCH_STORE_BYTE,   SCRATCH_STORE_BYTE_SADDR,
                   SCRATCH_STORE_BYTE_ST, SCRATCH_STORE_BYTE_SVS,
                   si_llvm_i8_ty,   1>,
  ScratchStoreDesc<SCRATCH_STORE_SHORT,  SCRATCH_STORE_SHORT_SADDR,
                   SCRATCH_STORE_SHORT_ST, SCRATCH_STORE_SHORT_SVS,
                   si_llvm_i16_ty,  1>,
  ScratchStoreDesc<SCRATCH_STORE_DWORD,  SCRATCH_STORE_DWORD_SADDR,
                   SCRATCH_STORE_DWORD_ST, SCRATCH_STORE_DWORD_SVS,
                   si_llvm_i32_ty,  0>,
  ScratchStoreDesc<SCRATCH_STORE_DWORDX2, SCRATCH_STORE_DWORDX2_SADDR,
                   SCRATCH_STORE_DWORDX2_ST, SCRATCH_STORE_DWORDX2_SVS,
                   si_llvm_v2i32_ty, 0>,
  ScratchStoreDesc<SCRATCH_STORE_DWORDX3, SCRATCH_STORE_DWORDX3_SADDR,
                   SCRATCH_STORE_DWORDX3_ST, SCRATCH_STORE_DWORDX3_SVS,
                   si_llvm_v3i32_ty, 0>,
  ScratchStoreDesc<SCRATCH_STORE_DWORDX4, SCRATCH_STORE_DWORDX4_SADDR,
                   SCRATCH_STORE_DWORDX4_ST, SCRATCH_STORE_DWORDX4_SVS,
                   si_llvm_v4i32_ty, 0>
] in {
  def : InstSISemantic<desc.Base,
    ScratchStoreBody<desc.StoreTy, desc.NeedsTrunc,
      CalculateScratchAddr<(GetNamedOperand $vaddr),
                            (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.Sad,
    ScratchStoreBody<desc.StoreTy, desc.NeedsTrunc,
      CalculateScratchSAddrAddr<(GetNamedOperand $saddr),
                                 (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.ST,
    ScratchStoreBody<desc.StoreTy, desc.NeedsTrunc,
      CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.SVS,
    ScratchStoreBody<desc.StoreTy, desc.NeedsTrunc,
      CalculateScratchSVSAddr<(GetNamedOperand $saddr),
                               (GetNamedOperand $vaddr),
                               (GetNamedOperand $offset)>.ret>.body>;
}

//------------------------------------------------------------------------------
// SCRATCH D16 / D16_HI Loads — 4 address forms × {SBYTE, UBYTE, SHORT}
// × {D16, D16_HI}. Same half-preserve via $vdst tied-source workaround as
// FLAT D16. See FLAT D16 comment for the upstream llvm-mc tied-operand bug.
//------------------------------------------------------------------------------

class ScratchD16LoadDesc<SILLVMType memTy, bit isSigned,
                          InstSI lo, InstSI loSad, InstSI loSt, InstSI loSvs,
                          InstSI hi, InstSI hiSad, InstSI hiSt, InstSI hiSvs> {
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
  InstSI Lo = lo;
  InstSI LoSad = loSad;
  InstSI LoSt = loSt;
  InstSI LoSvs = loSvs;
  InstSI Hi = hi;
  InstSI HiSad = hiSad;
  InstSI HiSt = hiSt;
  InstSI HiSvs = hiSvs;
}

class ScratchD16LoadBody<SILLVMType memTy, bit isSigned, bit isHi, dag addr> {
  list<dag> body = [
    (DefVal $a, addr),
    (DefVal $raw, (LLVMLoad memTy, (GetVal $a))),
    (DefVal $half, !if(!eq(memTy.ElemSize, 16),
                        (GetVal $raw),
                        !if(isSigned,
                            (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
                            (LLVMZExt (GetVal $raw), si_llvm_i16_ty)))),
    (DefVal $half32, (LLVMZExt (GetVal $half), si_llvm_i32_ty)),
    !if(isHi,
        (DefVal $shifted, (LLVMShl (GetVal $half32),
                                    (ConstantInt si_llvm_i32_ty, 16))),
        (DefVal $shifted, (GetVal $half32))),
    (DefVal $other, (LLVMAnd (GetNamedOperand $vdst_in, si_llvm_i32_ty),
                              (ConstantInt si_llvm_i32_ty,
                                !if(isHi, 0x0000FFFF, 0xFFFF0000)))),
    (SetNamedOperand $vdst, (LLVMOr (GetVal $other), (GetVal $shifted)))
  ];
}

foreach desc = [
  ScratchD16LoadDesc<si_llvm_i8_ty,  0,
    SCRATCH_LOAD_UBYTE_D16,    SCRATCH_LOAD_UBYTE_D16_SADDR,
    SCRATCH_LOAD_UBYTE_D16_ST, SCRATCH_LOAD_UBYTE_D16_SVS,
    SCRATCH_LOAD_UBYTE_D16_HI, SCRATCH_LOAD_UBYTE_D16_HI_SADDR,
    SCRATCH_LOAD_UBYTE_D16_HI_ST, SCRATCH_LOAD_UBYTE_D16_HI_SVS>,
  ScratchD16LoadDesc<si_llvm_i8_ty,  1,
    SCRATCH_LOAD_SBYTE_D16,    SCRATCH_LOAD_SBYTE_D16_SADDR,
    SCRATCH_LOAD_SBYTE_D16_ST, SCRATCH_LOAD_SBYTE_D16_SVS,
    SCRATCH_LOAD_SBYTE_D16_HI, SCRATCH_LOAD_SBYTE_D16_HI_SADDR,
    SCRATCH_LOAD_SBYTE_D16_HI_ST, SCRATCH_LOAD_SBYTE_D16_HI_SVS>,
  ScratchD16LoadDesc<si_llvm_i16_ty, 0,
    SCRATCH_LOAD_SHORT_D16,    SCRATCH_LOAD_SHORT_D16_SADDR,
    SCRATCH_LOAD_SHORT_D16_ST, SCRATCH_LOAD_SHORT_D16_SVS,
    SCRATCH_LOAD_SHORT_D16_HI, SCRATCH_LOAD_SHORT_D16_HI_SADDR,
    SCRATCH_LOAD_SHORT_D16_HI_ST, SCRATCH_LOAD_SHORT_D16_HI_SVS>
] in {
  // Low (D16) — 4 addr forms
  def : InstSISemantic<desc.Lo,    ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 0,
    CalculateScratchAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.LoSad, ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 0,
    CalculateScratchSAddrAddr<(GetNamedOperand $saddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.LoSt,  ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 0,
    CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.LoSvs, ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 0,
    CalculateScratchSVSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                             (GetNamedOperand $offset)>.ret>.body>;
  // High (D16_HI) — 4 addr forms
  def : InstSISemantic<desc.Hi,    ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 1,
    CalculateScratchAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.HiSad, ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 1,
    CalculateScratchSAddrAddr<(GetNamedOperand $saddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.HiSt,  ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 1,
    CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.HiSvs, ScratchD16LoadBody<desc.MemTy, desc.IsSigned, 1,
    CalculateScratchSVSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                             (GetNamedOperand $offset)>.ret>.body>;
}

//------------------------------------------------------------------------------
// SCRATCH D16_HI Stores — extract upper 16 bits of $vdata + 4 addr forms.
//------------------------------------------------------------------------------

class ScratchD16HiStoreDesc<SILLVMType storeTy,
                             InstSI base, InstSI sad, InstSI st, InstSI svs> {
  SILLVMType StoreTy = storeTy;
  InstSI Base = base;
  InstSI Sad  = sad;
  InstSI ST   = st;
  InstSI SVS  = svs;
}

class ScratchD16HiStoreBody<SILLVMType storeTy, dag addr> {
  list<dag> body = [
    (DefVal $a, addr),
    (LLVMStore (LLVMTrunc (LLVMLShr (GetNamedOperand $vdata, si_llvm_i32_ty),
                                     (ConstantInt si_llvm_i32_ty, 16)),
                           storeTy),
                (GetVal $a))
  ];
}

foreach desc = [
  ScratchD16HiStoreDesc<si_llvm_i8_ty,
                         SCRATCH_STORE_BYTE_D16_HI, SCRATCH_STORE_BYTE_D16_HI_SADDR,
                         SCRATCH_STORE_BYTE_D16_HI_ST, SCRATCH_STORE_BYTE_D16_HI_SVS>,
  ScratchD16HiStoreDesc<si_llvm_i16_ty,
                         SCRATCH_STORE_SHORT_D16_HI, SCRATCH_STORE_SHORT_D16_HI_SADDR,
                         SCRATCH_STORE_SHORT_D16_HI_ST, SCRATCH_STORE_SHORT_D16_HI_SVS>
] in {
  def : InstSISemantic<desc.Base, ScratchD16HiStoreBody<desc.StoreTy,
    CalculateScratchAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.Sad, ScratchD16HiStoreBody<desc.StoreTy,
    CalculateScratchSAddrAddr<(GetNamedOperand $saddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.ST, ScratchD16HiStoreBody<desc.StoreTy,
    CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.SVS, ScratchD16HiStoreBody<desc.StoreTy,
    CalculateScratchSVSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                             (GetNamedOperand $offset)>.ret>.body>;
}

//------------------------------------------------------------------------------
// GLOBAL_LOAD_LDS_* — DRAM-to-LDS direct load, modeled via
// int_amdgcn_global_load_lds(global_ptr, local_ptr, data_byte_size,
//                             imm_offset, aux_cachepolicy).
//
// MIR operands (per FLAT_Global_Load_LDS_Pseudo IsAsync=0):
//   (outs) (ins $vaddr [SADDR: + $saddr], $offset, $cpol)
// The LDS pointer is held in $vdst (named so by the pseudo but used as input).
// We pass `imm_offset=0` since the byte offset is folded into the global ptr.
//
// SCRATCH_LOAD_LDS_* (addrspace 5) is NOT modeled — `int_amdgcn_load_to_lds`
// doesn't yet support addrspace(5) per its docstring TODO. Those records
// fall through to MIInlineAsmEmitter.
//------------------------------------------------------------------------------

class GlobalLoadLdsDesc<InstSI base, InstSI sad, int dataBytes> {
  InstSI Base = base;
  InstSI Sad  = sad;
  int DataBytes = dataBytes;
}

class GlobalLoadLdsBody<int dataBytes, dag gAddr> {
  list<dag> body = [
    (DefVal $g, gAddr),
    // Non-async GLOBAL_LOAD_LDS_* has no $vdst — the LDS destination comes
    // from the M0 register (implicit-use). Use M0 truncated to the
    // 16-bit LDS offset field.
    (DefVal $l, (LLVMIntToPtr (ImplicitUse M0), local_ptr_ty)),
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_global_load_lds,
                   [(GetVal $g), (GetVal $l),
                    (ConstantInt si_llvm_i32_ty, dataBytes),
                    (ConstantInt si_llvm_i32_ty, 0),
                    (GetNamedOperand $cpol, si_llvm_i32_ty)])
  ];
}

foreach desc = [
  GlobalLoadLdsDesc<GLOBAL_LOAD_LDS_UBYTE,   GLOBAL_LOAD_LDS_UBYTE_SADDR,   1>,
  GlobalLoadLdsDesc<GLOBAL_LOAD_LDS_SBYTE,   GLOBAL_LOAD_LDS_SBYTE_SADDR,   1>,
  GlobalLoadLdsDesc<GLOBAL_LOAD_LDS_USHORT,  GLOBAL_LOAD_LDS_USHORT_SADDR,  2>,
  GlobalLoadLdsDesc<GLOBAL_LOAD_LDS_SSHORT,  GLOBAL_LOAD_LDS_SSHORT_SADDR,  2>,
  GlobalLoadLdsDesc<GLOBAL_LOAD_LDS_DWORD,   GLOBAL_LOAD_LDS_DWORD_SADDR,   4>,
  GlobalLoadLdsDesc<GLOBAL_LOAD_LDS_DWORDX3, GLOBAL_LOAD_LDS_DWORDX3_SADDR, 12>,
  GlobalLoadLdsDesc<GLOBAL_LOAD_LDS_DWORDX4, GLOBAL_LOAD_LDS_DWORDX4_SADDR, 16>
] in {
  def : InstSISemantic<desc.Base, GlobalLoadLdsBody<desc.DataBytes,
    CalculateGlobalAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.Sad, GlobalLoadLdsBody<desc.DataBytes,
    CalculateGlobalSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret>.body>;
}

//------------------------------------------------------------------------------
// GLOBAL_LOAD_TR_B{64,128}_w{32,64} (+ _SADDR) — WMMA transpose loads.
// Each MIR pseudo maps to a wave-size-specific return type:
//   _w32  reads two 64-bit values per wave32 (returns v2i32 for B64, v4i32 for B128).
//   _w64  reads one 64-bit value per wave64 (returns i32  for B64, v2i32 for B128).
// Intrinsic: int_amdgcn_global_load_tr_b{64,128} — overloaded on return type;
// the LLVMIntrinsic primitive with explicit retTy resolves it correctly.
//------------------------------------------------------------------------------

class GlobalLoadTRDesc<InstSI base, InstSI sad, SILLVMType retTy, Intrinsic intr> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType RetTy = retTy;
  Intrinsic Intr  = intr;
}

foreach desc = [
  GlobalLoadTRDesc<GLOBAL_LOAD_TR_B64_w32,  GLOBAL_LOAD_TR_B64_w32_SADDR,
                    si_llvm_v2i32_ty, int_amdgcn_global_load_tr_b64>,
  GlobalLoadTRDesc<GLOBAL_LOAD_TR_B64_w64,  GLOBAL_LOAD_TR_B64_w64_SADDR,
                    si_llvm_i32_ty,   int_amdgcn_global_load_tr_b64>,
  GlobalLoadTRDesc<GLOBAL_LOAD_TR_B128_w32, GLOBAL_LOAD_TR_B128_w32_SADDR,
                    si_llvm_v4i32_ty, int_amdgcn_global_load_tr_b128>,
  GlobalLoadTRDesc<GLOBAL_LOAD_TR_B128_w64, GLOBAL_LOAD_TR_B128_w64_SADDR,
                    si_llvm_v2i32_ty, int_amdgcn_global_load_tr_b128>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr, [(GetVal $addr)]))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (SetNamedOperand $vdst,
       (LLVMIntrinsic desc.RetTy, desc.Intr, [(GetVal $addr)]))
  ]>;
}

//------------------------------------------------------------------------------
// GLOBAL_LOAD/STORE_DWORD_ADDTID (+ SADDR) — per-thread-id global addressing.
//
// byte_addr = (SADDR ? saddr : 0) + offset + lane_id * 4
//   where lane_id = mbcnt_hi(-1, mbcnt_lo(-1, 0))
//
// MIR operands per FLAT_Global_Load_AddTid_Pseudo:
//   (outs $vdst) (ins [$saddr if SADDR,] $offset, $cpol)
//   No $vaddr — the lane id is derived implicitly via mbcnt.
//------------------------------------------------------------------------------

class GlobalAddTidAddr<bit hasSaddr> {
  // Per-thread byte address. mbcnt_hi(-1, mbcnt_lo(-1, 0)) yields the absolute
  // workitem id within the wave (0..wavesize-1), which is the per-thread
  // increment in 4-byte units.
  dag laneId4 = (LLVMMul
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_hi,
      [(ConstantSignedInt si_llvm_i32_ty, -1),
       (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_lo,
         [(ConstantSignedInt si_llvm_i32_ty, -1),
          (ConstantInt si_llvm_i32_ty, 0)])]),
    (ConstantInt si_llvm_i32_ty, 4));
  // Final i64 byte address.
  dag i64Addr = !if(hasSaddr,
    (LLVMAdd (GetNamedOperand $saddr),
              (LLVMZExt (LLVMAdd (GetNamedOperand $offset, si_llvm_i32_ty),
                                  laneId4),
                         si_llvm_i64_ty)),
    (LLVMZExt (LLVMAdd (GetNamedOperand $offset, si_llvm_i32_ty), laneId4),
              si_llvm_i64_ty));
  dag ret = (LLVMIntToPtr i64Addr, global_ptr_ty);
}

def GLOBAL_LOAD_DWORD_ADDTID_sem : InstSISemantic<GLOBAL_LOAD_DWORD_ADDTID, [
  (SetNamedOperand $vdst, (LLVMLoad si_llvm_i32_ty,
                                     GlobalAddTidAddr<0>.ret))
]>;

def GLOBAL_LOAD_DWORD_ADDTID_SADDR_sem :
    InstSISemantic<GLOBAL_LOAD_DWORD_ADDTID_SADDR, [
  (SetNamedOperand $vdst, (LLVMLoad si_llvm_i32_ty,
                                     GlobalAddTidAddr<1>.ret))
]>;

def GLOBAL_STORE_DWORD_ADDTID_sem :
    InstSISemantic<GLOBAL_STORE_DWORD_ADDTID, [
  (LLVMStore (GetNamedOperand $vdata), GlobalAddTidAddr<0>.ret)
]>;

def GLOBAL_STORE_DWORD_ADDTID_SADDR_sem :
    InstSISemantic<GLOBAL_STORE_DWORD_ADDTID_SADDR, [
  (LLVMStore (GetNamedOperand $vdata), GlobalAddTidAddr<1>.ret)
]>;

//==============================================================================
// _t16 (true-16) FLAT / GLOBAL / SCRATCH stores + GLOBAL / SCRATCH _D16_t16
// loads. With t16 the per-channel register is VGPR_16 (i16 wide), so:
//   * Loads write the loaded value directly to the 16-bit $vdst (no merge
//     with vdst_in needed — the register is only 16 bits).
//   * Stores read the 16-bit $vdata and either store it as-is (SHORT) or
//     truncate to i8 (BYTE).
//==============================================================================

//------------------------------------------------------------------------------
// GLOBAL _D16_t16 loads — base + SADDR. Same shape as FLAT _t16 loads,
// just using global address calc.
//------------------------------------------------------------------------------

class GlobalT16LoadDesc<InstSI base, InstSI sad, SILLVMType memTy, bit isSigned> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
}

foreach desc = [
  GlobalT16LoadDesc<GLOBAL_LOAD_UBYTE_D16_t16, GLOBAL_LOAD_UBYTE_D16_SADDR_t16,
                     si_llvm_i8_ty,  0>,
  GlobalT16LoadDesc<GLOBAL_LOAD_SBYTE_D16_t16, GLOBAL_LOAD_SBYTE_D16_SADDR_t16,
                     si_llvm_i8_ty,  1>,
  GlobalT16LoadDesc<GLOBAL_LOAD_SHORT_D16_t16, GLOBAL_LOAD_SHORT_D16_SADDR_t16,
                     si_llvm_i16_ty, 0>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $addr, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                        (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(!eq(desc.MemTy.ElemSize, 16),
           (GetVal $raw),
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
               (LLVMZExt (GetVal $raw), si_llvm_i16_ty))))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $addr, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                         (GetNamedOperand $vaddr),
                                         (GetNamedOperand $offset)>.ret),
    (DefVal $raw, (LLVMLoad desc.MemTy, (GetVal $addr))),
    (SetNamedOperand $vdst,
       !if(!eq(desc.MemTy.ElemSize, 16),
           (GetVal $raw),
           !if(desc.IsSigned,
               (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
               (LLVMZExt (GetVal $raw), si_llvm_i16_ty))))
  ]>;
}

//------------------------------------------------------------------------------
// SCRATCH _D16_t16 loads — base + SADDR + ST + SVS.
//------------------------------------------------------------------------------

class ScratchT16LoadDesc<InstSI base, InstSI sad, InstSI st, InstSI svs,
                          SILLVMType memTy, bit isSigned> {
  InstSI Base = base;
  InstSI Sad  = sad;
  InstSI ST   = st;
  InstSI SVS  = svs;
  SILLVMType MemTy = memTy;
  bit IsSigned = isSigned;
}

class ScratchT16LoadBody<SILLVMType memTy, bit isSigned, dag addr> {
  list<dag> body = [
    (DefVal $a, addr),
    (DefVal $raw, (LLVMLoad memTy, (GetVal $a))),
    (SetNamedOperand $vdst,
       !if(!eq(memTy.ElemSize, 16),
           (GetVal $raw),
           !if(isSigned,
               (LLVMSExt (GetVal $raw), si_llvm_i16_ty),
               (LLVMZExt (GetVal $raw), si_llvm_i16_ty))))
  ];
}

foreach desc = [
  ScratchT16LoadDesc<SCRATCH_LOAD_UBYTE_D16_t16,
                      SCRATCH_LOAD_UBYTE_D16_SADDR_t16,
                      SCRATCH_LOAD_UBYTE_D16_ST_t16,
                      SCRATCH_LOAD_UBYTE_D16_SVS_t16,
                      si_llvm_i8_ty,  0>,
  ScratchT16LoadDesc<SCRATCH_LOAD_SBYTE_D16_t16,
                      SCRATCH_LOAD_SBYTE_D16_SADDR_t16,
                      SCRATCH_LOAD_SBYTE_D16_ST_t16,
                      SCRATCH_LOAD_SBYTE_D16_SVS_t16,
                      si_llvm_i8_ty,  1>,
  ScratchT16LoadDesc<SCRATCH_LOAD_SHORT_D16_t16,
                      SCRATCH_LOAD_SHORT_D16_SADDR_t16,
                      SCRATCH_LOAD_SHORT_D16_ST_t16,
                      SCRATCH_LOAD_SHORT_D16_SVS_t16,
                      si_llvm_i16_ty, 0>
] in {
  def : InstSISemantic<desc.Base, ScratchT16LoadBody<desc.MemTy, desc.IsSigned,
    CalculateScratchAddr<(GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.Sad, ScratchT16LoadBody<desc.MemTy, desc.IsSigned,
    CalculateScratchSAddrAddr<(GetNamedOperand $saddr),
                                (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.ST,  ScratchT16LoadBody<desc.MemTy, desc.IsSigned,
    CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.SVS, ScratchT16LoadBody<desc.MemTy, desc.IsSigned,
    CalculateScratchSVSAddr<(GetNamedOperand $saddr),
                             (GetNamedOperand $vaddr),
                             (GetNamedOperand $offset)>.ret>.body>;
}

//------------------------------------------------------------------------------
// FLAT / GLOBAL / SCRATCH _t16 stores. $vdata is a 16-bit register; for
// BYTE_t16 we trunc to i8, for SHORT_t16 store as-is.
//------------------------------------------------------------------------------

// FLAT _t16 stores: BYTE / SHORT × {base, SADDR}.
class FLATT16StoreDesc<InstSI base, InstSI sad, SILLVMType storeTy,
                        bit needsTrunc> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType StoreTy = storeTy;
  bit NeedsTrunc = needsTrunc;
}

foreach desc = [
  FLATT16StoreDesc<FLAT_STORE_BYTE_t16,  FLAT_STORE_BYTE_SADDR_t16,
                    si_llvm_i8_ty,  1>,
  FLATT16StoreDesc<FLAT_STORE_SHORT_t16, FLAT_STORE_SHORT_SADDR_t16,
                    si_llvm_i16_ty, 0>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $a, CalculateFlatAddr<(GetNamedOperand $vaddr),
                                   (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.NeedsTrunc,
                   (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i16_ty),
                               desc.StoreTy),
                   (GetNamedOperand $vdata, si_llvm_i16_ty)),
                (GetVal $a))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $a, CalculateFlatSAddr<(GetNamedOperand $saddr),
                                    (GetNamedOperand $vaddr),
                                    (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.NeedsTrunc,
                   (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i16_ty),
                               desc.StoreTy),
                   (GetNamedOperand $vdata, si_llvm_i16_ty)),
                (GetVal $a))
  ]>;
}

// GLOBAL _t16 stores.
class GlobalT16StoreDesc<InstSI base, InstSI sad, SILLVMType storeTy,
                          bit needsTrunc> {
  InstSI Base = base;
  InstSI Sad  = sad;
  SILLVMType StoreTy = storeTy;
  bit NeedsTrunc = needsTrunc;
}

foreach desc = [
  GlobalT16StoreDesc<GLOBAL_STORE_BYTE_t16,  GLOBAL_STORE_BYTE_SADDR_t16,
                      si_llvm_i8_ty,  1>,
  GlobalT16StoreDesc<GLOBAL_STORE_SHORT_t16, GLOBAL_STORE_SHORT_SADDR_t16,
                      si_llvm_i16_ty, 0>
] in {
  def : InstSISemantic<desc.Base, [
    (DefVal $a, CalculateGlobalAddr<(GetNamedOperand $vaddr),
                                     (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.NeedsTrunc,
                   (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i16_ty),
                               desc.StoreTy),
                   (GetNamedOperand $vdata, si_llvm_i16_ty)),
                (GetVal $a))
  ]>;
  def : InstSISemantic<desc.Sad, [
    (DefVal $a, CalculateGlobalSAddr<(GetNamedOperand $saddr),
                                      (GetNamedOperand $vaddr),
                                      (GetNamedOperand $offset)>.ret),
    (LLVMStore !if(desc.NeedsTrunc,
                   (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i16_ty),
                               desc.StoreTy),
                   (GetNamedOperand $vdata, si_llvm_i16_ty)),
                (GetVal $a))
  ]>;
}

// SCRATCH _t16 stores: BYTE / SHORT × {base, SADDR, ST, SVS}.
class ScratchT16StoreDesc<InstSI base, InstSI sad, InstSI st, InstSI svs,
                           SILLVMType storeTy, bit needsTrunc> {
  InstSI Base = base;
  InstSI Sad  = sad;
  InstSI ST   = st;
  InstSI SVS  = svs;
  SILLVMType StoreTy = storeTy;
  bit NeedsTrunc = needsTrunc;
}

class ScratchT16StoreBody<SILLVMType storeTy, bit needsTrunc, dag addr> {
  list<dag> body = [
    (DefVal $a, addr),
    (LLVMStore !if(needsTrunc,
                   (LLVMTrunc (GetNamedOperand $vdata, si_llvm_i16_ty), storeTy),
                   (GetNamedOperand $vdata, si_llvm_i16_ty)),
                (GetVal $a))
  ];
}

foreach desc = [
  ScratchT16StoreDesc<SCRATCH_STORE_BYTE_t16,  SCRATCH_STORE_BYTE_SADDR_t16,
                       SCRATCH_STORE_BYTE_ST_t16, SCRATCH_STORE_BYTE_SVS_t16,
                       si_llvm_i8_ty,  1>,
  ScratchT16StoreDesc<SCRATCH_STORE_SHORT_t16, SCRATCH_STORE_SHORT_SADDR_t16,
                       SCRATCH_STORE_SHORT_ST_t16, SCRATCH_STORE_SHORT_SVS_t16,
                       si_llvm_i16_ty, 0>
] in {
  def : InstSISemantic<desc.Base, ScratchT16StoreBody<desc.StoreTy, desc.NeedsTrunc,
    CalculateScratchAddr<(GetNamedOperand $vaddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.Sad, ScratchT16StoreBody<desc.StoreTy, desc.NeedsTrunc,
    CalculateScratchSAddrAddr<(GetNamedOperand $saddr), (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.ST, ScratchT16StoreBody<desc.StoreTy, desc.NeedsTrunc,
    CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.SVS, ScratchT16StoreBody<desc.StoreTy, desc.NeedsTrunc,
    CalculateScratchSVSAddr<(GetNamedOperand $saddr), (GetNamedOperand $vaddr),
                             (GetNamedOperand $offset)>.ret>.body>;
}

//------------------------------------------------------------------------------
// SCRATCH_LOAD_LDS_* — DRAM/scratch → LDS direct copy (no VGPR landing).
//
// Hardware semantics:
//   scratch_byte_addr = (per-thread scratch base, addrspace 5) +
//                       address from {base | SADDR | ST | SVS} mode
//   lds_byte_addr     = M0 (implicit-use; the wave-uniform LDS offset)
//   LDS[lds_byte_addr] = MEM[scratch_byte_addr]   (data_byte_size bytes)
//
// We model as load i{N} from scratch_ptr + store i{N} to LDS pointer. The
// sub-DWORD forms transfer raw bytes (no sign/zero extension) — the S/U
// distinction in the opcode name is only for the assembler mnemonic; the
// data path is byte-identical.
//------------------------------------------------------------------------------

class ScratchLoadLdsDesc<InstSI base, InstSI sad, InstSI st, InstSI svs,
                          SILLVMType memTy> {
  InstSI Base = base;
  InstSI Sad  = sad;
  InstSI ST   = st;
  InstSI SVS  = svs;
  SILLVMType MemTy = memTy;
}

class ScratchLoadLdsBody<SILLVMType memTy, dag scratchAddr> {
  list<dag> body = [
    (DefVal $sa, scratchAddr),
    (DefVal $tmp, (LLVMLoad memTy, (GetVal $sa))),
    (DefVal $lds, (LLVMIntToPtr (ImplicitUse M0), local_ptr_ty)),
    (LLVMStore (GetVal $tmp), (GetVal $lds))
  ];
}

foreach desc = [
  ScratchLoadLdsDesc<SCRATCH_LOAD_LDS_UBYTE,
                      SCRATCH_LOAD_LDS_UBYTE_SADDR,
                      SCRATCH_LOAD_LDS_UBYTE_ST,
                      SCRATCH_LOAD_LDS_UBYTE_SVS,
                      si_llvm_i8_ty>,
  ScratchLoadLdsDesc<SCRATCH_LOAD_LDS_SBYTE,
                      SCRATCH_LOAD_LDS_SBYTE_SADDR,
                      SCRATCH_LOAD_LDS_SBYTE_ST,
                      SCRATCH_LOAD_LDS_SBYTE_SVS,
                      si_llvm_i8_ty>,
  ScratchLoadLdsDesc<SCRATCH_LOAD_LDS_USHORT,
                      SCRATCH_LOAD_LDS_USHORT_SADDR,
                      SCRATCH_LOAD_LDS_USHORT_ST,
                      SCRATCH_LOAD_LDS_USHORT_SVS,
                      si_llvm_i16_ty>,
  ScratchLoadLdsDesc<SCRATCH_LOAD_LDS_SSHORT,
                      SCRATCH_LOAD_LDS_SSHORT_SADDR,
                      SCRATCH_LOAD_LDS_SSHORT_ST,
                      SCRATCH_LOAD_LDS_SSHORT_SVS,
                      si_llvm_i16_ty>,
  ScratchLoadLdsDesc<SCRATCH_LOAD_LDS_DWORD,
                      SCRATCH_LOAD_LDS_DWORD_SADDR,
                      SCRATCH_LOAD_LDS_DWORD_ST,
                      SCRATCH_LOAD_LDS_DWORD_SVS,
                      si_llvm_i32_ty>
] in {
  def : InstSISemantic<desc.Base, ScratchLoadLdsBody<desc.MemTy,
    CalculateScratchAddr<(GetNamedOperand $vaddr),
                          (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.Sad, ScratchLoadLdsBody<desc.MemTy,
    CalculateScratchSAddrAddr<(GetNamedOperand $saddr),
                                (GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.ST, ScratchLoadLdsBody<desc.MemTy,
    CalculateScratchSTAddr<(GetNamedOperand $offset)>.ret>.body>;
  def : InstSISemantic<desc.SVS, ScratchLoadLdsBody<desc.MemTy,
    CalculateScratchSVSAddr<(GetNamedOperand $saddr),
                             (GetNamedOperand $vaddr),
                             (GetNamedOperand $offset)>.ret>.body>;
}
