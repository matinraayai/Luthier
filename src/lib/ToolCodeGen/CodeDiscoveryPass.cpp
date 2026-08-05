//===-- CodeDiscoveryPass.cpp ---------------------------------------------===//
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
/// \file CodeDiscoveryPass.cpp
/// Implements the \c CodeDiscoveryPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/CodeDiscoveryPass.h"
#include "AMDGPUTargetMachine.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/EntryPoint.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/InitialExecutionPointAnalysis.h"
#include "luthier/ToolCodeGen/InstructionTracesAnalysis.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/MIRToIRTranslator.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/PseudoOpcodeAndRegMapper.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include "luthier/ToolCodeGen/TraceCallGraph.h"
#include <MCTargetDesc/AMDGPUMCExpr.h>
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/ReachingDefAnalysis.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/TargetRegistry.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-code-discovery"

namespace luthier {

static inline llvm::Error
parseKDRsrc1(const llvm::amdhsa::kernel_descriptor_t &KD,
             const llvm::GCNTargetMachine &TM, llvm::Function &F) {
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);
  /// GRANULATED_WORKITEM_VGPR_COUNT is automatically calculated via the
  /// resource usage analysis pass, but we add its count as an attribute
  /// to the function for later use
  uint32_t GranulatedWorkitemVGPRCount = AMDHSA_BITS_GET(
      KD.compute_pgm_rsrc1,
      llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint32_t NextFreeVGPR =
      (GranulatedWorkitemVGPRCount + 1) * ST.getVGPREncodingGranule();
  F.addFnAttr("amdgpu-num-vgpr", llvm::formatv("{0}", NextFreeVGPR).str());
  /// PRIORITY is set by the CP at dispatch time

  /// FLOAT_ROUND_MODE_32, FLOAT_ROUND_MODE_16_64 fields are set to
  /// FP_ROUND_ROUND_TO_NEAREST no matter what by the AMDGPU AsmPrinter
  /// (see \c getFPMode in \c AMDGPUAsmPrinter.cpp). Capture the original
  /// 2-bit field values as \c amdgpu.kd.* attributes so that we
  /// pass can patch the rsrc1 bits back after the assembly is printed
  F.addFnAttr(
      "amdgpu.kd.float_round_mode_32",
      llvm::formatv(
          "{0}",
          AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                          llvm::amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32))
          .str());
  F.addFnAttr(
      "amdgpu.kd.float_round_mode_16_64",
      llvm::formatv("{0}",
                    AMDHSA_BITS_GET(
                        KD.compute_pgm_rsrc1,
                        llvm::amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64))
          .str());

  /// Lift FLOAT_DENORM_MODE_32 field
  auto Float32Denorm =
      AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                      llvm::amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  /// LLVM parses "denormal-fp-math[-f32]" as "<output>,<input>" and defaults
  /// an empty input component to the output component, so each component must
  /// be spelled out explicitly: "preserve-sign," would wrongly round-trip as
  /// flush-both rather than flush-output-only.
  std::string Denorm32Val;
  switch (Float32Denorm) {
  case FP_DENORM_FLUSH_IN_FLUSH_OUT:
    Denorm32Val = "preserve-sign,preserve-sign";
    break;
  case FP_DENORM_FLUSH_OUT:
    Denorm32Val = "preserve-sign,ieee";
    break;
  case FP_DENORM_FLUSH_IN:
    Denorm32Val = "ieee,preserve-sign";
    break;
  case FP_DENORM_FLUSH_NONE:
    Denorm32Val = "ieee,ieee";
    break;
  default:
    return LUTHIER_MAKE_GENERIC_ERROR("Invalid FP 32 denorm field " +
                                      llvm::to_string(Float32Denorm) + ".");
  }
  F.addFnAttr("denormal-fp-math-f32", Denorm32Val);
  /// Lift FLOAT_DENORM_MODE_16_64 field
  auto Float1664Denorm =
      AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                      llvm::amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  std::string Denorm1664Val;
  switch (Float1664Denorm) {
  case FP_DENORM_FLUSH_IN_FLUSH_OUT:
    Denorm1664Val = "preserve-sign,preserve-sign";
    break;
  case FP_DENORM_FLUSH_OUT:
    Denorm1664Val = "preserve-sign,ieee";
    break;
  case FP_DENORM_FLUSH_IN:
    Denorm1664Val = "ieee,preserve-sign";
    break;
  case FP_DENORM_FLUSH_NONE:
    Denorm1664Val = "ieee,ieee";
    break;
  default:
    return LUTHIER_MAKE_GENERIC_ERROR("Invalid FP 16/64 denorm field " +
                                      llvm::to_string(Float1664Denorm) + ".");
  }
  F.addFnAttr("denormal-fp-math", Denorm1664Val);

  /// PRIV is set by CP and is zero

  /// Lift ENABLE_DX10_CLAMP if gfx11-; Otherwise lift WG_RR_EN
  if (ST.hasRrWGMode()) {
    /// WG_RR_EN is set to zero by the backend; capture original bit as a
    /// harmless \c amdgpu.kd.* attribute for the post-AsmPrinter patcher.
    F.addFnAttr("amdgpu.kd.enable_wg_rr_en",
                AMDHSA_BITS_GET(
                    KD.compute_pgm_rsrc1,
                    llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX12_PLUS_ENABLE_WG_RR_EN)
                    ? "1"
                    : "0");
  } else {
    F.addFnAttr(
        "amdgpu-dx10-clamp",
        AMDHSA_BITS_GET(
            KD.compute_pgm_rsrc1,
            llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP)
            ? "true"
            : "false");
  }

  /// DEBUG_MODE is set by the CP

  /// Lift ENABLE_IEEE_MODE if gfx11-; DISABLE_PERF is reserved (gfx12+) and
  /// is set to zero by the AsmPrinter
  if (ST.hasDX10ClampAndIEEEMode()) {
    F.addFnAttr("amdgpu-ieee",
                AMDHSA_BITS_GET(
                    KD.compute_pgm_rsrc1,
                    llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE)
                    ? "true"
                    : "false");
  }
  if (llvm::AMDGPU::isGFX12Plus(ST)) {
    F.addFnAttr(
        "amdgpu.kd.disable_perf",
        AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                        llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX12_PLUS_DISABLE_PERF)
            ? "1"
            : "0");
  }

  /// BULKY is set by the CP at dispatch time

  /// CDBG_USER is set by the CP at dispatch time

  /// FP16_OVFL (gfx9+) can be queried on device code, but there's no place to
  /// set it at the MIR level
  if (llvm::AMDGPU::isGFX9Plus(ST)) {
    F.addFnAttr(
        "amdgpu.kd.fp16_ovfl",
        AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                        llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX9_PLUS_FP16_OVFL)
            ? "1"
            : "0");
  }

  /// FLAT_SCRATCH_IS_NV (gfx12.5, bit 27) — no IR handle, AsmPrinter doesn't
  /// write it. Capture for the post-AsmPrinter patcher
  if (llvm::AMDGPU::isGFX1250(ST)) {
    F.addFnAttr("amdgpu.kd.flat_scratch_is_nv",
                AMDHSA_BITS_GET(
                    KD.compute_pgm_rsrc1,
                    llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX125_FLAT_SCRATCH_IS_NV)
                    ? "1"
                    : "0");
  }

  /// WGP_MODE is set by the cumode feature in the subtarget

  /// MEM_ORDERED (gfx10+) is always set to 1 by the AsmPrinter, so we will fix
  /// it once we have printed the relocatable of the instrumented object
  if (llvm::AMDGPU::isGFX10Plus(ST)) {
    F.addFnAttr(
        "amdgpu.kd.mem_ordered",
        AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                        llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX10_PLUS_MEM_ORDERED)
            ? "1"
            : "0");

    /// FWD_PROGRESS (gfx10+) is always set to 1 by the AsmPrinter. Capture the
    /// original bit so the post-AsmPrinter patcher can restore it
    F.addFnAttr(
        "amdgpu.kd.fwd_progress",
        AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                        llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX10_PLUS_FWD_PROGRESS)
            ? "1"
            : "0");
  }

  return llvm::Error::success();
}

static inline llvm::Error
parseKDRsrc2(const llvm::amdhsa::kernel_descriptor_t &KD,
             const llvm::GCNTargetMachine &TM, llvm::Function &F) {
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);
  /// ENABLE_PRIVATE_SEGMENT is set by the backend when the stack size is
  /// non-zero; We capture it here anyway for more info
  F.addFnAttr(
      "amdgpu.kd.enable_private_segment",
      AMDHSA_BITS_GET(KD.compute_pgm_rsrc2,
                      llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT)
          ? "1"
          : "0");

  /// USER_SGPR_COUNT is automatically set by the backend from MFI's user-SGPR
  /// usage (5 bits on gfx6-gfx12.0, 6 bits on gfx12.5). Capture for safety.
  if (llvm::AMDGPU::isGFX1250(ST)) {
    F.addFnAttr(
        "amdgpu.kd.user_sgpr_count",
        llvm::formatv(
            "{0}", AMDHSA_BITS_GET(
                       KD.compute_pgm_rsrc2,
                       llvm::amdhsa::COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT))
            .str());
  } else {
    F.addFnAttr(
        "amdgpu.kd.user_sgpr_count",
        llvm::formatv(
            "{0}",
            AMDHSA_BITS_GET(
                KD.compute_pgm_rsrc2,
                llvm::amdhsa::COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT))
            .str());
  }

  /// ENABLE_TRAP_HANDLER (gfx6-gfx11) is not set in HSA; For other OSes, it
  /// should be set when creating the target. Capture the original bit as a
  /// harmless \c amdgpu.kd.* attribute for the post-AsmPrinter patcher
  if (!llvm::AMDGPU::isGFX12Plus(ST)) {
    F.addFnAttr(
        "amdgpu.kd.enable_trap_handler",
        AMDHSA_BITS_GET(
            KD.compute_pgm_rsrc2,
            llvm::amdhsa::COMPUTE_PGM_RSRC2_GFX6_GFX11_ENABLE_TRAP_HANDLER)
            ? "1"
            : "0");
  }

  /// ENABLE_DYNAMIC_VGPR (gfx12.0 in rsrc2; gfx12.5 lives in rsrc3) has no
  /// MIR / MC handle. Capture the bit for the post-AsmPrinter patcher
  if (llvm::AMDGPU::isGFX12Plus(ST) && !llvm::AMDGPU::isGFX1250(ST)) {
    F.addFnAttr("amdgpu.kd.enable_dynamic_vgpr",
                AMDHSA_BITS_GET(
                    KD.compute_pgm_rsrc2,
                    llvm::amdhsa::COMPUTE_PGM_RSRC2_GFX120_ENABLE_DYNAMIC_VGPR)
                    ? "1"
                    : "0");
  }

  /// Lift ENABLE_SGPR_WORKGROUP_ID_X, Y and Z. Each dimension also gets
  /// the matching \c amdgpu-no-cluster-id-* attribute because
  /// SIMachineFunctionInfo's ctor enables \c WorkGroupID<dim> if EITHER
  /// the workgroup-id OR the cluster-id attribute is absent (an OR, not
  /// an AND — see SIMachineFunctionInfo.cpp:141-147). Without the
  /// cluster-id companion, our \c amdgpu-no-workgroup-id-y/z attrs are
  /// effectively a no-op and the relifted KD wrongly enables
  /// system_sgpr_workgroup_id_y/z, shifting all downstream SGPR slots
  /// and breaking kernarg loads. The cluster-id concept is gfx12.5-only;
  /// for every other target there is no cluster, so disabling it
  /// unconditionally when we'd disable workgroup-id is correct.
  if (AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) == 0) {
    F.addFnAttr("amdgpu-no-workgroup-id-x");
    F.addFnAttr("amdgpu-no-cluster-id-x");
  }

  if (AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) == 0) {
    F.addFnAttr("amdgpu-no-workgroup-id-y");
    F.addFnAttr("amdgpu-no-cluster-id-y");
  }
  if (AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) == 0) {
    F.addFnAttr("amdgpu-no-workgroup-id-z");
    F.addFnAttr("amdgpu-no-cluster-id-z");
  }
  /// ENABLE_SGPR_WORKGROUP_INFO is represented in MFI, but it is always set
  /// to false and there is no way to set it. Capture the original bit for the
  /// post-AsmPrinter patcher
  F.addFnAttr("amdgpu.kd.enable_sgpr_workgroup_info",
              AMDHSA_BITS_GET(
                  KD.compute_pgm_rsrc2,
                  llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO)
                  ? "1"
                  : "0");

  /// Lift ENABLE_VGPR_WORKITEM_ID
  switch (AMDHSA_BITS_GET(
      KD.compute_pgm_rsrc2,
      llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID)) {
  case 0:
    // TIDIG_COMP_CNT == 0 → only X is enabled; disable both Y and Z.
    F.addFnAttr("amdgpu-no-workitem-id-y");
    [[fallthrough]];
  case 1:
    // TIDIG_COMP_CNT == 1 → X and Y enabled; disable only Z.
    F.addFnAttr("amdgpu-no-workitem-id-z");
    [[fallthrough]];
  case 2:
    // TIDIG_COMP_CNT == 2 → X, Y, Z all enabled; nothing to disable.
    break;
  default:
    return LUTHIER_MAKE_GENERIC_ERROR("KD's VGPR workitem ID is not valid");
  }
  /// ENABLE_EXCEPTION_ADDRESS_WATCH / ENABLE_EXCEPTION_MEMORY are always set
  /// to zero by the AsmPrinter. Capture the original bits for the
  /// post-AsmPrinter patcher
  F.addFnAttr(
      "amdgpu.kd.enable_exception_address_watch",
      AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_ADDRESS_WATCH)
          ? "1"
          : "0");
  F.addFnAttr(
      "amdgpu.kd.enable_exception_memory",
      AMDHSA_BITS_GET(KD.compute_pgm_rsrc2,
                      llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_MEMORY)
          ? "1"
          : "0");

  /// GRANULATED_LDS_SIZE is automatically via the dispatch packet

  /// The six IEEE_754_FP exception-enable bits and
  /// ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO are all written as zero by the
  /// AsmPrinter. Capture each original bit so the post-AsmPrinter patcher can
  /// restore them.
  F.addFnAttr(
      "amdgpu.kd.enable_exception_ieee_754_fp_invalid_operation",
      AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::
              COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION)
          ? "1"
          : "0");
  F.addFnAttr(
      "amdgpu.kd.enable_exception_fp_denormal_source",
      AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE)
          ? "1"
          : "0");
  F.addFnAttr(
      "amdgpu.kd.enable_exception_ieee_754_fp_division_by_zero",
      AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::
              COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO)
          ? "1"
          : "0");
  F.addFnAttr(
      "amdgpu.kd.enable_exception_ieee_754_fp_overflow",
      AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW)
          ? "1"
          : "0");
  F.addFnAttr("amdgpu.kd.enable_exception_ieee_754_fp_underflow",
              AMDHSA_BITS_GET(
                  KD.compute_pgm_rsrc2,
                  llvm::amdhsa::
                      COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW)
                  ? "1"
                  : "0");
  F.addFnAttr(
      "amdgpu.kd.enable_exception_ieee_754_fp_inexact",
      AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT)
          ? "1"
          : "0");
  F.addFnAttr(
      "amdgpu.kd.enable_exception_int_divide_by_zero",
      AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO)
          ? "1"
          : "0");

  return llvm::Error::success();
}

static inline llvm::Error
parseKDRsrc3(const llvm::amdhsa::kernel_descriptor_t &KD,
             const llvm::GCNTargetMachine &TM, llvm::Function &F) {
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);

  /// Rsrc3 for GFX90A and GFX942 ==============================================
  /// ACCUM_OFFSET is automatically set by the backend from vgpr/agpr usage of
  /// the MF
  /// TG_SPLIT is set by the backend via the cumode subtarget feature

  /// Rsrc3 for GFX10 and GFX11 ================================================
  /// SHARED_VGPR_COUNT (4 bits) is not written by the AsmPrinter into the
  /// binary KD. Capture for the post-AsmPrinter patcher
  if (llvm::AMDGPU::isGFX10_GFX11(ST)) {
    F.addFnAttr(
        "amdgpu.kd.shared_vgpr_count",
        llvm::formatv(
            "{0}",
            AMDHSA_BITS_GET(
                KD.compute_pgm_rsrc3,
                llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX10_GFX11_SHARED_VGPR_COUNT))
            .str());
  }

  /// Rsrc3 for GFX11 only =====================================================
  /// INST_PREF_SIZE (gfx11: 6 bits) is automatically calculated by the
  /// backend from the kernel code size. After instrumentation grows the
  /// kernel the recomputed value will legitimately differ from the original,
  /// so the post-AsmPrinter patcher generally should NOT clobber the
  /// backend's value — the captured attribute is provided as the original
  /// prefetch hint for any patcher policy that wants to preserve it.
  /// TRAP_ON_START / TRAP_ON_END are filled in by the CP at dispatch time —
  /// not a compiler concern, so no capture
  if (llvm::AMDGPU::isGFX11(ST)) {
    F.addFnAttr(
        "amdgpu.kd.inst_pref_size",
        llvm::formatv("{0}",
                      AMDHSA_BITS_GET(
                          KD.compute_pgm_rsrc3,
                          llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX11_INST_PREF_SIZE))
            .str());
  }

  /// Rsrc3 for GFX11+ =========================================================
  /// IMAGE_OP is not set by the backend. Capture for the post-AsmPrinter
  /// patcher
  if (llvm::AMDGPU::isGFX11Plus(ST)) {
    F.addFnAttr(
        "amdgpu.kd.image_op",
        AMDHSA_BITS_GET(KD.compute_pgm_rsrc3,
                        llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX11_PLUS_IMAGE_OP)
            ? "1"
            : "0");
  }

  /// Rsrc3 for GFX12+ =========================================================
  /// INST_PREF_SIZE (gfx12+: 8 bits) is automatically calculated by the
  /// backend. Same caveat as the gfx11 capture: the post-patcher should
  /// generally let the backend's recomputed value win once instrumentation
  /// grows the kernel; the captured attribute is only useful for policies
  /// that want to preserve the original prefetch hint.
  /// GLG_EN is not set by either the backend or MC. Capture
  if (llvm::AMDGPU::isGFX12Plus(ST)) {
    F.addFnAttr(
        "amdgpu.kd.inst_pref_size",
        llvm::formatv(
            "{0}",
            AMDHSA_BITS_GET(
                KD.compute_pgm_rsrc3,
                llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE))
            .str());
    F.addFnAttr(
        "amdgpu.kd.glg_en",
        AMDHSA_BITS_GET(KD.compute_pgm_rsrc3,
                        llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_GLG_EN)
            ? "1"
            : "0");
  }

  /// Rsrc3 for GFX12.5 only ===================================================
  if (llvm::AMDGPU::isGFX1250(ST)) {
    /// NAMED_BAR_CNT (3 bits) is set by the backend from \c
    /// ProgInfo.NamedBarCnt which is derived from the LDS named-barrier
    /// GlobalVariables present in the IR module (via \c isNamedBarrier checks
    /// during SelectionDAG lowering). If the lifter doesn't reconstruct those
    /// LDS globals the recomputed count will drop to zero, so capture the
    /// original 3-bit field as a safety net for the post-AsmPrinter patcher.
    F.addFnAttr(
        "amdgpu.kd.named_bar_cnt",
        llvm::formatv("{0}",
                      AMDHSA_BITS_GET(
                          KD.compute_pgm_rsrc3,
                          llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT))
            .str());
    /// ENABLE_DYNAMIC_VGPR (gfx12.5 puts this in rsrc3; gfx12.0 puts the same
    /// flag in rsrc2 — captured there). Not written by the AsmPrinter
    /// TCP_SPLIT (3 bits) and ENABLE_DIDT_THROTTLE (1 bit) likewise only appear
    /// in the disassembler's pseudo-directive output and are not written by the
    /// AsmPrinter. Capture all three for the post-AsmPrinter patcher
    F.addFnAttr("amdgpu.kd.enable_dynamic_vgpr",
                AMDHSA_BITS_GET(
                    KD.compute_pgm_rsrc3,
                    llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX125_ENABLE_DYNAMIC_VGPR)
                    ? "1"
                    : "0");
    F.addFnAttr(
        "amdgpu.kd.tcp_split",
        llvm::formatv(
            "{0}",
            AMDHSA_BITS_GET(KD.compute_pgm_rsrc3,
                            llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT))
            .str());
    F.addFnAttr("amdgpu.kd.enable_didt_throttle",
                AMDHSA_BITS_GET(
                    KD.compute_pgm_rsrc3,
                    llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX125_ENABLE_DIDT_THROTTLE)
                    ? "1"
                    : "0");
  }

  return llvm::Error::success();
}

/// Mirror each \c kernel_code_properties bit that's NOT set in the original
/// \p KD as the corresponding \c amdgpu-no-* function attribute. Must run
/// BEFORE the \c MachineFunction is constructed for \p F. The AMDGPU
/// backend's \c GCNUserSGPRUsageInfo (a private sub-object of
/// \c SIMachineFunctionInfo) is what \c AMDGPUAsmPrinter reads to decide
/// which \c KERNEL_CODE_PROPERTY_ENABLE_SGPR_* bits to set on the emitted
/// KD — NOT \c MFI->ArgInfo or the \c MFI->addX(TRI)-style calls below.
/// \c UserSGPRInfo's private bool flags are populated once, by its
/// constructor reading the function's \c amdgpu-no-* attrs (default-TRUE
/// if absent, see GCNSubtarget.cpp:706-716), and the class exposes no
/// setter to flip a flag back to false post-construction. Setting the
/// attrs here, before \c MachineFunctionAnalysis::run materializes the
/// MF, keeps \c UserSGPRInfo in lockstep with the original KD: the asm
/// printer then emits a KD whose \c kernel_code_properties matches what
/// the lifted kernel code was compiled for. Without this, the kernarg
/// pointer ends up at a later SGPR pair than \c $sgpr0_sgpr1 and the
/// lifted code reads garbage as kernargs.
static inline llvm::Error
parseKDKernelCodeAttrs(const llvm::amdhsa::kernel_descriptor_t &KD,
                       const llvm::GCNTargetMachine &TM, llvm::Function &F) {
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) == 0)
    F.addFnAttr("amdgpu-no-dispatch-ptr");

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) == 0)
    F.addFnAttr("amdgpu-no-queue-ptr");

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) ==
      0)
    F.addFnAttr("amdgpu-no-kernarg-segment-ptr");

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) == 0)
    F.addFnAttr("amdgpu-no-dispatch-id");

  // FLAT_SCRATCH_INIT is subtarget-dependent: under hasArchitectedFlatScratch
  // (gfx10.3+, gfx11+, gfx12) the FS is preloaded into a fixed architectural
  // register and the user-SGPR FS-init path is never used, so the negative
  // attr is correct unconditionally. On non-architected targets the attr
  // mirrors the KD bit.
  if (ST.hasArchitectedFlatScratch() ||
      AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) ==
          0)
    F.addFnAttr("amdgpu-no-flat-scratch-init");

  return llvm::Error::success();
}

/// Slim post-\c MachineFunction kernel-code parser: now responsible only for
/// the parts that genuinely need an MF (frame-info dynamic stack setup) and
/// for seeding \c MFI->ArgInfo physreg slots from the same KD bits the
/// pre-MF \c parseKDKernelCodeAttrs already mirrored into the function
/// attribute set. The attr-driven \c GCNUserSGPRUsageInfo path is what
/// drives the emitted KD's \c kernel_code_properties bits (see
/// \c parseKDKernelCodeAttrs); the \c MFI->addX(*TRI) calls below populate
/// \c ArgInfo so later passes can ask "what physreg holds dispatch_ptr?"
/// without re-deriving it from attrs.
inline llvm::Error
parseKDKernelCode(const llvm::amdhsa::kernel_descriptor_t &KD,
                  const llvm::GCNTargetMachine &TM, llvm::MachineFunction &MF) {
  llvm::Function &F = MF.getFunction();
  auto MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
  auto TRI = static_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(F)->getRegisterInfo());
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);

  if (!ST.hasArchitectedFlatScratch()) {
    if (AMDHSA_BITS_GET(
            KD.kernel_code_properties,
            llvm::amdhsa::
                KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) == 1) {
      MFI->addPrivateSegmentBuffer(*TRI);
    }
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) == 1) {
    MFI->addDispatchPtr(*TRI);
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) == 1) {
    MFI->addQueuePtr(*TRI);
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) ==
      1) {
    MFI->addKernargSegmentPtr(*TRI);
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) == 1) {
    MFI->addDispatchID(*TRI);
  }

  if (ST.hasArchitectedFlatScratch()) {
    if (AMDHSA_BITS_GET(
            KD.kernel_code_properties,
            llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) ==
        1) {
      MFI->addFlatScratchInit(*TRI);
    }
  }
  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::
              KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) == 1) {
    MFI->addPrivateSegmentSize(*TRI);
  }

  /// Wavefront32 should be taken care of when creating the target machine

  /// Set the size of the stack based on whether we have a dynamic stack or not
  if (KD.private_segment_fixed_size != 0) {
    llvm::MachineFrameInfo &FrameInfo = MF.getFrameInfo();
    FrameInfo.CreateFixedObject(KD.private_segment_fixed_size, 0, true);
    FrameInfo.setStackSize(KD.private_segment_fixed_size);
    if (AMDHSA_BITS_GET(
            KD.kernel_code_properties,
            llvm::amdhsa::KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK)) {
      FrameInfo.CreateVariableSizedObject(llvm::Align(4), nullptr);
    }
  }

  return llvm::Error::success();
}

/// Initializes the \c llvm::Function and \c llvm::MachineFunction for the
/// entry point for the \p KD
static llvm::Expected<std::pair<llvm::MachineFunction &,
                                std::optional<object::AMDGCNElfSymbolRef>>>
initKernelEntryPointFunction(const llvm::amdhsa::kernel_descriptor_t &KD,
                             const MemoryAllocationAccessor &SegAccessor,
                             llvm::Module &TargetModule,
                             const llvm::GCNTargetMachine &TM,
                             llvm::FunctionAnalysisManager &FAM) {
  llvm::LLVMContext &LLVMContext = TargetModule.getContext();
  /// Get the memory allocation associated with the kernel descriptor
  auto KDLoadAddr = reinterpret_cast<uint64_t>(&KD);
  auto SegmentOrErr = SegAccessor.getAllocationDescriptor(KDLoadAddr);
  LUTHIER_RETURN_ON_ERROR(SegmentOrErr.takeError());

  if (SegmentOrErr->empty())
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Failed to find the memory allocation descriptor "
                      "associated with KD {0:x}.",
                      KDLoadAddr));

  uint64_t KDLoadOffset =
      KDLoadAddr -
      reinterpret_cast<uint64_t>(SegmentOrErr->getDeviceAllocation().data());

  std::optional<object::AMDGCNKernelDescSymbolRef> KDSymbolIfPresent{
      std::nullopt};

  /// If the KD allocation has a code object associated with it,
  /// locate the KD's symbol
  if (auto *ObjFile = SegmentOrErr->getAllocationCodeObject()) {
    /// Lambda is used instead of a plain loop for easier break/error checking
    llvm::Expected<object::AMDGCNKernelDescSymbolRef> KDSymbolOrErr =
        [&]() -> llvm::Expected<object::AMDGCNKernelDescSymbolRef> {
      llvm::Error Err = llvm::Error::success();
      for (object::AMDGCNKernelDescSymbolRef CurrentKD :
           ObjFile->kernel_descriptors(Err)) {
        LUTHIER_RETURN_ON_ERROR(Err);
        llvm::Expected<uint64_t> CurrentKDLoadAddrOrErr =
            CurrentKD.getAddress();
        LUTHIER_RETURN_ON_ERROR(CurrentKDLoadAddrOrErr.takeError());
        if (*CurrentKDLoadAddrOrErr == KDLoadOffset) {
          return CurrentKD;
        }
      }
      LUTHIER_RETURN_ON_ERROR(Err);
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Failed to get the KD associated with address {0:x}", KDLoadOffset));
    }();
    LUTHIER_RETURN_ON_ERROR(KDSymbolOrErr.takeError());

    KDSymbolIfPresent = *KDSymbolOrErr;
  }

  /// If we have an object symbol associated with the kernel descriptor, then
  /// we find its name from its kernel symbol; Otherwise, give it a name
  /// based on its load address
  std::string KernelName;
  if (KDSymbolIfPresent.has_value()) {
    llvm::Expected<llvm::StringRef> KernelNameOrErr =
        KDSymbolIfPresent->getName();
    LUTHIER_RETURN_ON_ERROR(KernelNameOrErr.takeError());
    KernelName = KernelNameOrErr->substr(0, KernelNameOrErr->rfind(".kd"));
  } else {
    KernelName = llvm::formatv("kernel-{0:x}", KDLoadAddr);
  }

  auto &KDOnHost = *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      &SegmentOrErr->getHostAllocation()[KDLoadOffset]);

  /// Kernel's return type is always void.
  ///
  /// We do not attempt to recover the original high-level kernel
  /// prototype from the metadata; the raised LLVM IR has to translate
  /// low-level argument-buffer loads, not "read from the i-th IR arg"
  /// (e.g. \c s_load_dword \c s[4:5], \c s[4:5], \c 0x0 raises to a
  /// memory load, regardless of how the source language declared the
  /// arg). A kernel hand-written in assembly may also interleave
  /// explicit and implicit arguments in ways that don't fit the LLVM
  /// "explicit first, then implicit" convention.
  ///
  /// When \c KD.kernarg_size is non-zero, we synthesize a single
  /// \c byref([KD.kernarg_size x i8]) parameter so the AsmPrinter
  /// emits a matching \c kernarg_size in the relifted KD.
  /// \c AMDGPUSubtarget::getKernArgSegmentSize is driven by
  /// \c getExplicitKernArgSize, which walks \c F.args() and only
  /// consults \c byref types / param-align; there is no attribute-based
  /// override. An empty arg list would produce \c kernarg_size=0 in
  /// the emitted KD, and a kernel that loads from \c s[0:1]+offset
  /// would read garbage at dispatch. A single opaque byref-array arg
  /// has the right total byte count without implying any layout on
  /// the kernel body, which still reads via raw \c s_load instructions.
  /// When \c KD.kernarg_size is zero we keep the arg list empty so
  /// the lifted IR signature matches the source: no kernarg buffer
  /// means no IR param.
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);

  llvm::SmallVector<llvm::Type *, 1> ParamTypes;
  if (KDOnHost.kernarg_size != 0)
    ParamTypes.push_back(
        llvm::PointerType::get(LLVMContext, llvm::AMDGPUAS::CONSTANT_ADDRESS));
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, ParamTypes, false);

  llvm::Function *F =
      llvm::Function::Create(FunctionType, llvm::GlobalValue::WeakAnyLinkage,
                             KernelName, TargetModule);
  F->setVisibility(llvm::GlobalValue::ProtectedVisibility);

  // Populate the Attributes =================================================

  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated

  /// Tag the kernarg buffer param with \c byref([N x i8]) so
  /// \c getExplicitKernArgSize totals \c KD.kernarg_size for the
  /// AsmPrinter. Use 8-byte alignment to match the HSA kernarg buffer
  /// alignment requirement (\c hsa_kernel_dispatch_packet_t's
  /// \c kernarg_address is 8-byte-aligned by the runtime).
  if (KDOnHost.kernarg_size != 0) {
    llvm::ArrayType *KernArgBufTy = llvm::ArrayType::get(
        llvm::Type::getInt8Ty(LLVMContext), KDOnHost.kernarg_size);
    llvm::AttrBuilder AB(LLVMContext);
    AB.addByRefAttr(KernArgBufTy);
    AB.addAlignmentAttr(llvm::Align(8));
    F->addParamAttrs(0, AB);
  }

  F->addFnAttr("amdgpu-lds-size",
               llvm::to_string(KDOnHost.group_segment_fixed_size));

  /// Lift Rsrc1-Rsrc2; Rsrc3 is mostly automatically computed so we don't
  /// lift it
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc1(KDOnHost, TM, *F));
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc2(KDOnHost, TM, *F));
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc3(KDOnHost, TM, *F));

  /// Stamp the kernel_code_properties-derived `amdgpu-no-*` attrs BEFORE
  /// the MF is constructed. SIMachineFunctionInfo's GCNUserSGPRUsageInfo
  /// sub-object reads them only at ctor time and exposes no setters, so
  /// missing this pre-MF write would leave UserSGPRInfo with all the
  /// default-enabled user-SGPR slots set even when the original KD had
  /// those bits cleared.
  LUTHIER_RETURN_ON_ERROR(parseKDKernelCodeAttrs(KDOnHost, TM, *F));

  // Populate the MFI ==========================================================

  llvm::MachineFunction &MF =
      FAM.getResult<llvm::MachineFunctionAnalysis>(*F).getMF();

  /// Parse the kernel code field of the kernel
  LUTHIER_RETURN_ON_ERROR(parseKDKernelCode(KDOnHost, TM, MF));

  auto *MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(*F);

  /// Set pre-loaded kernel argument field for targets that support it.
  /// The preloaded kernargs occupy the SGPRs immediately after the standard
  /// user SGPRs; the MIRToIRTranslator seeds them with the equivalent kernarg
  /// loads (using the captured offset/length attributes).
  if (ST.hasKernargPreload()) {
    unsigned PreloadLength = AMDHSA_BITS_GET(
        KDOnHost.kernarg_preload, llvm::amdhsa::KERNARG_PRELOAD_SPEC_LENGTH);
    /// Tracks the preload count in \c GCNUserSGPRUsageInfo (so the re-emitted
    /// KD carries the right preload length).
    MFI->getUserSGPRInfo().allocKernargPreloadSGPRs(PreloadLength);
    /// \c allocKernargPreloadSGPRs only updates \c GCNUserSGPRUsageInfo; we
    /// must also advance \c SIMachineFunctionInfo's user-SGPR count, otherwise
    /// the workgroup-ID system SGPRs added below — placed at
    /// \c getNextSystemSGPR() == \c SGPR0+NumUserSGPRs+NumSystemSGPRs — would
    /// alias the registers that hold the preloaded kernargs.
    for (unsigned I = 0; I < PreloadLength; ++I)
      (void)MFI->addReservedUserSGPR();

    F->addFnAttr("amdgpu.kd.kernarg_preload_length",
                 llvm::formatv("{0}", PreloadLength).str());
    F->addFnAttr(
        "amdgpu.kd.kernarg_preload_offset",
        llvm::formatv(
            "{0}", AMDHSA_BITS_GET(KDOnHost.kernarg_preload,
                                   llvm::amdhsa::KERNARG_PRELOAD_SPEC_OFFSET))
            .str());
  }

  /// Fix up some of the MFI fields that have a direct IR attribute but doesn't
  /// get populated on creating the machine function

  /// Workitem argument values
  constexpr unsigned PackedMask = 0x3ff;
  bool HasPackedTID = ST.hasFeature(llvm::AMDGPU::FeaturePackedTID);

  if (!F->hasFnAttribute("amdgpu-no-workitem-id-x")) {
    unsigned WorkItemXMask = HasPackedTID ? PackedMask : ~0u;
    MFI->setWorkItemIDX(llvm::ArgDescriptor::createRegister(llvm::AMDGPU::VGPR0,
                                                            WorkItemXMask));
  }
  if (!F->hasFnAttribute("amdgpu-no-workitem-id-y")) {
    llvm::MCRegister WorkItemYReg =
        HasPackedTID ? llvm::AMDGPU::VGPR0 : llvm::AMDGPU::VGPR1;
    unsigned WorkItemYMask = HasPackedTID ? PackedMask << 10 : ~0u;
    MFI->setWorkItemIDY(
        llvm::ArgDescriptor::createRegister(WorkItemYReg, WorkItemYMask));
  }

  if (!F->hasFnAttribute("amdgpu-no-workitem-id-z")) {
    llvm::MCRegister WorkItemZReg =
        HasPackedTID ? llvm::AMDGPU::VGPR0 : llvm::AMDGPU::VGPR2;
    unsigned WorkItemZMask = HasPackedTID ? PackedMask << 20 : ~0u;
    MFI->setWorkItemIDZ(
        llvm::ArgDescriptor::createRegister(WorkItemZReg, WorkItemZMask));
  }

  /// Workgroup IDs. On targets with the architected-SGPRs feature the
  /// workgroup IDs are sourced from fixed architected registers (TTMP9 for X,
  /// TTMP7 for the packed Y/Z) rather than from allocated system SGPRs, so we
  /// must NOT consume system SGPRs for them — this mirrors
  /// \c SITargetLowering::allocateSystemSGPRs, which gates the workgroup-ID
  /// SGPR allocation on \c !hasArchitectedSGPRs(). The feature is on by
  /// default for GFX12+, but is a plain subtarget feature (\c
  /// +architected-sgprs) that can also be enabled on some GFX9 parts, so gate
  /// on the predicate rather than a generation check. The \c MIRToIRTranslator
  /// seeds the architected TTMP registers directly for these targets.
  if (!ST.hasArchitectedSGPRs()) {
    if (!F->hasFnAttribute("amdgpu-no-workgroup-id-x"))
      MFI->addWorkGroupIDX();
    if (!F->hasFnAttribute("amdgpu-no-workgroup-id-y"))
      MFI->addWorkGroupIDY();
    if (!F->hasFnAttribute("amdgpu-no-workgroup-id-z"))
      MFI->addWorkGroupIDZ();
  }

  /// Work-group info is an ordinary system SGPR on every target, allocated in
  /// HSA order right after the workgroup IDs. Its KD bit maps directly to the
  /// backend's \c hasWorkGroupInfo() (AsmPrinter sets TGID_SIZE_EN from it).
  if (AMDHSA_BITS_GET(
          KDOnHost.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO))
    MFI->addWorkGroupInfo();

  /// The private-segment wave byte offset is the last system SGPR. The backend
  /// allocates it for every entry function on targets WITHOUT architected flat
  /// scratch and never on architected ones (see SIMachineFunctionInfo's ctor,
  /// gated on \c !hasArchitectedFlatScratch()). It is NOT gated on the
  /// \c COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT bit — that bit is an
  /// architected-flat-scratch-only "uses scratch" flag (the
  /// \c .amdhsa_enable_private_segment directive is rejected on non-architected
  /// targets), so it would be set exactly where the wave-offset SGPR does NOT
  /// exist. Mirror the backend's predicate directly.
  if (!ST.hasArchitectedFlatScratch())
    MFI->addPrivateSegmentWaveByteOffset();

  /// Kernel functions are 2^8 byte aligned
  MF.setAlignment(llvm::Align(256));

  /// We don't model implicit args separately: the kernarg buffer that
  /// the AsmPrinter emits is sized off the single explicit byref param
  /// we synthesized above (\c KD.kernarg_size bytes), which already
  /// covers both the explicit and implicit halves of the original
  /// kernarg layout.
  F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");

  /// GRANULATED_WAVEFRONT_SGPR_COUNT is automatically calculated via the
  /// resouce usage analysis pass, but we add its count as an attribute
  /// to the function for later use
  uint32_t NextFreeSGPR = [&] {
    if (!llvm::AMDGPU::isGFX10Plus(ST)) {
      uint32_t GranulatedWavefrontSGPRCount = AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc1,
          llvm::amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
      return (GranulatedWavefrontSGPRCount + 1) * ST.getSGPREncodingGranule();
    } else {
      return ST.getAddressableNumSGPRs() + 2;
    }
  }();
  F->addFnAttr("amdgpu-num-sgpr", llvm::formatv("{0}", NextFreeSGPR).str());

  return std::make_pair(std::ref(MF), KDSymbolIfPresent);
}

llvm::Expected<std::pair<llvm::MachineFunction &,
                         std::optional<object::AMDGCNElfSymbolRef>>>
initLiftedDeviceFunctionEntry(uint64_t DeviceEntryPointAddr,
                              const MemoryAllocationAccessor &SegAccessor,
                              llvm::Module &TargetModule,
                              const llvm::Function &InitialExecutionPoint,
                              llvm::FunctionAnalysisManager &FAM,
                              llvm::FunctionType *DevFuncTy) {
  auto SegmentOrErr = SegAccessor.getAllocationDescriptor(DeviceEntryPointAddr);
  LUTHIER_RETURN_ON_ERROR(SegmentOrErr.takeError());
  if (SegmentOrErr->empty())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Failed to find the allocation associated with address {0:x}",
        DeviceEntryPointAddr));

  uint64_t DevFuncLoadOffset =
      DeviceEntryPointAddr -
      reinterpret_cast<uint64_t>(SegmentOrErr->getDeviceAllocation().data());

  /// Locate the symbol this entry address is part of
  std::optional<object::AMDGCNElfSymbolRef> FuncSymRef{std::nullopt};
  uint64_t EntryFromStartOfSymbol{0};

  if (auto *ObjFile = SegmentOrErr->getAllocationCodeObject()) {
    for (object::AMDGCNElfSymbolRef Symbol : ObjFile->symbols()) {
      if (Symbol.getELFType() == llvm::ELF::STT_FUNC) {
        llvm::Expected<uint64_t> SymbolAddrOrErr = Symbol.getAddress();
        LUTHIER_RETURN_ON_ERROR(SymbolAddrOrErr.takeError());
        uint64_t SymbolSize = Symbol.getSize();
        /// Half-open range: [SymbolAddr, SymbolAddr + SymbolSize). An offset
        /// exactly at \c SymbolAddr + SymbolSize is the first byte of the next
        /// symbol, not this one. Stop at the first containing symbol.
        if (*SymbolAddrOrErr <= DevFuncLoadOffset &&
            DevFuncLoadOffset < (*SymbolAddrOrErr + SymbolSize)) {
          FuncSymRef = Symbol;
          EntryFromStartOfSymbol = DevFuncLoadOffset - *SymbolAddrOrErr;
          break;
        }
      }
    }
  }

  std::string FuncName;
  if (FuncSymRef.has_value()) {
    LUTHIER_RETURN_ON_ERROR(FuncSymRef->getName().moveInto(FuncName));
    FuncName += llvm::formatv("x{0:x}", EntryFromStartOfSymbol);
  } else {
    FuncName = llvm::formatv("x{0:x}", DeviceEntryPointAddr);
  }

  assert(DevFuncTy != nullptr &&
         "device function prototype has not been initialized");

  llvm::Function *F = llvm::Function::Create(
      DevFuncTy, llvm::GlobalValue::PrivateLinkage, FuncName, TargetModule);
  F->setCallingConv(llvm::CallingConv::C);
  /// Inherit \c amdgpu-num-vgpr / \c amdgpu-num-sgpr from the initial
  /// execution point handle
  assert(InitialExecutionPoint.getCallingConv() ==
             llvm::CallingConv::AMDGPU_KERNEL &&
         "initial execution point is not a kernel");
  unsigned NumVGPRs =
      InitialExecutionPoint.getFnAttributeAsParsedInteger("amdgpu-num-vgpr");
  unsigned NumSGPRs =
      InitialExecutionPoint.getFnAttributeAsParsedInteger("amdgpu-num-sgpr");
  F->addFnAttr("amdgpu-num-vgpr", llvm::formatv("{0}", NumVGPRs).str());
  F->addFnAttr("amdgpu-num-sgpr", llvm::formatv("{0}", NumSGPRs).str());

  llvm::MachineFunction &MF =
      FAM.getResult<llvm::MachineFunctionAnalysis>(*F).getMF();

  MF.setAlignment(llvm::Align(4));
  return std::make_pair(std::ref(MF), FuncSymRef);
}

/// Walks over the \p MCOperands and converts them to \c llvm::MachineOperand
/// instances before adding them to the \c llvm::MachineInstr managed
/// by the \p MIBuilder
/// \note Does not convert direct branch target immediate operands to a machine
/// basic block
/// \return \c llvm::Error indicating the success or failure of the operation
static llvm::Error
convertAndAddMCOperandsToMI(llvm::ArrayRef<llvm::MCOperand> MCOperands,
                            llvm::MachineInstrBuilder &MIBuilder) {
  const unsigned Opcode = MIBuilder->getOpcode();
  const llvm::MCInstrDesc &MCID = MIBuilder->getDesc();
  llvm::MachineBasicBlock *MBB = MIBuilder->getParent();
  assert(MBB && "MI is not part of a machine basic block");
  llvm::MachineFunction *MF = MBB->getParent();
  assert(MF && "MI is not part of a machine function");
  const bool IsDirectBranch =
      MIBuilder->isBranch() && !MIBuilder->isIndirectBranch();
  for (auto [MCOpIdx, MCOp] : llvm::enumerate(MCOperands)) {
    if (MCOp.isReg()) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[CodeDiscoveryPass] Converting MC register operand "
                 << llvm::printReg(MCOp.getReg(),
                                   MF->getSubtarget().getRegisterInfo())
                 << "\n");
      unsigned RegNum = RealToPseudoRegisterMapTable(MCOp.getReg());
      const bool IsDef = MCOpIdx < MCID.getNumDefs();
      llvm::RegState Flags = llvm::RegState::NoFlags;
      const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[MCOpIdx];
      if (IsDef && !OpInfo.isOptionalDef()) {
        Flags |= llvm::RegState::Define;
      }
      LLVM_DEBUG(luthier::dbgs()
                     << "[CodeDiscoveryPass] Adding pseudo register "
                     << llvm::printReg(RegNum,
                                       MF->getSubtarget().getRegisterInfo())
                     << llvm::formatv(" with flags 0x{0:X}\n", Flags););
      (void)MIBuilder.addReg(RegNum, Flags);
    } else if (MCOp.isImm()) {
      LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                     "[CodeDiscoveryPass] Resolving immediate operand: {0}\n",
                     MCOp.getImm()));

      if (!IsDirectBranch) {
        LLVM_DEBUG(
            luthier::dbgs()
            << "[CodeDiscoveryPass] Not a direct branch, adding immediate "
               "directly to instruction\n");
        /// SOPK needs some special attention to be converted correctly
        if (llvm::SIInstrInfo::isSOPK(*MIBuilder)) {
          LLVM_DEBUG(luthier::dbgs()
                     << "[CodeDiscoveryPass] Instruction is in SOPK format\n");
          if (llvm::SIInstrInfo::sopkIsZext(Opcode)) {
            auto Imm = static_cast<uint16_t>(MCOp.getImm());
            LLVM_DEBUG(
                luthier::dbgs() << llvm::formatv(
                    "[CodeDiscoveryPass] Adding truncated imm (zext): {0}\n",
                    Imm));
            (void)MIBuilder.addImm(Imm);
          } else {
            auto Imm = static_cast<int16_t>(MCOp.getImm());
            LLVM_DEBUG(
                luthier::dbgs() << llvm::formatv(
                    "[CodeDiscoveryPass] Adding truncated imm (sex): {0}\n",
                    Imm));
            (void)MIBuilder.addImm(Imm);
          }
        } else {
          LLVM_DEBUG(luthier::dbgs()
                     << llvm::formatv("[CodeDiscoveryPass] Adding Imm: {0}\n",
                                      MCOp.getImm()));
          (void)MIBuilder.addImm(MCOp.getImm());
        }
      }
    } else if (MCOp.isExpr() && llvm::isa<llvm::AMDGPUMCExpr>(MCOp.getExpr())) {
      const auto *TargetExpr = llvm::cast<llvm::AMDGPUMCExpr>(MCOp.getExpr());

      switch (TargetExpr->getKind()) {
      case llvm::AMDGPUMCExpr::AGVK_Lit:
      case llvm::AMDGPUMCExpr::AGVK_Lit64:
        LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
            TargetExpr->getArgs().size() == 1 &&
                llvm::isa<llvm::MCConstantExpr>(TargetExpr->getSubExpr(0)),
            "Literal expr operands should only have one constant sub "
            "expression"));
        (void)MIBuilder.addImm(
            llvm::cast<llvm::MCConstantExpr>(TargetExpr->getSubExpr(0))
                ->getValue());
        break;
      default:
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "Unexpected expression type: {0}", TargetExpr->getKind()));
      }
    } else {
      return LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Unexpected MC operand: {0}", MCOp));
    }
  }
  // Create a (fake) memory operand to keep the machine verifier happy
  // when encountering image instructions
  if (llvm::SIInstrInfo::isImage(*MIBuilder)) {
    llvm::MachinePointerInfo PtrInfo =
        llvm::MachinePointerInfo::getConstantPool(*MF);
    llvm::MachineMemOperand *MMO = MF->getMachineMemOperand(
        PtrInfo,
        MCID.mayLoad() ? llvm::MachineMemOperand::MOLoad
                       : llvm::MachineMemOperand::MOStore,
        16, llvm::Align(8));
    MIBuilder->addMemOperand(*MF, MMO);
  }

  if (size_t NumMCOps = MCOperands.size(); NumMCOps < MCID.NumOperands) {
    LLVM_DEBUG(luthier::dbgs() << "[CodeDiscoveryPass] Must fixup instruction ";
               MIBuilder->print(luthier::dbgs()); luthier::dbgs() << "\n";
               luthier::dbgs()
               << "[CodeDiscoveryPass] Num explicit operands added: "
               << NumMCOps << ", "
               << "expected: " << MCID.NumOperands << "\n");
    // Loop over missing explicit operands (if any) and synthesize them
    //
    // Two kinds need fixup here:
    //  1. Immediate slots (e.g. AMDGPU `op_sel` / `clamp` / `omod` and other
    //     defaulted-to-zero modifier immediates that aren't part of the asm
    //     syntax). We emit a 0 immediate
    //  2. Tied-input register slots (e.g. FLAT_LOAD_*_D16 / _D16_HI / _t16
    //     all carry a `$vdst_in` tied to `$vdst`; VOP3 `src2_modifiers` ties
    //     similarly; many AGPR/D16 partial-write atomics tie `$vdst_in` to
    //     `$vdst`). MCInst never contains tied operands; we synthesize them
    //     from the def at the tied-to index and call `tieOperands`
    //
    // Other operand kinds (e.g. expressions) are not synthesizable from this
    // limited context; we leave them unfilled for later passes
    for (unsigned int MissingExpOpIdx = NumMCOps;
         MissingExpOpIdx < MCID.NumOperands; MissingExpOpIdx++) {
      LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                     "[CodeDiscoveryPass] Fixing up operand index {0}\n",
                     MissingExpOpIdx));
      auto OpType = MCID.operands()[MissingExpOpIdx].OperandType;
      if (OpType == llvm::MCOI::OPERAND_IMMEDIATE ||
          OpType == llvm::AMDGPU::OPERAND_KIMM32 ||
          // Untyped immediate slots (e.g. the defaulted-to-zero i1 $IsAsync
          // flag on legacy LDS-DMA loads) report OPERAND_UNKNOWN with no
          // register class
          (OpType == llvm::MCOI::OPERAND_UNKNOWN &&
           MCID.operands()[MissingExpOpIdx].RegClass == -1)) {
        LLVM_DEBUG(luthier::dbgs()
                       << "[CodeDiscoveryPass] Adding 0-immediate for "
                          "missing operand\n";);
        (void)MIBuilder.addImm(0);
        continue;
      }
      // Tied input: the missing slot is tied to an earlier def — read the
      // same physical register the def writes (live-in value)
      // `MachineInstr::addOperand` auto-ties uses based on the MCID's
      // TIED_TO constraint, so just adding the register operand sets up
      // the tie — no explicit `tieOperands` call needed (calling it would
      // trip the "Def is already tied" assertion)
      int TiedToIdx =
          MCID.getOperandConstraint(MissingExpOpIdx, llvm::MCOI::TIED_TO);
      if (TiedToIdx >= 0 &&
          static_cast<unsigned>(TiedToIdx) < MIBuilder->getNumOperands()) {
        const llvm::MachineOperand &Def = MIBuilder->getOperand(TiedToIdx);
        if (Def.isReg()) {
          LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                         "[CodeDiscoveryPass] Synthesizing tied use of {0} at "
                         "operand index {1} (tied to def index {2})\n",
                         llvm::printReg(Def.getReg(),
                                        MF->getSubtarget().getRegisterInfo()),
                         MissingExpOpIdx, TiedToIdx));
          (void)MIBuilder.addReg(Def.getReg(), llvm::RegState::NoFlags);
          continue;
        }
      }
      // Singleton register class (e.g. the explicit $m0 slot on LDS-DMA
      // loads): the slot can only name one register, which is never encoded
      // in the MCInst — synthesize it directly.
      if (int16_t RC = MCID.operands()[MissingExpOpIdx].RegClass; RC != -1) {
        const llvm::TargetRegisterClass &TRC =
            *MF->getSubtarget().getRegisterInfo()->getRegClass(RC);
        if (TRC.getNumRegs() == 1) {
          LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                         "[CodeDiscoveryPass] Synthesizing singleton register "
                         "{0} at operand index {1}\n",
                         llvm::printReg(TRC.getRegister(0),
                                        MF->getSubtarget().getRegisterInfo()),
                         MissingExpOpIdx));
          (void)MIBuilder.addReg(TRC.getRegister(0));
          continue;
        }
      }
      LLVM_DEBUG(luthier::dbgs() << "[CodeDiscoveryPass] Unhandled missing "
                                    "operand kind; leaving unfilled\n");
    }
  }

  /// Add implicit use of the execute mask if it's not already reflected in
  /// the machine instruction
  if (MCID.hasImplicitUseOfPhysReg(llvm::AMDGPU::EXEC) &&
      !MIBuilder->hasRegisterImplicitUseOperand(llvm::AMDGPU::EXEC)) {
    MIBuilder->addOperand(
        llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
  }

  return llvm::Error::success();
}

static llvm::Error
populateMF(const InstructionTraces &MFTrace, llvm::MachineFunction &MF,
           llvm::SmallVector<llvm::MachineInstr *> &UnresolvedReturnInsts) {
  llvm::LLVMContext &Ctx = MF.getFunction().getContext();
  llvm::MachineBasicBlock *CurrentMBB = MF.CreateMachineBasicBlock();

  const auto &TM =
      *static_cast<const llvm::GCNTargetMachine *>(&MF.getTarget());

  MF.push_back(CurrentMBB);

  const llvm::MCRegisterInfo &MRI = TM.getMCRegisterInfo();

  std::unique_ptr<llvm::MCInstPrinter> IP(TM.getTarget().createMCInstPrinter(
      TM.getTargetTriple(), TM.getMCAsmInfo().getAssemblerDialect(),
      TM.getMCAsmInfo(), *TM.getMCInstrInfo(), MRI));

  llvm::MCContext &MCContext = MF.getContext();

  const llvm::MCInstrInfo &MCInstInfo = *TM.getMCInstrInfo();
  std::unique_ptr<llvm::MCInstrAnalysis> MIA(
      TM.getTarget().createMCInstrAnalysis(&MCInstInfo));

  llvm::DenseMap<uint64_t,
                 llvm::SmallVector<llvm::MachineInstr *>>
      UnresolvedBranchMIs; // < Set of branch instructions located at a
                           // device address waiting for their
                           // target to be resolved after MBBs and MIs
                           // are created
  llvm::DenseMap<uint64_t, llvm::MachineBasicBlock *>
      BranchTargetMBBs; // < Set of MBBs that will be the target of the
                        // UnresolvedBranchMIs

  /// The start address of the first trace
  const uint64_t FirstTraceStartAddr = MFTrace.traces_begin()->first.first;
  /// The end address of the last trace
  const uint64_t LastTraceEndAddr = MFTrace.traces_rbegin()->first.second;

  const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  llvm::MachineInstr *EntryInst{nullptr};

  for (const auto &[TraceStartEndAddr, Trace] : MFTrace.traces()) {
    uint64_t CurrentInstrAddr = TraceStartEndAddr.first;
    uint64_t LastTraceInstrAddr = TraceStartEndAddr.second;
    LLVM_DEBUG(luthier::dbgs()
               << llvm::formatv(
                      "\n[CodeDiscoveryPass] Processing trace [{0:x}, {1:x}]",
                      TraceStartEndAddr.first, TraceStartEndAddr.second)
               << " with " << Trace->size() << " instructions\n");

    while (CurrentInstrAddr <= LastTraceInstrAddr) {
      auto InstIt = Trace->find(CurrentInstrAddr);
      if (InstIt == Trace->end())
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "Trace has no instruction at address {0:x}.", CurrentInstrAddr));
      const TraceInstr &Inst = InstIt->second;
      auto MCInst = Inst.getMCInst();
      const unsigned Opcode = getPseudoOpcodeFromReal(MCInst.getOpcode());
      const llvm::MCInstrDesc &MCID = MCInstInfo.get(Opcode);
      bool IsIndirectBranch = MCID.isIndirectBranch();
      bool IsDirectBranch = MCID.isBranch() && !IsIndirectBranch;
      bool IsDirectBranchTarget =
          MFTrace.isAddressDirectBranchTarget(Inst.getLoadedDeviceAddress());
      LLVM_DEBUG(luthier::dbgs()
                     << "[CodeDiscoveryPass] Processing MCInst at "
                     << llvm::formatv("{0:x}:\n",
                                      Inst.getLoadedDeviceAddress());
                 MCInst.dump_pretty(luthier::dbgs(), IP.get(), " ", &MCContext);
                 luthier::dbgs() << llvm::formatv(
                     "\n  Opcode: {0:x}, IsBranch={1}, IsIndirectBranch={2}\n",
                     Opcode, MCID.isBranch(), IsIndirectBranch););

      if (IsDirectBranchTarget) {
        LLVM_DEBUG(luthier::dbgs()
                   << "[CodeDiscoveryPass] Instruction is a branch target\n");
        if (!CurrentMBB->empty()) {
          LLVM_DEBUG(luthier::dbgs()
                     << "[CodeDiscoveryPass] Current MBB is not "
                        "empty, creating new MBB\n");
          llvm::MachineBasicBlock *OldMBB = CurrentMBB;
          CurrentMBB = MF.CreateMachineBasicBlock();
          MF.push_back(CurrentMBB);
          OldMBB->addSuccessor(CurrentMBB);

        } else {
          LLVM_DEBUG(
              luthier::dbgs()
              << "[CodeDiscoveryPass] Current MBB is empty, using existing\n");
        }
        BranchTargetMBBs.insert({Inst.getLoadedDeviceAddress(), CurrentMBB});
        LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                       "[CodeDiscoveryPass] New MBB created at "
                       "address {0:x}, MBB idx {1}\n",
                       Inst.getLoadedDeviceAddress(), CurrentMBB->getNumber()));
      }
      llvm::MachineInstrBuilder Builder =
          llvm::BuildMI(CurrentMBB, llvm::DebugLoc(), MCID);
      auto MDNodeOrErr = TargetMachineInstrMDNode::initializeMDNode(*Builder);
      LUTHIER_RETURN_ON_ERROR(MDNodeOrErr.takeError());
      MDNodeOrErr->setTraceInstrAddress(Ctx, CurrentInstrAddr);

      LLVM_DEBUG(
          luthier::dbgs() << llvm::formatv(
              "[CodeDiscoveryPass] Populating {0} operands for instruction\n",
              MCID.operands().size()));
      LUTHIER_RETURN_ON_ERROR(
          convertAndAddMCOperandsToMI(MCInst.getOperands(), Builder));

      LLVM_DEBUG(luthier::dbgs() << "[CodeDiscoveryPass] Built instruction: ";
                 Builder->print(luthier::dbgs()); luthier::dbgs() << "\n");
      if (Inst.getLoadedDeviceAddress() ==
          MFTrace.getInitialEntryPoint().getEntryPointAddress()) {
        EntryInst = Builder.getInstr();
      }
      // Basic Block resolving; We also split blocks further down to "vector"
      // and scalar block to make it easier to deal with predication calculation
      if (MCID.isTerminator()) {
        LLVM_DEBUG(luthier::dbgs()
                   << "[CodeDiscoveryPass] Instruction is a terminator\n");
        if (IsDirectBranch) {
          LLVM_DEBUG(luthier::dbgs()
                     << "[CodeDiscoveryPass] Direct branch terminator\n");
          uint64_t BranchTarget;
          LUTHIER_RETURN_ON_ERROR(
              InstructionTracesAnalysis::evaluateDirectBranchOrCall(
                  MCInst, Inst.getLoadedDeviceAddress())
                  .moveInto(BranchTarget));
          LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                         "[CodeDiscoveryPass] Branch target resolved: {0:x}\n",
                         BranchTarget));
          if (!UnresolvedBranchMIs.contains(BranchTarget)) {
            UnresolvedBranchMIs.insert({BranchTarget, {Builder.getInstr()}});
          } else {
            UnresolvedBranchMIs[BranchTarget].push_back(Builder.getInstr());
          }
        }
        // if this is the last instruction in the trace group, no need for
        // creating a new basic block
        if (CurrentInstrAddr != LastTraceEndAddr) {
          LLVM_DEBUG(
              luthier::dbgs()
              << "[CodeDiscoveryPass] Creating new MBB after terminator\n");
          llvm::MachineBasicBlock *OldMBB = CurrentMBB;
          CurrentMBB = MF.CreateMachineBasicBlock();
          MF.push_back(CurrentMBB);
          // Don't add the next block to the list of successors if the
          // terminator is an unconditional branch
          if (!MCID.isUnconditionalBranch())
            OldMBB->addSuccessor(CurrentMBB);
          LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                         "[CodeDiscoveryPass] New MBB idx {0} created\n",
                         CurrentMBB->getNumber()));
        }
      } else if (llvm::MachineInstr *PrevMI = Builder->getPrevNode()) {
        bool IsCurrentMIVector = shouldImplicitReadExec(*Builder);
        bool IsFormerMIVector = shouldImplicitReadExec(*PrevMI);
        bool CurrentMIWritesExecMask =
            Builder->modifiesRegister(llvm::AMDGPU::EXEC, TRI);
        bool ShouldSplitCurrentMBB =
            CurrentMIWritesExecMask || IsCurrentMIVector ^ IsFormerMIVector;
        if (ShouldSplitCurrentMBB) {
          LLVM_DEBUG(luthier::dbgs() << "[CodeDiscoveryPass] Splitting MBB for "
                                        "vector/scalar transition\n");
          llvm::MachineBasicBlock *OldMBB = CurrentMBB;
          CurrentMBB = OldMBB->splitAt(*PrevMI, false);
        }
      }
      /// Indirect branch and all call targets require further processing so
      /// we return them to the code discovery pass
      if (Builder->isIndirectBranch() || Builder->isCall()) {
        UnresolvedReturnInsts.push_back(Builder.getInstr());
      }

      CurrentInstrAddr += Inst.getSize();
    }
  }
  // Resolve the direct branch and target MIs/MBBs
  LLVM_DEBUG(luthier::dbgs()
             << "[CodeDiscoveryPass] Resolving " << UnresolvedBranchMIs.size()
             << " direct branch MIs\n");
  for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
    LLVM_DEBUG(
        luthier::dbgs() << llvm::formatv(
            "[CodeDiscoveryPass] Resolving {0} branches to target {1:x}\n",
            BranchMIs.size(), TargetAddress));
    CurrentMBB = BranchTargetMBBs[TargetAddress];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        CurrentMBB != nullptr,
        llvm::formatv("Failed to find the MachineBasicBlock associated with "
                      "the branch target address {0:x}.",
                      TargetAddress)));
    for (auto &MI : BranchMIs) {
      LLVM_DEBUG(luthier::dbgs() << "[CodeDiscoveryPass] Resolving branch: ";
                 MI->print(luthier::dbgs()); luthier::dbgs() << "\n");
      MI->addOperand(llvm::MachineOperand::CreateMBB(CurrentMBB));
      MI->getParent()->addSuccessor(CurrentMBB);
      LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                     "[CodeDiscoveryPass] MBB {0} set as branch target\n",
                     CurrentMBB->getNumber()));
    }
  }

  /// If the entry address of the trace group is not equal to the address of the
  /// first trace then add a new basic block in the beginning of the MF,
  /// put a jump to the entry instruction
  if (MFTrace.getInitialEntryPoint().getEntryPointAddress() !=
      FirstTraceStartAddr) {
    if (!EntryInst)
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Code discovery could not locate the entry-point instruction at "
          "address {0:x} within the discovered trace.",
          MFTrace.getInitialEntryPoint().getEntryPointAddress()));
    llvm::MachineBasicBlock *EntryMBB = MF.CreateMachineBasicBlock();
    MF.push_front(EntryMBB);

    llvm::MachineInstrBuilder Builder = llvm::BuildMI(
        EntryMBB, llvm::DebugLoc(), MCInstInfo.get(llvm::AMDGPU::S_BRANCH));
    auto MDNodeOrErr = TargetMachineInstrMDNode::initializeMDNode(*Builder);
    LUTHIER_RETURN_ON_ERROR(MDNodeOrErr.takeError());
    MDNodeOrErr->setCanRelaxDirectBranch(Ctx, true);
    Builder->addOperand(
        llvm::MachineOperand::CreateMBB(EntryInst->getParent()));
    EntryMBB->addSuccessor(EntryInst->getParent());
  }

  /// LLVM IR/MIR convention requires the entry block to have no predecessors.
  /// If the current first MBB does (e.g. the entry address sits at the head of
  /// a loop and some later MBB branches back to it), splice a synthetic
  /// preheader at the front whose only job is to unconditionally branch into
  /// the original entry. The new S_BRANCH carries the same metadata shape as
  /// the entry-in-middle synthesis above: \c canRelaxDirectBranch=true and no
  /// trace-instruction address, so downstream passes recognize it as
  /// non-trace.
  if (!MF.empty() && MF.front().pred_size() != 0) {
    llvm::MachineBasicBlock *OriginalEntry = &MF.front();
    llvm::MachineBasicBlock *NewEntry = MF.CreateMachineBasicBlock();
    MF.push_front(NewEntry);

    llvm::MachineInstrBuilder Builder = llvm::BuildMI(
        NewEntry, llvm::DebugLoc(), MCInstInfo.get(llvm::AMDGPU::S_BRANCH));
    auto MDNodeOrErr = TargetMachineInstrMDNode::initializeMDNode(*Builder);
    LUTHIER_RETURN_ON_ERROR(MDNodeOrErr.takeError());
    MDNodeOrErr->setCanRelaxDirectBranch(Ctx, true);
    Builder->addOperand(llvm::MachineOperand::CreateMBB(OriginalEntry));
    NewEntry->addSuccessor(OriginalEntry);
  }

  /// Freeze the set of reserved register because we will not do any register
  /// allocations here
  MF.getRegInfo().freezeReservedRegs();

  /// Populate the properties of MF
  llvm::MachineFunctionProperties &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Properties.reset(llvm::MachineFunctionProperties::Property::TracksLiveness);
  Properties.set(llvm::MachineFunctionProperties::Property::Selected);

  LLVM_DEBUG(luthier::dbgs()
                 << "\n[CodeDiscoveryPass] Machine function '" << MF.getName()
                 << "' created with " << MF.size() << " MBBs\n";);

  return llvm::Error::success();
}

llvm::PreservedAnalyses
CodeDiscoveryPass::run(llvm::Module &TargetModule,
                       llvm::ModuleAnalysisManager &TargetMAM) {
  llvm::LLVMContext &Ctx = TargetModule.getContext();

  LLVM_DEBUG(luthier::dbgs()
                 << "[CodeDiscoveryPass] Running code discovery pass\n";);

  llvm::MachineModuleInfo &TargetMMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetModule).getMMI();
  auto &TM =
      *static_cast<const llvm::GCNTargetMachine *>(&TargetMMI.getTarget());

  const MemoryAllocationAccessor &SegAccessor =
      TargetMAM.getResult<MemoryAllocationAnalysis>(TargetModule).getAccessor();

  llvm::MachineFunctionAnalysisManager &MFAM =
      TargetMAM
          .getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(
              TargetModule)
          .getManager();
  llvm::FunctionAnalysisManager &FAM =
      TargetMAM
          .getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();

  EntryPoint InitialEntryPoint =
      TargetMAM.getResult<InitialEntryPointAnalysis>(TargetModule)
          .getInitialEntryPoint();

  const llvm::amdhsa::kernel_descriptor_t &InitialExecutionPoint =
      TargetMAM.getResult<InitialExecutionPointAnalysis>(TargetModule)
          .getInitialExecutionPoint();
  /// Initialize the handle for the initial execution point (i.e. the entry
  /// point used to launch the currently running/about to be launched
  /// shared/kernel)
  auto InitialExecPointMFAndSymbol = initKernelEntryPointFunction(
      InitialExecutionPoint, SegAccessor, TargetModule, TM, FAM);
  if (llvm::Error Err = InitialExecPointMFAndSymbol.takeError()) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    /// initKernelEntryPointFunction may have created a partial entry-point
    /// Function/MF in the module before failing, so nothing is preserved.
    return llvm::PreservedAnalyses::none();
  }

  InitialExecPointMFAndSymbol->first.getFunction().addFnAttr(
      InitialExecutionPointAttr);

  /// Start of the code discovery loop

  llvm::SmallDenseSet<EntryPoint> UnvisitedPointsOfEntry{InitialEntryPoint};

  llvm::SmallDenseSet<EntryPoint> VisitedPointsOfEntry{};

  /// Prototype of the device functions, computed up front via the stateless
  /// MIRToIRTranslator factory so we don't construct (and re-construct) a
  /// translator over the initial entry-point MF before the worklist loop.
  const llvm::MachineFunction &InitialExecPointMF =
      InitialExecPointMFAndSymbol->first;
  const llvm::Function &InitialExecPointFn = InitialExecPointMF.getFunction();
  llvm::Expected<llvm::FunctionType *> DeviceFuncPrototypeOrErr =
      MIRToIRTranslator::computeStandardDeviceFunctionType(
          Ctx, InitialExecPointMF.getSubtarget<llvm::GCNSubtarget>(),
          InitialExecPointFn.getFnAttributeAsParsedInteger("amdgpu-num-sgpr"),
          InitialExecPointFn.getFnAttributeAsParsedInteger("amdgpu-num-vgpr"));
  if (llvm::Error Err = DeviceFuncPrototypeOrErr.takeError()) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    /// The initial entry-point Function was already attributed above
    /// (InitialExecutionPointAttr), so the module is already mutated.
    return llvm::PreservedAnalyses::none();
  }

  /// Collected across iterations. For each S_CALL_B64 encountered, the MI
  /// initially has a raw 16-bit signed displacement (\c MO_Immediate) as
  /// operand 1. After the main discovery loop completes (all callees
  /// materialized), we walk this list and rewrite each MI's operand 1 to
  /// \c MO_GlobalAddress pointing at the resolved callee \c Function*.
  llvm::SmallVector<std::pair<llvm::MachineInstr *, uint64_t>>
      UnresolvedShortCallInsts{};

  while (!UnvisitedPointsOfEntry.empty()) {
    EntryPoint CurrentEntryPoint = *UnvisitedPointsOfEntry.begin();

    /// Early exit if we've already visited the entry point
    if (VisitedPointsOfEntry.contains(CurrentEntryPoint)) {
      UnvisitedPointsOfEntry.erase(CurrentEntryPoint);
      continue;
    }

    LLVM_DEBUG(luthier::dbgs()
               << "\n[CodeDiscoveryPass] Processing entry point at address "
               << llvm::formatv("{0:x}\n",
                                CurrentEntryPoint.getEntryPointAddress()));

    /// Initialize the function handle associated with the entry point
    const auto *KDOnDevice = CurrentEntryPoint.getKernelDescriptor();

    auto MFAndFuncSymRefOrErr = [&]() -> decltype(InitialExecPointMFAndSymbol) {
      if (KDOnDevice) {
        if (KDOnDevice != &InitialExecutionPoint)
          return LUTHIER_MAKE_GENERIC_ERROR(
              "Initial execution point does not "
              "match the kernel initial entry point");
        return *InitialExecPointMFAndSymbol;
      } else {
        return initLiftedDeviceFunctionEntry(
            CurrentEntryPoint.getEntryPointAddress(), SegAccessor, TargetModule,
            InitialExecPointMFAndSymbol->first.getFunction(), FAM,
            *DeviceFuncPrototypeOrErr);
      }
    }();

    if (!MFAndFuncSymRefOrErr) {
      Ctx.emitError(llvm::toString(MFAndFuncSymRefOrErr.takeError()));
      /// Prior iterations and initLiftedDeviceFunctionEntry may have already
      /// created Functions/MFs in the module, so nothing is preserved.
      return llvm::PreservedAnalyses::none();
    }
    [[maybe_unused]] auto [MF, FuncSymRef] = *MFAndFuncSymRefOrErr;

    /// Set the function's entry point as an attribute
    setFunctionEntryPoint(MF.getFunction(), CurrentEntryPoint);

    // assign elfsymbol attribute to machine function
    if (FuncSymRef.has_value()) {
      MF.getFunction().addFnAttr(ElfSymbolAttr);
    }

    if (CurrentEntryPoint == InitialEntryPoint) {
      MF.getFunction().addFnAttr(InitialEntryPointAttr);
    }

    /// Add the newly created MF's entry point
    /// Ask for the trace of the instructions for this machine function
    auto &TraceResults = MFAM.getResult<InstructionTracesAnalysis>(MF);
    if (!TraceResults.getTraces()) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Failed to get trace results for function {0}", MF.getName()))));
      /// The MF for this entry point was already created and attributed
      /// (setFunctionEntryPoint), so the module is already mutated.
      return llvm::PreservedAnalyses::none();
    }
    llvm::SmallVector<llvm::MachineInstr *> TraceTermInstructions{};
    /// Populate the current machine function using the trace info we just
    /// obtained
    if (llvm::Error Err =
            populateMF(*TraceResults.getTraces(), MF, TraceTermInstructions)) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      /// populateMF builds MIR into MF and may have partially populated it
      /// before failing, so nothing is preserved.
      return llvm::PreservedAnalyses::none();
    }

    /// Invalidate all module-level analysis not related to Functions and
    /// Machine Functions proxies because we just added a new machine function
    /// and they are now stale
    llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
    PA.preserve<llvm::MachineFunctionAnalysisManagerModuleProxy>();
    PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
    TargetMAM.invalidate(TargetModule, PA);

    llvm::Error Err = llvm::Error::success();

    /// Translate the machine function to LLVM IR
    MIRToIRTranslator Translator{MF, Err};

    if (Err) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      /// MF is fully populated with MIR and the translator ctor may have
      /// started emitting IR, so nothing is preserved.
      return llvm::PreservedAnalyses::none();
    }

    Translator.translate();

    /// Go over all discovered call target addresses and add them to be visited
    /// (if not visited already)
    const TraceCallGraph &CG =
        TargetMAM.getResult<TraceCallGraphAnalysis>(TargetModule);
    for (uint64_t Addr : CG.discovered_addrs()) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[CodeDiscoveryPass] Callgraph discovered target 0x"
                 << llvm::utohexstr(Addr) << "\n");
      if (EntryPoint DiscoveredEP{Addr};
          !VisitedPointsOfEntry.contains(DiscoveredEP))
        UnvisitedPointsOfEntry.insert(DiscoveredEP);
    }

    LLVM_DEBUG(luthier::dbgs()
               << "[CodeDiscoveryPass] Processing "
               << TraceTermInstructions.size() << " trace terminators\n");

    const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    for (llvm::MachineInstr *TraceTermMI : TraceTermInstructions) {
      /// If a terminator is a call, then it is likely that the instruction
      /// after the call is reachable. Add it to the list of unvisited entry
      /// points if it is a trace instruction
      if (TraceTermMI->isCall() && Opts.EagerDiscoverCallReturnEntryPoint) {
        TargetMachineInstrMDNode *TraceTermMD =
            TargetMachineInstrMDNode::getInstrMDNodeIfExists(*TraceTermMI);
        if (!TraceTermMD)
          continue;
        std::optional<uint64_t> TraceTermAddr =
            TraceTermMD->getTraceInstrAddress();
        if (!TraceTermAddr.has_value())
          continue;
        UnvisitedPointsOfEntry.insert(
            EntryPoint{*TraceTermAddr + TII->getInstSizeInBytes(*TraceTermMI)});

        if (TraceTermMI->getOpcode() == llvm::AMDGPU::S_CALL_B64) {
          uint64_t CallTarget =
              InstructionTracesAnalysis::evaluateDirectBranchOrCall(
                  TraceTermMI->getOperand(1).getImm(), *TraceTermAddr);
          UnresolvedShortCallInsts.emplace_back(TraceTermMI, CallTarget);
          LLVM_DEBUG(luthier::dbgs()
                     << "[CodeDiscoveryPass] Direct call found, target: "
                     << llvm::formatv("{0:x}\n", CallTarget));
        }
      }
    }

    UnvisitedPointsOfEntry.erase(CurrentEntryPoint);
    VisitedPointsOfEntry.insert(CurrentEntryPoint);
  }

  /// Rewrite each S_CALL_B64's $simm16 operand from the raw 16-bit
  /// displacement immediate to the resolved callee Function
  llvm::DenseMap<uint64_t, llvm::Function *> AddrToFunction;
  for (llvm::Function &F : TargetModule) {
    if (auto EP = getFunctionEntryPoint(F))
      AddrToFunction[EP->getEntryPointAddress()] = &F;
  }
  for (auto [MI, TargetAddr] : UnresolvedShortCallInsts) {
    auto It = AddrToFunction.find(TargetAddr);
    if (It == AddrToFunction.end()) {
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("[CodeDiscoveryPass] S_CALL_B64 target {0:x} did not "
                        "resolve to a Function",
                        TargetAddr))));
      /// All Functions/MFs and their IR have already been built at this
      /// point, so nothing is preserved.
      return llvm::PreservedAnalyses::none();
    }
    // Replace call target with MO_GlobalAddress(F, 0).
    assert(MI->getNumOperands() >= 2 &&
           "S_CALL_B64 must have at least 2 operands");
    MI->getOperand(1).ChangeToGA(It->second, /*Offset=*/0);
    LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                   "[CodeDiscoveryPass] Rewrote S_CALL_B64 target → "
                   "Function '{0}' (addr {1:x})\n",
                   It->second->getName(), TargetAddr));

    /// Replace the original indirect call emitted with a direct call for
    /// S_CALL_B64
    const llvm::MachineBasicBlock *MBB = MI->getParent();
    auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
    /// In cases where the call is not a trace instruction, it might be
    /// optimized away; No need to freak out if we can't find its associated
    /// terminator instruction
    if (!BB)
      continue;
    /// Walk BB backwards from the terminator to find the tail call emitted
    /// by the translation pass. The terminator itself is the trailing
    /// \c unreachable; the CallInst we want sits just before it.
    llvm::CallInst *CallTerm = nullptr;
    for (llvm::Instruction &I : llvm::reverse(*BB)) {
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);
          Call && Call->isTailCall()) {
        CallTerm = Call;
        break;
      }
    }
    if (!CallTerm)
      continue;
    CallTerm->setCalledFunction(It->second);
  }

  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<llvm::MachineFunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::MachineFunctionAnalysis>();
  return PA;
}
} // namespace luthier