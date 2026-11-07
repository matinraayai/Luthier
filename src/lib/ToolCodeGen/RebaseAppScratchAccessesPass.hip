//===-- RebaseAppScratchAccessesPass.hip ----------------------------------===//
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
/// \file
/// Implements \c RebaseAppScratchAccessesPass.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/RebaseAppScratchAccessesPass.h"

#include "luthier/Intrinsic/Intrinsics.h"
#include "luthier/Intrinsic/ScalarValueArgument.h"

#include <SIInstrInfo.h>
#include <hsa/hsa.h>

#ifndef __HIP_DEVICE_COMPILE__
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/PrototypeCallGraph.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#endif // __HIP_DEVICE_COMPILE__

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-rebase-app-scratch-accesses"

//===----------------------------------------------------------------------===//
// Device side
//===----------------------------------------------------------------------===//

LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(
    luthier::RebaseAppScratchAccessesPass::Parser)

namespace {

/// A pointer into the wavefront's private segment. On AMDGPU an
/// <tt>addrspace(5)</tt> pointer *is* the 32-bit per-work-item byte offset the
/// \c SCRATCH_* instructions take, which is exactly the quantity this pass
/// displaces.
template <typename T> using Private = __attribute__((address_space(5))) T;

/// \return the number of bytes at the bottom of this work-item's private
/// segment that belong to Luthier, and therefore the displacement every
/// application access has to carry. Written into its state value array lane by
/// \c TargetModulePatcherPass ; wave-uniform, so this is one
/// \c V_READLANE_B32 .
__attribute__((device)) inline std::uint32_t instrumentationReserve() {
  return luthier::readSVA<std::uint32_t>(
      luthier::WORK_ITEM_INSTRUMENTATION_PRIVATE_SEGMENT_SIZE);
}

/// \return the displaced per-work-item private byte offset of a \c SCRATCH_*
/// access. Forms that lack a VGPR or SGPR offset operand pass zero for it; the
/// ISA adds all three terms inside the swizzle
/// (<tt>SWIZZLE(SGPR_offset + VGPR_offset + INST_OFFSET, TID)</tt>), so they
/// can simply be summed here.
__attribute__((device)) inline std::uint32_t
displacedPrivateOffset(std::uint32_t VOffset, std::uint32_t SOffset,
                       std::int32_t InstOffset) {
  return VOffset + SOffset + static_cast<std::uint32_t>(InstOffset) +
         instrumentationReserve();
}

/// \return \p Offset displaced, given a generic pointer already known to be
/// private.
__attribute__((device)) inline std::uint32_t
displacedPrivateOffsetOf(const void *GenericAddr) {
  auto *P = (Private<const char> *)GenericAddr;
  return static_cast<std::uint32_t>((std::size_t)P) + instrumentationReserve();
}

/// \return the flat (generic) alias of the per-work-item private byte offset
/// \p Offset.
///
/// The private-to-flat conversion is *not* open-coded here. Splicing the
/// aperture in by hand would have to reproduce all three of the sources
/// \c SITargetLowering::getSegmentAperture picks between
/// (SIISelLowering.cpp):
///
///   * \c hasApertureRegs() targets read \c SRC_PRIVATE_BASE --- and must read
///     it as a *64-bit* copy and keep the high half, because the register
///     returns zeroes when used as a 32-bit operand;
///   * code object v5 and later get \c private_base through an implicit
///     kernarg;
///   * v4 loads it from the queue pointer at byte \c 0x44
///     (\c private_segment_aperture_base_hi).
///
/// and would still be wrong on \c hasGloballyAddressableScratch() parts, where
/// that helper asserts \c SRC_PRIVATE_BASE must not be used at all. Letting the
/// backend lower an \c addrspacecast keeps every one of those cases correct by
/// construction, on every target, and is the reason this is a cast rather than
/// a shift and an or.
__attribute__((device)) inline const void *
privateOffsetToFlat(std::uint32_t Offset) {
  return (const void *)(Private<const char> *)(std::size_t)Offset;
}

/// \return the \c S_GETREG_B32 operand that reads all 32 bits of hardware
/// register \p ID .
constexpr std::uint16_t getRegAll32Bits(std::uint16_t ID) {
  return static_cast<std::uint16_t>(ID | (31u << 11));
}

/// \return the base address of this wavefront's scratch space, i.e. its
/// \c FLAT_SCRATCH .
///
/// On targets with architected flat scratch the hardware initializes the base
/// itself: GFX11+ exposes it through \c HW_REG_FLAT_SCR_LO / \c _HI (IDs 20 and
/// 21), while gfx94x has no such hardware registers and keeps it in the
/// \c FLAT_SCRATCH SGPR pair. On targets with absolute flat scratch it is
/// whatever the kernel prologue installed, which the state value array keeps.
__attribute__((device)) inline std::uint64_t wavefrontScratchBase() {
#if defined(__GFX11__) || defined(__GFX12__) || defined(__GFX13__)
  constexpr std::uint16_t HwRegFlatScrLo = 20;
  constexpr std::uint16_t HwRegFlatScrHi = 21;
  const std::uint32_t Lo =
      __builtin_amdgcn_s_getreg(getRegAll32Bits(HwRegFlatScrLo));
  const std::uint32_t Hi =
      __builtin_amdgcn_s_getreg(getRegAll32Bits(HwRegFlatScrHi));
  return static_cast<std::uint64_t>(Lo) |
         (static_cast<std::uint64_t>(Hi) << 32);
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||     \
    defined(__gfx950__) || defined(__gfx9_4_generic__)
  return luthier::readReg<std::uint64_t>(llvm::AMDGPU::FLAT_SCR);
#else
  return luthier::readSVA<std::uint64_t>(luthier::FLAT_SCRATCH);
#endif
}

/// \return the size in bytes of this wavefront's scratch space: the dispatch's
/// per-work-item private segment size times the wavefront size. The dispatch
/// was launched with the instrumentation's reserve included, so this covers
/// both the instrumentation's and the application's part of the segment.
__attribute__((device)) inline std::uint64_t wavefrontScratchSize() {
  const auto *Packet = reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(
      luthier::readSVA<std::uint64_t>(luthier::DISPATCH_PTR));
  return static_cast<std::uint64_t>(Packet->private_segment_size) *
         static_cast<std::uint64_t>(__builtin_amdgcn_wavefrontsize());
}

/// \return \p SOffset adjusted for a buffer access, iff that access lands in
/// this wavefront's scratch space.
///
/// FIXME: How to virtualize global accesses to scratch?
/// Word 1 of a V# carries \c base[47:32] in its low half only; the upper half
/// is stride and swizzle configuration, hence the mask.
///
/// The displacement goes into \c soffset rather than into the offsets inside
/// the swizzle: \c soffset is added *outside* it
/// (<tt>Final = const_base + sgpr_offset + buffer_offset</tt>), so a
/// dword-aligned per-lane displacement of \c N bytes is \c N * index_stride
/// there, and \c index_stride for a scratch buffer is the wavefront size ---
/// which is exactly LLVM's own \c getScratchScaleFactor for a stack that is not
/// on flat scratch.
__attribute__((device)) inline std::uint32_t
displacedBufferSOffset(std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                       std::uint32_t SOffset) {
  const std::uint64_t WaveAddr =
      (static_cast<std::uint64_t>(RsrcLo) |
       (static_cast<std::uint64_t>(RsrcHi & 0xFFFFu) << 32)) +
      static_cast<std::uint64_t>(SOffset);
  const std::uint64_t ScratchBase = wavefrontScratchBase();
  if (WaveAddr >= ScratchBase &&
      WaveAddr - ScratchBase < wavefrontScratchSize())
    SOffset += instrumentationReserve() *
               static_cast<std::uint32_t>(__builtin_amdgcn_wavefrontsize());
  return SOffset;
}

// Vector types for multi-dword inline asm operands.
using v2u32 = std::uint32_t __attribute__((ext_vector_type(2)));
using v3u32 = std::uint32_t __attribute__((ext_vector_type(3)));
using v4u32 = std::uint32_t __attribute__((ext_vector_type(4)));

} // namespace

//===----------------------------------------------------------------------===//
// ISA mnemonic macros
//===----------------------------------------------------------------------===//

#if defined(__GFX12__) || defined(__GFX13__)
#define MN_SCRATCH_LOAD_B32 "scratch_load_b32"
#define MN_SCRATCH_LOAD_B64 "scratch_load_b64"
#define MN_SCRATCH_LOAD_B96 "scratch_load_b96"
#define MN_SCRATCH_LOAD_B128 "scratch_load_b128"
#define MN_SCRATCH_LOAD_U8 "scratch_load_u8"
#define MN_SCRATCH_LOAD_I8 "scratch_load_i8"
#define MN_SCRATCH_LOAD_U16 "scratch_load_u16"
#define MN_SCRATCH_LOAD_I16 "scratch_load_i16"
#define MN_SCRATCH_STORE_B32 "scratch_store_b32"
#define MN_SCRATCH_STORE_B64 "scratch_store_b64"
#define MN_SCRATCH_STORE_B96 "scratch_store_b96"
#define MN_SCRATCH_STORE_B128 "scratch_store_b128"
#define MN_SCRATCH_STORE_B8 "scratch_store_b8"
#define MN_SCRATCH_STORE_B16 "scratch_store_b16"
#define MN_FLAT_LOAD_B32 "flat_load_b32"
#define MN_FLAT_LOAD_B64 "flat_load_b64"
#define MN_FLAT_LOAD_B96 "flat_load_b96"
#define MN_FLAT_LOAD_B128 "flat_load_b128"
#define MN_FLAT_LOAD_U8 "flat_load_u8"
#define MN_FLAT_LOAD_I8 "flat_load_i8"
#define MN_FLAT_LOAD_U16 "flat_load_u16"
#define MN_FLAT_LOAD_I16 "flat_load_i16"
#define MN_FLAT_STORE_B32 "flat_store_b32"
#define MN_FLAT_STORE_B64 "flat_store_b64"
#define MN_FLAT_STORE_B96 "flat_store_b96"
#define MN_FLAT_STORE_B128 "flat_store_b128"
#define MN_FLAT_STORE_B8 "flat_store_b8"
#define MN_FLAT_STORE_B16 "flat_store_b16"
#else
#define MN_SCRATCH_LOAD_B32 "scratch_load_dword"
#define MN_SCRATCH_LOAD_B64 "scratch_load_dwordx2"
#define MN_SCRATCH_LOAD_B96 "scratch_load_dwordx3"
#define MN_SCRATCH_LOAD_B128 "scratch_load_dwordx4"
#define MN_SCRATCH_LOAD_U8 "scratch_load_ubyte"
#define MN_SCRATCH_LOAD_I8 "scratch_load_sbyte"
#define MN_SCRATCH_LOAD_U16 "scratch_load_ushort"
#define MN_SCRATCH_LOAD_I16 "scratch_load_sshort"
#define MN_SCRATCH_STORE_B32 "scratch_store_dword"
#define MN_SCRATCH_STORE_B64 "scratch_store_dwordx2"
#define MN_SCRATCH_STORE_B96 "scratch_store_dwordx3"
#define MN_SCRATCH_STORE_B128 "scratch_store_dwordx4"
#define MN_SCRATCH_STORE_B8 "scratch_store_byte"
#define MN_SCRATCH_STORE_B16 "scratch_store_short"
#define MN_FLAT_LOAD_B32 "flat_load_dword"
#define MN_FLAT_LOAD_B64 "flat_load_dwordx2"
#define MN_FLAT_LOAD_B96 "flat_load_dwordx3"
#define MN_FLAT_LOAD_B128 "flat_load_dwordx4"
#define MN_FLAT_LOAD_U8 "flat_load_ubyte"
#define MN_FLAT_LOAD_I8 "flat_load_sbyte"
#define MN_FLAT_LOAD_U16 "flat_load_ushort"
#define MN_FLAT_LOAD_I16 "flat_load_sshort"
#define MN_FLAT_STORE_B32 "flat_store_dword"
#define MN_FLAT_STORE_B64 "flat_store_dwordx2"
#define MN_FLAT_STORE_B96 "flat_store_dwordx3"
#define MN_FLAT_STORE_B128 "flat_store_dwordx4"
#define MN_FLAT_STORE_B8 "flat_store_byte"
#define MN_FLAT_STORE_B16 "flat_store_short"
#endif

namespace luthier::payloads {
__attribute__((device, used)) void
scratchLoadB32Vaddr(llvm::MCRegister D0, std::uint32_t VAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_B32 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void
scratchLoadB32Saddr(llvm::MCRegister D0, std::uint32_t SAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_B32 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadB32St(llvm::MCRegister D0,
                                                    std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_B32 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadB32Svs(llvm::MCRegister D0,
                                                     std::uint32_t VAddr,
                                                     std::uint32_t SAddr,
                                                     std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_B32 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void
scratchLoadB64Vaddr(llvm::MCRegister D0, llvm::MCRegister D1,
                    std::uint32_t VAddr, std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B64 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
}

__attribute__((device, used)) void
scratchLoadB64Saddr(llvm::MCRegister D0, llvm::MCRegister D1,
                    std::uint32_t SAddr, std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B64 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
}

__attribute__((device, used)) void scratchLoadB64St(llvm::MCRegister D0,
                                                    llvm::MCRegister D1,
                                                    std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B64 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
}

__attribute__((device, used)) void
scratchLoadB64Svs(llvm::MCRegister D0, llvm::MCRegister D1, std::uint32_t VAddr,
                  std::uint32_t SAddr, std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B64 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
}

__attribute__((device, used)) void
scratchLoadB96Vaddr(llvm::MCRegister D0, llvm::MCRegister D1,
                    llvm::MCRegister D2, std::uint32_t VAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B96 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
}

__attribute__((device, used)) void
scratchLoadB96Saddr(llvm::MCRegister D0, llvm::MCRegister D1,
                    llvm::MCRegister D2, std::uint32_t SAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B96 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
}

__attribute__((device, used)) void scratchLoadB96St(llvm::MCRegister D0,
                                                    llvm::MCRegister D1,
                                                    llvm::MCRegister D2,
                                                    std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B96 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
}

__attribute__((device, used)) void
scratchLoadB96Svs(llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
                  std::uint32_t VAddr, std::uint32_t SAddr,
                  std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B96 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
}

__attribute__((device, used)) void
scratchLoadB128Vaddr(llvm::MCRegister D0, llvm::MCRegister D1,
                     llvm::MCRegister D2, llvm::MCRegister D3,
                     std::uint32_t VAddr, std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B128 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
  luthier::writeReg<std::uint32_t>(D3, Result[3]);
}

__attribute__((device, used)) void
scratchLoadB128Saddr(llvm::MCRegister D0, llvm::MCRegister D1,
                     llvm::MCRegister D2, llvm::MCRegister D3,
                     std::uint32_t SAddr, std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B128 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
  luthier::writeReg<std::uint32_t>(D3, Result[3]);
}

__attribute__((device, used)) void
scratchLoadB128St(llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
                  llvm::MCRegister D3, std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B128 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
  luthier::writeReg<std::uint32_t>(D3, Result[3]);
}

__attribute__((device, used)) void
scratchLoadB128Svs(llvm::MCRegister D0, llvm::MCRegister D1,
                   llvm::MCRegister D2, llvm::MCRegister D3,
                   std::uint32_t VAddr, std::uint32_t SAddr,
                   std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Result;
  asm volatile(MN_SCRATCH_LOAD_B128 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
  luthier::writeReg<std::uint32_t>(D3, Result[3]);
}

__attribute__((device, used)) void scratchLoadU8Vaddr(llvm::MCRegister D0,
                                                      std::uint32_t VAddr,
                                                      std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U8 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadU8Saddr(llvm::MCRegister D0,
                                                      std::uint32_t SAddr,
                                                      std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U8 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadU8St(llvm::MCRegister D0,
                                                   std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U8 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadU8Svs(llvm::MCRegister D0,
                                                    std::uint32_t VAddr,
                                                    std::uint32_t SAddr,
                                                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U8 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadI8Vaddr(llvm::MCRegister D0,
                                                      std::uint32_t VAddr,
                                                      std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I8 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadI8Saddr(llvm::MCRegister D0,
                                                      std::uint32_t SAddr,
                                                      std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I8 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadI8St(llvm::MCRegister D0,
                                                   std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I8 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadI8Svs(llvm::MCRegister D0,
                                                    std::uint32_t VAddr,
                                                    std::uint32_t SAddr,
                                                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I8 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void
scratchLoadU16Vaddr(llvm::MCRegister D0, std::uint32_t VAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U16 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void
scratchLoadU16Saddr(llvm::MCRegister D0, std::uint32_t SAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U16 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadU16St(llvm::MCRegister D0,
                                                    std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U16 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadU16Svs(llvm::MCRegister D0,
                                                     std::uint32_t VAddr,
                                                     std::uint32_t SAddr,
                                                     std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_U16 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void
scratchLoadI16Vaddr(llvm::MCRegister D0, std::uint32_t VAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I16 " %0, %1, off" : "=v"(Result) : "v"(DV));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void
scratchLoadI16Saddr(llvm::MCRegister D0, std::uint32_t SAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I16 " %0, off, %1" : "=v"(Result) : "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadI16St(llvm::MCRegister D0,
                                                    std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I16 " %0, off, %1" : "=v"(Result) : "s"(S));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void scratchLoadI16Svs(llvm::MCRegister D0,
                                                     std::uint32_t VAddr,
                                                     std::uint32_t SAddr,
                                                     std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Result;
  asm volatile(MN_SCRATCH_LOAD_I16 " %0, %1, %2"
               : "=v"(Result)
               : "v"(VAddr), "s"(DS));
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void
scratchStoreB32Vaddr(std::uint32_t V0, std::uint32_t VAddr,
                     std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B32 " %0, %1, off" : : "v"(Data), "v"(DV));
}

__attribute__((device, used)) void
scratchStoreB32Saddr(std::uint32_t V0, std::uint32_t SAddr,
                     std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B32 " %0, off, %1" : : "v"(Data), "s"(DS));
}

__attribute__((device, used)) void scratchStoreB32St(std::uint32_t V0,
                                                     std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B32 " %0, off, %1" : : "v"(Data), "s"(S));
}

__attribute__((device, used)) void scratchStoreB32Svs(std::uint32_t V0,
                                                      std::uint32_t VAddr,
                                                      std::uint32_t SAddr,
                                                      std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B32 " %0, %1, %2"
               :
               : "v"(Data), "v"(VAddr), "s"(DS));
}

__attribute__((device, used)) void
scratchStoreB64Vaddr(std::uint32_t V0, std::uint32_t V1, std::uint32_t VAddr,
                     std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Data = {V0, V1};
  asm volatile(MN_SCRATCH_STORE_B64 " %0, %1, off" : : "v"(Data), "v"(DV));
}

__attribute__((device, used)) void
scratchStoreB64Saddr(std::uint32_t V0, std::uint32_t V1, std::uint32_t SAddr,
                     std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Data = {V0, V1};
  asm volatile(MN_SCRATCH_STORE_B64 " %0, off, %1" : : "v"(Data), "s"(DS));
}

__attribute__((device, used)) void
scratchStoreB64St(std::uint32_t V0, std::uint32_t V1, std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Data = {V0, V1};
  asm volatile(MN_SCRATCH_STORE_B64 " %0, off, %1" : : "v"(Data), "s"(S));
}

__attribute__((device, used)) void
scratchStoreB64Svs(std::uint32_t V0, std::uint32_t V1, std::uint32_t VAddr,
                   std::uint32_t SAddr, std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v2u32 Data = {V0, V1};
  asm volatile(MN_SCRATCH_STORE_B64 " %0, %1, %2"
               :
               : "v"(Data), "v"(VAddr), "s"(DS));
}

__attribute__((device, used)) void
scratchStoreB96Vaddr(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                     std::uint32_t VAddr, std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Data = {V0, V1, V2};
  asm volatile(MN_SCRATCH_STORE_B96 " %0, %1, off" : : "v"(Data), "v"(DV));
}

__attribute__((device, used)) void
scratchStoreB96Saddr(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                     std::uint32_t SAddr, std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Data = {V0, V1, V2};
  asm volatile(MN_SCRATCH_STORE_B96 " %0, off, %1" : : "v"(Data), "s"(DS));
}

__attribute__((device, used)) void scratchStoreB96St(std::uint32_t V0,
                                                     std::uint32_t V1,
                                                     std::uint32_t V2,
                                                     std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Data = {V0, V1, V2};
  asm volatile(MN_SCRATCH_STORE_B96 " %0, off, %1" : : "v"(Data), "s"(S));
}

__attribute__((device, used)) void
scratchStoreB96Svs(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                   std::uint32_t VAddr, std::uint32_t SAddr,
                   std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v3u32 Data = {V0, V1, V2};
  asm volatile(MN_SCRATCH_STORE_B96 " %0, %1, %2"
               :
               : "v"(Data), "v"(VAddr), "s"(DS));
}

__attribute__((device, used)) void
scratchStoreB128Vaddr(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                      std::uint32_t V3, std::uint32_t VAddr,
                      std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Data = {V0, V1, V2, V3};
  asm volatile(MN_SCRATCH_STORE_B128 " %0, %1, off" : : "v"(Data), "v"(DV));
}

__attribute__((device, used)) void
scratchStoreB128Saddr(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                      std::uint32_t V3, std::uint32_t SAddr,
                      std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Data = {V0, V1, V2, V3};
  asm volatile(MN_SCRATCH_STORE_B128 " %0, off, %1" : : "v"(Data), "s"(DS));
}

__attribute__((device, used)) void
scratchStoreB128St(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                   std::uint32_t V3, std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Data = {V0, V1, V2, V3};
  asm volatile(MN_SCRATCH_STORE_B128 " %0, off, %1" : : "v"(Data), "s"(S));
}

__attribute__((device, used)) void
scratchStoreB128Svs(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                    std::uint32_t V3, std::uint32_t VAddr, std::uint32_t SAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  v4u32 Data = {V0, V1, V2, V3};
  asm volatile(MN_SCRATCH_STORE_B128 " %0, %1, %2"
               :
               : "v"(Data), "v"(VAddr), "s"(DS));
}

__attribute__((device, used)) void
scratchStoreB8Vaddr(std::uint32_t V0, std::uint32_t VAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B8 " %0, %1, off" : : "v"(Data), "v"(DV));
}

__attribute__((device, used)) void
scratchStoreB8Saddr(std::uint32_t V0, std::uint32_t SAddr,
                    std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B8 " %0, off, %1" : : "v"(Data), "s"(DS));
}

__attribute__((device, used)) void scratchStoreB8St(std::uint32_t V0,
                                                    std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B8 " %0, off, %1" : : "v"(Data), "s"(S));
}

__attribute__((device, used)) void scratchStoreB8Svs(std::uint32_t V0,
                                                     std::uint32_t VAddr,
                                                     std::uint32_t SAddr,
                                                     std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B8 " %0, %1, %2"
               :
               : "v"(Data), "v"(VAddr), "s"(DS));
}

__attribute__((device, used)) void
scratchStoreB16Vaddr(std::uint32_t V0, std::uint32_t VAddr,
                     std::int32_t InstOffset) {
  const std::uint32_t DV =
      VAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B16 " %0, %1, off" : : "v"(Data), "v"(DV));
}

__attribute__((device, used)) void
scratchStoreB16Saddr(std::uint32_t V0, std::uint32_t SAddr,
                     std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B16 " %0, off, %1" : : "v"(Data), "s"(DS));
}

__attribute__((device, used)) void scratchStoreB16St(std::uint32_t V0,
                                                     std::int32_t InstOffset) {
  const std::uint32_t S =
      static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B16 " %0, off, %1" : : "v"(Data), "s"(S));
}

__attribute__((device, used)) void scratchStoreB16Svs(std::uint32_t V0,
                                                      std::uint32_t VAddr,
                                                      std::uint32_t SAddr,
                                                      std::int32_t InstOffset) {
  const std::uint32_t DS =
      SAddr + static_cast<std::uint32_t>(InstOffset) + instrumentationReserve();
  std::uint32_t Data = V0;
  asm volatile(MN_SCRATCH_STORE_B16 " %0, %1, %2"
               :
               : "v"(Data), "v"(VAddr), "s"(DS));
}

__attribute__((device, used)) void
flatLoadB32(llvm::MCRegister D0, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  std::uint32_t Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 1; ++I)
      reinterpret_cast<std::uint32_t *>(&Result)[I] =
          *(Private<std::uint32_t> *)(Off + 4 * I);
  } else {
    asm volatile(MN_FLAT_LOAD_B32 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(D0, Result);
}

__attribute__((device, used)) void flatLoadB64(llvm::MCRegister D0,
                                               llvm::MCRegister D1,
                                               std::uint64_t Addr,
                                               std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  v2u32 Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 2; ++I)
      reinterpret_cast<std::uint32_t *>(&Result)[I] =
          *(Private<std::uint32_t> *)(Off + 4 * I);
  } else {
    asm volatile(MN_FLAT_LOAD_B64 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
}

__attribute__((device, used)) void
flatLoadB96(llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
            std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  v3u32 Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 3; ++I)
      reinterpret_cast<std::uint32_t *>(&Result)[I] =
          *(Private<std::uint32_t> *)(Off + 4 * I);
  } else {
    asm volatile(MN_FLAT_LOAD_B96 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
}

__attribute__((device, used)) void
flatLoadB128(llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
             llvm::MCRegister D3, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  v4u32 Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 4; ++I)
      reinterpret_cast<std::uint32_t *>(&Result)[I] =
          *(Private<std::uint32_t> *)(Off + 4 * I);
  } else {
    asm volatile(MN_FLAT_LOAD_B128 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(D0, Result[0]);
  luthier::writeReg<std::uint32_t>(D1, Result[1]);
  luthier::writeReg<std::uint32_t>(D2, Result[2]);
  luthier::writeReg<std::uint32_t>(D3, Result[3]);
}

__attribute__((device, used)) void
flatStoreB32(std::uint32_t V0, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  std::uint32_t Data = V0;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 1; ++I)
      *(Private<std::uint32_t> *)(Off + 4 * I) =
          reinterpret_cast<const std::uint32_t *>(&Data)[I];
  } else {
    asm volatile(MN_FLAT_STORE_B32 " %0, %1" : : "v"(Data), "v"(FlatAddr));
  }
}

__attribute__((device, used)) void flatStoreB64(std::uint32_t V0,
                                                std::uint32_t V1,
                                                std::uint64_t Addr,
                                                std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  v2u32 Data = {V0, V1};
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 2; ++I)
      *(Private<std::uint32_t> *)(Off + 4 * I) =
          reinterpret_cast<const std::uint32_t *>(&Data)[I];
  } else {
    asm volatile(MN_FLAT_STORE_B64 " %0, %1" : : "v"(Data), "v"(FlatAddr));
  }
}

__attribute__((device, used)) void
flatStoreB96(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
             std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  v3u32 Data = {V0, V1, V2};
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 3; ++I)
      *(Private<std::uint32_t> *)(Off + 4 * I) =
          reinterpret_cast<const std::uint32_t *>(&Data)[I];
  } else {
    asm volatile(MN_FLAT_STORE_B96 " %0, %1" : : "v"(Data), "v"(FlatAddr));
  }
}

__attribute__((device, used)) void
flatStoreB128(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
              std::uint32_t V3, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  v4u32 Data = {V0, V1, V2, V3};
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    for (unsigned I = 0; I != 4; ++I)
      *(Private<std::uint32_t> *)(Off + 4 * I) =
          reinterpret_cast<const std::uint32_t *>(&Data)[I];
  } else {
    asm volatile(MN_FLAT_STORE_B128 " %0, %1" : : "v"(Data), "v"(FlatAddr));
  }
}

__attribute__((device, used)) void
flatLoadU8(llvm::MCRegister Dst, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  std::uint32_t Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    Result = static_cast<std::uint32_t>(
        static_cast<std::int32_t>(*(Private<std::uint8_t> *)(Off)));
  } else {
    asm volatile(MN_FLAT_LOAD_U8 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(Dst, Result);
}

__attribute__((device, used)) void
flatLoadI8(llvm::MCRegister Dst, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  std::uint32_t Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    Result = static_cast<std::uint32_t>(
        static_cast<std::int32_t>(*(Private<std::int8_t> *)(Off)));
  } else {
    asm volatile(MN_FLAT_LOAD_I8 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(Dst, Result);
}

__attribute__((device, used)) void
flatLoadU16(llvm::MCRegister Dst, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  std::uint32_t Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    Result = static_cast<std::uint32_t>(
        static_cast<std::int32_t>(*(Private<std::uint16_t> *)(Off)));
  } else {
    asm volatile(MN_FLAT_LOAD_U16 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(Dst, Result);
}

__attribute__((device, used)) void
flatLoadI16(llvm::MCRegister Dst, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  std::uint32_t Result;
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    Result = static_cast<std::uint32_t>(
        static_cast<std::int32_t>(*(Private<std::int16_t> *)(Off)));
  } else {
    asm volatile(MN_FLAT_LOAD_I16 " %0, %1" : "=v"(Result) : "v"(FlatAddr));
  }
  luthier::writeReg<std::uint32_t>(Dst, Result);
}

__attribute__((device, used)) void
flatStoreB8(std::uint32_t Value, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    *(Private<std::uint8_t> *)(Off) = static_cast<std::uint8_t>(Value);
  } else {
    asm volatile(MN_FLAT_STORE_B8 " %0, %1"
                 :
                 : "v"(static_cast<std::uint8_t>(Value)), "v"(FlatAddr));
  }
}

__attribute__((device, used)) void
flatStoreB16(std::uint32_t Value, std::uint64_t Addr, std::int32_t InstOffset) {
  const std::uint64_t FlatAddr =
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset));
  if (__builtin_amdgcn_is_private(reinterpret_cast<const void *>(FlatAddr))) {
    const std::uint32_t Off =
        displacedPrivateOffsetOf(reinterpret_cast<const void *>(FlatAddr));
    *(Private<std::uint16_t> *)(Off) = static_cast<std::uint16_t>(Value);
  } else {
    asm volatile(MN_FLAT_STORE_B16 " %0, %1"
                 :
                 : "v"(static_cast<std::uint16_t>(Value)), "v"(FlatAddr));
  }
}

__attribute__((device, used)) void
bufferLoadB32Offset(llvm::MCRegister D0, __amdgpu_buffer_rsrc_t Rsrc,
                    std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                    std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(1); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
}

__attribute__((device, used)) void
bufferLoadB32Offen(llvm::MCRegister D0, __amdgpu_buffer_rsrc_t Rsrc,
                   std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                   std::uint32_t VOffset, std::uint32_t SOffset,
                   std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(1); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
}

__attribute__((device, used)) void
bufferLoadB64Offset(llvm::MCRegister D0, llvm::MCRegister D1,
                    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo,
                    std::uint32_t RsrcHi, std::uint32_t SOffset,
                    std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(2); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
  luthier::writeReg<std::uint32_t>(D1, V[1]);
}

__attribute__((device, used)) void
bufferLoadB64Offen(llvm::MCRegister D0, llvm::MCRegister D1,
                   __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo,
                   std::uint32_t RsrcHi, std::uint32_t VOffset,
                   std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(2); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
  luthier::writeReg<std::uint32_t>(D1, V[1]);
}

__attribute__((device, used)) void
bufferLoadB96Offset(llvm::MCRegister D0, llvm::MCRegister D1,
                    llvm::MCRegister D2, __amdgpu_buffer_rsrc_t Rsrc,
                    std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                    std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(3); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
  luthier::writeReg<std::uint32_t>(D1, V[1]);
  luthier::writeReg<std::uint32_t>(D2, V[2]);
}

__attribute__((device, used)) void bufferLoadB96Offen(
    llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo, std::uint32_t RsrcHi,
    std::uint32_t VOffset, std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(3); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
  luthier::writeReg<std::uint32_t>(D1, V[1]);
  luthier::writeReg<std::uint32_t>(D2, V[2]);
}

__attribute__((device, used)) void bufferLoadB128Offset(
    llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
    llvm::MCRegister D3, __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo,
    std::uint32_t RsrcHi, std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(4); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
  luthier::writeReg<std::uint32_t>(D1, V[1]);
  luthier::writeReg<std::uint32_t>(D2, V[2]);
  luthier::writeReg<std::uint32_t>(D3, V[3]);
}

__attribute__((device, used)) void
bufferLoadB128Offen(llvm::MCRegister D0, llvm::MCRegister D1,
                    llvm::MCRegister D2, llvm::MCRegister D3,
                    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo,
                    std::uint32_t RsrcHi, std::uint32_t VOffset,
                    std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != static_cast<unsigned>(4); ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
  luthier::writeReg<std::uint32_t>(D0, V[0]);
  luthier::writeReg<std::uint32_t>(D1, V[1]);
  luthier::writeReg<std::uint32_t>(D2, V[2]);
  luthier::writeReg<std::uint32_t>(D3, V[3]);
}

__attribute__((device, used)) void
bufferStoreB32Offset(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                     std::uint32_t V3, __amdgpu_buffer_rsrc_t Rsrc,
                     std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                     std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(1); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void bufferStoreB32Offen(
    std::uint32_t V0, std::uint32_t V1, std::uint32_t V2, std::uint32_t V3,
    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo, std::uint32_t RsrcHi,
    std::uint32_t VOffset, std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(1); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void
bufferStoreB64Offset(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                     std::uint32_t V3, __amdgpu_buffer_rsrc_t Rsrc,
                     std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                     std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(2); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void bufferStoreB64Offen(
    std::uint32_t V0, std::uint32_t V1, std::uint32_t V2, std::uint32_t V3,
    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo, std::uint32_t RsrcHi,
    std::uint32_t VOffset, std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(2); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void
bufferStoreB96Offset(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                     std::uint32_t V3, __amdgpu_buffer_rsrc_t Rsrc,
                     std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                     std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(3); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void bufferStoreB96Offen(
    std::uint32_t V0, std::uint32_t V1, std::uint32_t V2, std::uint32_t V3,
    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo, std::uint32_t RsrcHi,
    std::uint32_t VOffset, std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(3); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void
bufferStoreB128Offset(std::uint32_t V0, std::uint32_t V1, std::uint32_t V2,
                      std::uint32_t V3, __amdgpu_buffer_rsrc_t Rsrc,
                      std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                      std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(4); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void bufferStoreB128Offen(
    std::uint32_t V0, std::uint32_t V1, std::uint32_t V2, std::uint32_t V3,
    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo, std::uint32_t RsrcHi,
    std::uint32_t VOffset, std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != static_cast<unsigned>(4); ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(LaneOffset + 4 * I), SOff, 0);
}

__attribute__((device, used)) void
bufferLoadU8Offset(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                   std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                   std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b8(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferLoadU8Offen(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                  std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                  std::uint32_t VOffset, std::uint32_t SOffset,
                  std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b8(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferLoadI8Offset(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                   std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                   std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b8(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferLoadI8Offen(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                  std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                  std::uint32_t VOffset, std::uint32_t SOffset,
                  std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b8(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferLoadU16Offset(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                    std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                    std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b16(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferLoadU16Offen(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                   std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                   std::uint32_t VOffset, std::uint32_t SOffset,
                   std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b16(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferLoadI16Offset(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                    std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                    std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b16(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferLoadI16Offen(llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc,
                   std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                   std::uint32_t VOffset, std::uint32_t SOffset,
                   std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  const auto Val = __builtin_amdgcn_raw_buffer_load_b16(
      Rsrc, static_cast<int>(LaneOffset), SOff, 0);
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

__attribute__((device, used)) void
bufferStoreB8Offset(std::uint32_t Value, __amdgpu_buffer_rsrc_t Rsrc,
                    std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                    std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  __builtin_amdgcn_raw_buffer_store_b8(static_cast<unsigned char>(Value), Rsrc,
                                       static_cast<int>(LaneOffset), SOff, 0);
}

__attribute__((device, used)) void
bufferStoreB8Offen(std::uint32_t Value, __amdgpu_buffer_rsrc_t Rsrc,
                   std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                   std::uint32_t VOffset, std::uint32_t SOffset,
                   std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  __builtin_amdgcn_raw_buffer_store_b8(static_cast<unsigned char>(Value), Rsrc,
                                       static_cast<int>(LaneOffset), SOff, 0);
}

__attribute__((device, used)) void
bufferStoreB16Offset(std::uint32_t Value, __amdgpu_buffer_rsrc_t Rsrc,
                     std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                     std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset = static_cast<std::uint32_t>(InstOffset);
  __builtin_amdgcn_raw_buffer_store_b16(static_cast<unsigned short>(Value),
                                        Rsrc, static_cast<int>(LaneOffset),
                                        SOff, 0);
}

__attribute__((device, used)) void
bufferStoreB16Offen(std::uint32_t Value, __amdgpu_buffer_rsrc_t Rsrc,
                    std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                    std::uint32_t VOffset, std::uint32_t SOffset,
                    std::int32_t InstOffset) {
  const int SOff =
      static_cast<int>(displacedBufferSOffset(RsrcLo, RsrcHi, SOffset));
  const std::uint32_t LaneOffset =
      VOffset + static_cast<std::uint32_t>(InstOffset);
  __builtin_amdgcn_raw_buffer_store_b16(static_cast<unsigned short>(Value),
                                        Rsrc, static_cast<int>(LaneOffset),
                                        SOff, 0);
}

} // namespace luthier::payloads

//===----------------------------------------------------------------------===//
// Host side
//===----------------------------------------------------------------------===//

#ifndef __HIP_DEVICE_COMPILE__

namespace Payloads = luthier::payloads;

namespace luthier {

namespace {

/// \return the operand named \p Name of \p MI , or \c nullptr when the opcode
/// has no such operand --- which is how the addressing forms differ.
const llvm::MachineOperand *findNamedOp(const llvm::MachineInstr &MI,
                                        llvm::AMDGPU::OpName Name) {
  int Idx = llvm::AMDGPU::getNamedOperandIdx(MI.getOpcode(), Name);
  return Idx == -1 ? nullptr : &MI.getOperand(Idx);
}

/// \return the immediate of \p MI 's operand named \p Name , or 0 when the
/// opcode has no such operand.
int64_t namedImmOr0(const llvm::MachineInstr &MI, llvm::AMDGPU::OpName Name) {
  const llvm::MachineOperand *Op = findNamedOp(MI, Name);
  return Op && Op->isImm() ? Op->getImm() : 0;
}

} // namespace

RebaseAppScratchAccessesPass::RebaseAppScratchAccessesPass(llvm::Error &Err)
    : OffloadParser(std::make_unique<Parser>(Err)) {}

llvm::PreservedAnalyses
RebaseAppScratchAccessesPass::run(Prototype &P, PrototypeAnalysisManager &PAM) {
  llvm::Module &IModule = P.getInstrumentationModule();
  llvm::LLVMContext &Ctx = IModule.getContext();
  llvm::FunctionAnalysisManager &IFAM =
      PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager();
  llvm::ModuleAnalysisManager &IMAM =
      PAM.getResult<IModuleAnalysisManagerPrototypeProxy>(P).getManager();
  const llvm::TargetMachine &TM =
      IMAM.getResult<llvm::MachineModuleAnalysis>(IModule).getMMI().getTarget();

  /// Collect a list of instructions that will potentially touch scratch
  llvm::SmallVector<llvm::MachineInstr *, 16> Worklist;

  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
    const auto &TII = *ST.getInstrInfo();

    for (llvm::MachineBasicBlock &MBB : MF) {
      for (llvm::MachineInstr &MI : MBB) {
        if (TII.isFLATScratch(MI) ||
            (TII.isFLAT(MI) && !TII.isFLATGlobal(MI)) || TII.isMUBUF(MI))
          Worklist.push_back(&MI);
      }
    }
  });

  if (Worklist.empty())
    return llvm::PreservedAnalyses::all();

  // ── Link the payload module ──
  /// Names of every payload the linked module contributed.
  ///
  /// Each payload is `__attribute__((used))`, which is what makes the device
  /// compile emit it into the bundle at all --- nothing in that translation
  /// unit calls it. The flip side is that the linked module pins all of them
  /// via `llvm.compiler.used`, so their out-of-line bodies survive into
  /// `ProcessIntrinsicsAtIRLevelPass`, which rejects them: a payload names the
  /// register it writes back through its own parameter, and the intrinsic
  /// lowering needs a literal. `createInjectedPayload` force-inlines the
  /// payloads this pass actually uses, which fixes them up at the call site,
  /// but the ones it does not use stay behind. So all of them get unpinned and
  /// erased once every payload has been built.
  llvm::SmallVector<std::string, 64> LinkedPayloadNames;
  {
    llvm::SubtargetFeatures Features(TM.getTargetFeatureString());
    auto SubOrErr = OffloadParser->parseModule(
        TM.getTargetTriple(), TM.getTargetCPU(), Features, Ctx);
    if (!SubOrErr) {
      Ctx.emitError(llvm::toString(SubOrErr.takeError()));
      return llvm::PreservedAnalyses::all();
    }
    std::unique_ptr<llvm::Module> Sub = std::move(*SubOrErr);
    Sub->setTargetTriple(IModule.getTargetTriple());
    Sub->setDataLayout(IModule.getDataLayout());
    llvm::SmallVector<llvm::GlobalValue *, 64> Used;
    llvm::collectUsedGlobalVariables(*Sub, Used, /*CompilerUsed=*/false);
    llvm::collectUsedGlobalVariables(*Sub, Used, /*CompilerUsed=*/true);
    for (llvm::GlobalValue *GV : Used) {
      auto *F = llvm::dyn_cast<llvm::Function>(GV);
      if (!F || F->isDeclaration())
        continue;
      if (F->hasLocalLinkage())
        F->setLinkage(llvm::GlobalValue::ExternalLinkage);
      LinkedPayloadNames.push_back(F->getName().str());
    }
    if (llvm::Linker(IModule).linkInModule(std::move(Sub))) {
      Ctx.emitError(
          "RebaseAppScratchAccessesPass: failed to link the pass's device-side "
          "payload module into the prototype's instrumentation module.");
      return llvm::PreservedAnalyses::all();
    }
  }

  auto hookFor = [&](auto *HostHandle) -> llvm::Expected<llvm::Function *> {
    auto *F = OffloadParser->getLLVMHandle<llvm::Function>(IModule, HostHandle);
    if (!F || F->isDeclaration())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "RebaseAppScratchAccessesPass: payload hook '{0:x}' is missing from "
          "the instrumentation module after linking.",
          HostHandle));
    return F;
  };

  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *I64 = llvm::Type::getInt64Ty(Ctx);
  llvm::Type *RsrcTy = llvm::PointerType::get(Ctx, 8);
  auto constI32 = [&](uint64_t V) -> PayloadArg {
    return llvm::ConstantInt::get(I32, V);
  };

  // ── Pass 2: instrument each collected instruction ──
  llvm::Error Err = llvm::Error::success();
  llvm::SmallVector<llvm::MachineInstr *, 16> ToErase;

  for (llvm::MachineInstr *MI : Worklist) {
    if (Err)
      break;
    const llvm::MachineFunction &MF = *MI->getParent()->getParent();
    const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
    const auto &TII = *ST.getInstrInfo();
    const auto &TRI = *ST.getRegisterInfo();

    // Cache-policy bits are a performance hint on private memory --- it is
    // wave-private and coherent with nothing --- so they are dropped on the
    // scratch path. On the flat and buffer paths the access may be global,
    // where they are not droppable, and the re-issued access has no way to
    // spell them: the buffer builtins take a compile-time-constant aux
    // operand and a plain pointer dereference carries no policy at all.
    const int64_t CPol = namedImmOr0(*MI, llvm::AMDGPU::OpName::cpol);
    if (CPol != 0 && !TII.isFLATScratch(*MI)) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "RebaseAppScratchAccessesPass: '{0}' in '{1}' carries cache "
          "policy {2}, which the re-issued access cannot reproduce.",
          TII.getName(MI->getOpcode()), MF.getName(), CPol));
      break;
    }

    const int64_t InstOffset = namedImmOr0(*MI, llvm::AMDGPU::OpName::offset);
    const llvm::MachineOperand *VAddr =
        findNamedOp(*MI, llvm::AMDGPU::OpName::vaddr);
    const llvm::MachineOperand *SAddr =
        findNamedOp(*MI, llvm::AMDGPU::OpName::saddr);
    const llvm::MachineOperand *SOffset =
        findNamedOp(*MI, llvm::AMDGPU::OpName::soffset);
    const llvm::MachineOperand *SRsrc =
        findNamedOp(*MI, llvm::AMDGPU::OpName::srsrc);
    const llvm::MachineOperand *Data =
        findNamedOp(*MI, llvm::AMDGPU::OpName::vdst);
    if (!Data)
      Data = findNamedOp(*MI, llvm::AMDGPU::OpName::vdata);
    if (!Data || !Data->isReg() || !Data->getReg()) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "RebaseAppScratchAccessesPass: '{0}' in '{1}' has no data "
          "register operand to re-issue through.",
          TII.getName(MI->getOpcode()), MF.getName()));
      break;
    }

    auto addrArg = [&](const llvm::MachineOperand *Op,
                       llvm::Type *Ty) -> PayloadArg {
      if (!Op)
        return constI32(0);
      if (Op->isImm())
        return llvm::ConstantInt::get(Ty, Op->getImm());
      if (!Op->isReg() || !Op->getReg())
        return constI32(0);
      return RegArg{Op->getReg().asMCReg(), Ty};
    };

    const llvm::MCRegister DataReg = Data->getReg().asMCReg();
    auto channel = [&](unsigned I, unsigned N) -> llvm::MCRegister {
      if (N <= 1)
        return DataReg;
      return TRI.getSubReg(DataReg,
                           llvm::SIRegisterInfo::getSubRegFromChannel(I));
    };

    llvm::SmallVector<PayloadArg, 10> Args;

    auto loadData = [&](unsigned N) {
      for (unsigned I = 0; I < N; ++I)
        Args.push_back(constI32(channel(I, N).id()));
    };
    auto storeData = [&](unsigned N) {
      for (unsigned I = 0; I < N; ++I)
        Args.push_back(PayloadArg{RegArg{channel(I, N), I32}});
    };
    auto subLoadData = [&]() { Args.push_back(constI32(DataReg.id())); };
    auto subStoreData = [&]() {
      Args.push_back(PayloadArg{RegArg{DataReg, I32}});
    };
    auto bufferStoreData = [&](unsigned N) {
      for (unsigned I = 0; I < 4; ++I) {
        if (I >= N) {
          Args.push_back(constI32(0));
          continue;
        }
        Args.push_back(PayloadArg{RegArg{channel(I, N), I32}});
      }
    };

    auto scratchVaddr = [&]() {
      Args.push_back(RegArg{VAddr->getReg().asMCReg(), I32});
      Args.push_back(constI32(InstOffset));
    };
    auto scratchSaddr = [&]() {
      Args.push_back(RegArg{SAddr->getReg().asMCReg(), I32});
      Args.push_back(constI32(InstOffset));
    };
    auto scratchSt = [&]() { Args.push_back(constI32(InstOffset)); };
    auto scratchSvs = [&]() {
      Args.push_back(RegArg{VAddr->getReg().asMCReg(), I32});
      Args.push_back(RegArg{SAddr->getReg().asMCReg(), I32});
      Args.push_back(constI32(InstOffset));
    };

    auto flatAddr = [&]() {
      Args.push_back(addrArg(VAddr, I64));
      Args.push_back(constI32(InstOffset));
    };

    auto bufferOffen = [&]() {
      const llvm::MCRegister Rsrc = SRsrc->getReg().asMCReg();
      Args.push_back(RegArg{Rsrc, RsrcTy});
      Args.push_back(RegArg{TRI.getSubReg(Rsrc, llvm::AMDGPU::sub0), I32});
      Args.push_back(RegArg{TRI.getSubReg(Rsrc, llvm::AMDGPU::sub1), I32});
      Args.push_back(addrArg(VAddr, I32));
      Args.push_back(addrArg(SOffset, I32));
      Args.push_back(constI32(InstOffset));
    };
    auto bufferOffset = [&]() {
      const llvm::MCRegister Rsrc = SRsrc->getReg().asMCReg();
      Args.push_back(RegArg{Rsrc, RsrcTy});
      Args.push_back(RegArg{TRI.getSubReg(Rsrc, llvm::AMDGPU::sub0), I32});
      Args.push_back(RegArg{TRI.getSubReg(Rsrc, llvm::AMDGPU::sub1), I32});
      Args.push_back(addrArg(SOffset, I32));
      Args.push_back(constI32(InstOffset));
    };

    llvm::Expected<llvm::Function *> HookOrErr =
        llvm::make_error<llvm::StringError>("unreachable",
                                            llvm::inconvertibleErrorCode());

    switch (MI->getOpcode()) {
    // ---- Scratch loads ----
    case llvm::AMDGPU::SCRATCH_LOAD_DWORD:
      loadData(1);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB32Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR:
      loadData(1);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB32Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORD_ST:
      loadData(1);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadB32St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORD_SVS:
      loadData(1);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadB32Svs);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX2:
      loadData(2);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB64Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX2_SADDR:
      loadData(2);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB64Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX2_ST:
      loadData(2);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadB64St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX2_SVS:
      loadData(2);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadB64Svs);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX3:
      loadData(3);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB96Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX3_SADDR:
      loadData(3);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB96Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX3_ST:
      loadData(3);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadB96St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX3_SVS:
      loadData(3);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadB96Svs);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX4:
      loadData(4);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB128Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX4_SADDR:
      loadData(4);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadB128Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX4_ST:
      loadData(4);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadB128St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_DWORDX4_SVS:
      loadData(4);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadB128Svs);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_UBYTE:
      subLoadData();
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadU8Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_UBYTE_SADDR:
      subLoadData();
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadU8Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_UBYTE_ST:
      subLoadData();
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadU8St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_UBYTE_SVS:
      subLoadData();
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadU8Svs);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SBYTE:
      subLoadData();
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadI8Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SBYTE_SADDR:
      subLoadData();
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadI8Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SBYTE_ST:
      subLoadData();
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadI8St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SBYTE_SVS:
      subLoadData();
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadI8Svs);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_USHORT:
      subLoadData();
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadU16Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_USHORT_SADDR:
      subLoadData();
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadU16Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_USHORT_ST:
      subLoadData();
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadU16St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_USHORT_SVS:
      subLoadData();
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadU16Svs);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SSHORT:
      subLoadData();
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadI16Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SSHORT_SADDR:
      subLoadData();
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchLoadI16Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SSHORT_ST:
      subLoadData();
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchLoadI16St);
      break;
    case llvm::AMDGPU::SCRATCH_LOAD_SSHORT_SVS:
      subLoadData();
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchLoadI16Svs);
      break;
    // ---- Scratch stores ----
    case llvm::AMDGPU::SCRATCH_STORE_DWORD:
      storeData(1);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB32Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR:
      storeData(1);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB32Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORD_ST:
      storeData(1);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchStoreB32St);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORD_SVS:
      storeData(1);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchStoreB32Svs);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX2:
      storeData(2);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB64Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX2_SADDR:
      storeData(2);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB64Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX2_ST:
      storeData(2);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchStoreB64St);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX2_SVS:
      storeData(2);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchStoreB64Svs);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX3:
      storeData(3);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB96Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX3_SADDR:
      storeData(3);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB96Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX3_ST:
      storeData(3);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchStoreB96St);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX3_SVS:
      storeData(3);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchStoreB96Svs);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX4:
      storeData(4);
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB128Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX4_SADDR:
      storeData(4);
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB128Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX4_ST:
      storeData(4);
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchStoreB128St);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_DWORDX4_SVS:
      storeData(4);
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchStoreB128Svs);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_BYTE:
      subStoreData();
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB8Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_BYTE_SADDR:
      subStoreData();
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB8Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_BYTE_ST:
      subStoreData();
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchStoreB8St);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_BYTE_SVS:
      subStoreData();
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchStoreB8Svs);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_SHORT:
      subStoreData();
      scratchVaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB16Vaddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_SHORT_SADDR:
      subStoreData();
      scratchSaddr();
      HookOrErr = hookFor(&Payloads::scratchStoreB16Saddr);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_SHORT_ST:
      subStoreData();
      scratchSt();
      HookOrErr = hookFor(&Payloads::scratchStoreB16St);
      break;
    case llvm::AMDGPU::SCRATCH_STORE_SHORT_SVS:
      subStoreData();
      scratchSvs();
      HookOrErr = hookFor(&Payloads::scratchStoreB16Svs);
      break;
    // ---- Flat loads ----
    case llvm::AMDGPU::FLAT_LOAD_DWORD:
      loadData(1);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadB32);
      break;
    case llvm::AMDGPU::FLAT_LOAD_DWORDX2:
      loadData(2);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadB64);
      break;
    case llvm::AMDGPU::FLAT_LOAD_DWORDX3:
      loadData(3);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadB96);
      break;
    case llvm::AMDGPU::FLAT_LOAD_DWORDX4:
      loadData(4);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadB128);
      break;
    case llvm::AMDGPU::FLAT_LOAD_UBYTE:
      subLoadData();
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadU8);
      break;
    case llvm::AMDGPU::FLAT_LOAD_SBYTE:
      subLoadData();
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadI8);
      break;
    case llvm::AMDGPU::FLAT_LOAD_USHORT:
      subLoadData();
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadU16);
      break;
    case llvm::AMDGPU::FLAT_LOAD_SSHORT:
      subLoadData();
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatLoadI16);
      break;
    // ---- Flat stores ----
    case llvm::AMDGPU::FLAT_STORE_DWORD:
      storeData(1);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatStoreB32);
      break;
    case llvm::AMDGPU::FLAT_STORE_DWORDX2:
      storeData(2);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatStoreB64);
      break;
    case llvm::AMDGPU::FLAT_STORE_DWORDX3:
      storeData(3);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatStoreB96);
      break;
    case llvm::AMDGPU::FLAT_STORE_DWORDX4:
      storeData(4);
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatStoreB128);
      break;
    case llvm::AMDGPU::FLAT_STORE_BYTE:
      subStoreData();
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatStoreB8);
      break;
    case llvm::AMDGPU::FLAT_STORE_SHORT:
      subStoreData();
      flatAddr();
      HookOrErr = hookFor(&Payloads::flatStoreB16);
      break;
    // ---- Buffer loads (OFFEN) ----
    case llvm::AMDGPU::BUFFER_LOAD_DWORD_OFFEN:
      loadData(1);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadB32Offen);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_DWORDX2_OFFEN:
      loadData(2);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadB64Offen);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_DWORDX3_OFFEN:
      loadData(3);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadB96Offen);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_DWORDX4_OFFEN:
      loadData(4);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadB128Offen);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_UBYTE_OFFEN:
      subLoadData();
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadU8Offen);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_SBYTE_OFFEN:
      subLoadData();
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadI8Offen);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_USHORT_OFFEN:
      subLoadData();
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadU16Offen);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_SSHORT_OFFEN:
      subLoadData();
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferLoadI16Offen);
      break;
    // ---- Buffer loads (OFFSET) ----
    case llvm::AMDGPU::BUFFER_LOAD_DWORD_OFFSET:
      loadData(1);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadB32Offset);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_DWORDX2_OFFSET:
      loadData(2);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadB64Offset);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_DWORDX3_OFFSET:
      loadData(3);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadB96Offset);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_DWORDX4_OFFSET:
      loadData(4);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadB128Offset);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_UBYTE_OFFSET:
      subLoadData();
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadU8Offset);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_SBYTE_OFFSET:
      subLoadData();
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadI8Offset);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_USHORT_OFFSET:
      subLoadData();
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadU16Offset);
      break;
    case llvm::AMDGPU::BUFFER_LOAD_SSHORT_OFFSET:
      subLoadData();
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferLoadI16Offset);
      break;
    // ---- Buffer stores (OFFEN) ----
    case llvm::AMDGPU::BUFFER_STORE_DWORD_OFFEN:
      bufferStoreData(1);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferStoreB32Offen);
      break;
    case llvm::AMDGPU::BUFFER_STORE_DWORDX2_OFFEN:
      bufferStoreData(2);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferStoreB64Offen);
      break;
    case llvm::AMDGPU::BUFFER_STORE_DWORDX3_OFFEN:
      bufferStoreData(3);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferStoreB96Offen);
      break;
    case llvm::AMDGPU::BUFFER_STORE_DWORDX4_OFFEN:
      bufferStoreData(4);
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferStoreB128Offen);
      break;
    case llvm::AMDGPU::BUFFER_STORE_BYTE_OFFEN:
      subStoreData();
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferStoreB8Offen);
      break;
    case llvm::AMDGPU::BUFFER_STORE_SHORT_OFFEN:
      subStoreData();
      bufferOffen();
      HookOrErr = hookFor(&Payloads::bufferStoreB16Offen);
      break;
    // ---- Buffer stores (OFFSET) ----
    case llvm::AMDGPU::BUFFER_STORE_DWORD_OFFSET:
      bufferStoreData(1);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferStoreB32Offset);
      break;
    case llvm::AMDGPU::BUFFER_STORE_DWORDX2_OFFSET:
      bufferStoreData(2);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferStoreB64Offset);
      break;
    case llvm::AMDGPU::BUFFER_STORE_DWORDX3_OFFSET:
      bufferStoreData(3);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferStoreB96Offset);
      break;
    case llvm::AMDGPU::BUFFER_STORE_DWORDX4_OFFSET:
      bufferStoreData(4);
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferStoreB128Offset);
      break;
    case llvm::AMDGPU::BUFFER_STORE_BYTE_OFFSET:
      subStoreData();
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferStoreB8Offset);
      break;
    case llvm::AMDGPU::BUFFER_STORE_SHORT_OFFSET:
      subStoreData();
      bufferOffset();
      HookOrErr = hookFor(&Payloads::bufferStoreB16Offset);
      break;
    default:
      if (TII.isFLATScratch(*MI)) {
        Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "RebaseAppScratchAccessesPass: '{0}' in '{1}' addresses the "
            "wavefront's private segment but is not a form this pass knows "
            "how to re-issue. Add a payload for it in "
            "RebaseAppScratchAccessesPass.hip.",
            TII.getName(MI->getOpcode()), MF.getName()));
        break;
      }
      continue;
    }

    if (!HookOrErr) {
      Err = HookOrErr.takeError();
      break;
    }
    auto PayloadOrErr = P.createInjectedPayload(**HookOrErr, *MI, IFAM, Args);
    if (!PayloadOrErr) {
      Err = PayloadOrErr.takeError();
      break;
    }
    ToErase.push_back(MI);
  }

  if (Err) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return llvm::PreservedAnalyses::all();
  }

  // The payloads are anchored on the PATCHPOINT markers `createInjectedPayload`
  // inserted *before* each of these instructions, not on the instructions
  // themselves, so erasing them now leaves every binding intact. Carry each
  // instruction's trace address over to its marker first: the callgraph, code
  // discovery and the trace translator all index by it.
  for (llvm::MachineInstr *MI : ToErase) {
    if (auto *MD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(*MI)) {
      if (auto Addr = MD->getTraceInstrAddress()) {
        llvm::MachineInstr *Marker = MI->getPrevNode();
        while (Marker && Marker->getOpcode() != llvm::TargetOpcode::PATCHPOINT)
          Marker = Marker->getPrevNode();
        if (Marker)
          if (auto *MarkerMD =
                  TargetMachineInstrMDNode::getInstrMDNodeIfExists(*Marker))
            MarkerMD->setTraceInstrAddress(Ctx, *Addr);
      }
    }
    MI->eraseFromParent();
  }

  // Retire the out-of-line payload bodies; see LinkedPayloadNames above for why
  // none of them can be left behind.
  if (!LinkedPayloadNames.empty()) {
    llvm::SmallPtrSet<llvm::Function *, 32> ToRetire;
    for (const std::string &Name : LinkedPayloadNames)
      if (llvm::Function *F = IModule.getFunction(Name))
        ToRetire.insert(F);
    llvm::removeFromUsedLists(IModule, [&](llvm::Constant *C) {
      auto *F = llvm::dyn_cast_or_null<llvm::Function>(
          C ? C->stripPointerCasts() : nullptr);
      return F && ToRetire.contains(F);
    });
    for (llvm::Function *F : ToRetire) {
      // Each payload's `appendToCompilerUsed` erases the old llvm.compiler.used
      // global and builds a new one, but LLVM keeps the superseded
      // ConstantArrays alive, and each still counts as a use of everything it
      // referenced. Drop those before concluding anything from use_empty().
      F->removeDeadConstantUsers();
      if (!F->use_empty()) {
        Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "RebaseAppScratchAccessesPass: payload '{0}' still has {1} use(s) "
            "after every injected payload was built; the ones this pass uses "
            "are force-inlined and the rest are unused, so it should have "
            "none.",
            F->getName(), F->getNumUses()))));
        return llvm::PreservedAnalyses::all();
      }
      F->eraseFromParent();
    }
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[RebaseAppScratchAccessesPass] rebased " << ToErase.size()
             << " application private-memory access(es)\n");

  if (ToErase.empty())
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.abandon<InjectedPayloadAndInstPointAnalysis>();
  PA.abandon<PrototypeCallGraphAnalysis>();
  return PA;
}

} // namespace luthier

#endif // __HIP_DEVICE_COMPILE__
