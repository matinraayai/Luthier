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
/// Implements \c RebaseAppScratchAccessesPass : both the device-side payloads
/// that re-perform an application's private-memory access at the displaced
/// address, and the host-side pass that attaches them and erases the original
/// instructions.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/RebaseAppScratchAccessesPass.h"

#include "luthier/Intrinsic/Intrinsics.h"
#include "luthier/Intrinsic/ScalarValueArgument.h"

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
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/SubtargetFeature.h>
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

/// \return the flat address a buffer instruction was going to access, given its
/// V# words 0 and 1 and its three offset terms.
///
/// The V#'s 48-bit base plus the instruction's offsets *is* the address the
/// hardware forms, so it is also the address to aperture-check --- there is no
/// need to know where the wavefront's scratch starts, which matters because
/// neither \c FLAT_SCRATCH nor \c WAVEFRONT_PRIVATE_SEGMENT_BUFFER is available
/// from the state value array on every target (the specs only reserve lanes for
/// them on non-architected-flat-scratch parts).
///
/// Word 1 of a V# carries \c base[47:32] in its low half only; the upper half
/// is stride and swizzle configuration, hence the mask.
///
/// \c soffset is added outside the swizzle while \c vaddr and \c inst_offset go
/// inside it, but that distinction does not change which *page* the access
/// lands on, so the plain sum is what the aperture test needs.
__attribute__((device)) inline const void *
bufferAccessFlatAddr(std::uint32_t RsrcLo, std::uint32_t RsrcHi,
                     std::uint32_t VOffset, std::uint32_t SOffset,
                     std::int32_t InstOffset) {
  const std::uint64_t Base =
      static_cast<std::uint64_t>(RsrcLo) |
      (static_cast<std::uint64_t>(RsrcHi & 0xFFFFu) << 32);
  const std::uint64_t Addr = Base + static_cast<std::uint64_t>(SOffset) +
                             static_cast<std::uint64_t>(VOffset) +
                             static_cast<std::uint64_t>(
                                 static_cast<std::int64_t>(InstOffset));
  return reinterpret_cast<const void *>(Addr);
}

/// \return \p SOffset adjusted for a buffer access, iff the address that access
/// was going to form falls inside this work-item's private segment.
///
/// The decision is the same aperture test the \c FLAT_* payloads use, applied
/// to the address reconstructed by \c bufferAccessFlatAddr . Deciding it from
/// the address rather than from the identity of the descriptor is what makes
/// this portable: it needs no state-value-array slot for the wavefront's
/// scratch base, and \c __builtin_amdgcn_is_private lowers to whichever
/// aperture source the target actually has --- \c SRC_PRIVATE_BASE , a code
/// object v5 implicit kernarg, or a v4 queue-pointer load --- via
/// \c SITargetLowering::getSegmentAperture .
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
                       std::uint32_t VOffset, std::uint32_t SOffset,
                       std::int32_t InstOffset) {
  if (__builtin_amdgcn_is_private(
          bufferAccessFlatAddr(RsrcLo, RsrcHi, VOffset, SOffset, InstOffset)))
    SOffset += instrumentationReserve() *
               static_cast<std::uint32_t>(__builtin_amdgcn_wavefrontsize());
  return SOffset;
}

/// Deposit \p N dwords into the destination registers the erased load defined.
/// Slots at or past \p N hold a null register and are never touched.
template <unsigned N>
__attribute__((device)) inline void
writeDwords(llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
            llvm::MCRegister D3, const std::uint32_t (&V)[4]) {
  luthier::writeReg<std::uint32_t>(D0, V[0]);
  if constexpr (N > 1)
    luthier::writeReg<std::uint32_t>(D1, V[1]);
  if constexpr (N > 2)
    luthier::writeReg<std::uint32_t>(D2, V[2]);
  if constexpr (N > 3)
    luthier::writeReg<std::uint32_t>(D3, V[3]);
}

} // namespace

namespace luthier::payloads {

//===----------------------------------------------------------------------===//
// Device payloads — flat scratch
//===----------------------------------------------------------------------===//

template <unsigned N>
__attribute__((device, used)) void
scratchLoadDwords(
    llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
    llvm::MCRegister D3, std::uint32_t VOffset, std::uint32_t SOffset,
    std::int32_t InstOffset) {
  const std::uint32_t Off =
      displacedPrivateOffset(VOffset, SOffset, InstOffset);
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != N; ++I)
    V[I] = *(Private<std::uint32_t> *)(Off + 4 * I);
  writeDwords<N>(D0, D1, D2, D3, V);
}

template <unsigned N>
__attribute__((device, used)) void
scratchStoreDwords(
    std::uint32_t V0, std::uint32_t V1, std::uint32_t V2, std::uint32_t V3,
    std::uint32_t VOffset, std::uint32_t SOffset, std::int32_t InstOffset) {
  const std::uint32_t Off =
      displacedPrivateOffset(VOffset, SOffset, InstOffset);
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != N; ++I)
    *(Private<std::uint32_t> *)(Off + 4 * I) = V[I];
}

template <typename T>
__attribute__((device, used)) void
scratchLoadSubDword(
    llvm::MCRegister Dst, std::uint32_t VOffset, std::uint32_t SOffset,
    std::int32_t InstOffset) {
  const std::uint32_t Off =
      displacedPrivateOffset(VOffset, SOffset, InstOffset);
  // The cast to std::int32_t before std::uint32_t is what sign-extends the
  // signed widths, matching SCRATCH_LOAD_SBYTE / _SSHORT.
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(
               *(Private<T> *)(Off))));
}

template <typename T>
__attribute__((device, used)) void
scratchStoreSubDword(
    std::uint32_t Value, std::uint32_t VOffset, std::uint32_t SOffset,
    std::int32_t InstOffset) {
  const std::uint32_t Off =
      displacedPrivateOffset(VOffset, SOffset, InstOffset);
  *(Private<T> *)(Off) = static_cast<T>(Value);
}

//===----------------------------------------------------------------------===//
// Device payloads — flat
//===----------------------------------------------------------------------===//

template <unsigned N>
__attribute__((device, used)) void
flatLoadDwords(
    llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
    llvm::MCRegister D3, std::uint64_t Addr, std::int32_t InstOffset) {
  auto *G = reinterpret_cast<std::uint32_t *>(
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset)));
  std::uint32_t V[4] = {0, 0, 0, 0};
  if (__builtin_amdgcn_is_private(static_cast<const void *>(G))) {
    const std::uint32_t Off = displacedPrivateOffsetOf(G);
    for (unsigned I = 0; I != N; ++I)
      V[I] = *(Private<std::uint32_t> *)(Off + 4 * I);
  } else {
    for (unsigned I = 0; I != N; ++I)
      V[I] = G[I];
  }
  writeDwords<N>(D0, D1, D2, D3, V);
}

template <unsigned N>
__attribute__((device, used)) void
flatStoreDwords(
    std::uint32_t V0, std::uint32_t V1, std::uint32_t V2, std::uint32_t V3,
    std::uint64_t Addr, std::int32_t InstOffset) {
  auto *G = reinterpret_cast<std::uint32_t *>(
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset)));
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  if (__builtin_amdgcn_is_private(static_cast<const void *>(G))) {
    const std::uint32_t Off = displacedPrivateOffsetOf(G);
    for (unsigned I = 0; I != N; ++I)
      *(Private<std::uint32_t> *)(Off + 4 * I) = V[I];
  } else {
    for (unsigned I = 0; I != N; ++I)
      G[I] = V[I];
  }
}

template <typename T>
__attribute__((device, used)) void
flatLoadSubDword(
    llvm::MCRegister Dst, std::uint64_t Addr, std::int32_t InstOffset) {
  auto *G = reinterpret_cast<T *>(
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset)));
  T Val;
  if (__builtin_amdgcn_is_private(static_cast<const void *>(G)))
    Val = *(Private<T> *)(displacedPrivateOffsetOf(G));
  else
    Val = *G;
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

template <typename T>
__attribute__((device, used)) void
flatStoreSubDword(
    std::uint32_t Value, std::uint64_t Addr, std::int32_t InstOffset) {
  auto *G = reinterpret_cast<T *>(
      Addr + static_cast<std::uint64_t>(static_cast<std::int64_t>(InstOffset)));
  if (__builtin_amdgcn_is_private(static_cast<const void *>(G)))
    *(Private<T> *)(displacedPrivateOffsetOf(G)) =
        static_cast<T>(Value);
  else
    *G = static_cast<T>(Value);
}

//===----------------------------------------------------------------------===//
// Device payloads — buffer
//===----------------------------------------------------------------------===//

template <unsigned N>
__attribute__((device, used)) void
bufferLoadDwords(
    llvm::MCRegister D0, llvm::MCRegister D1, llvm::MCRegister D2,
    llvm::MCRegister D3, __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo,
    std::uint32_t RsrcHi, std::uint32_t VOffset, std::uint32_t SOffset,
    std::int32_t InstOffset) {
  const int SOff = static_cast<int>(
      displacedBufferSOffset(RsrcLo, RsrcHi, VOffset, SOffset,
                             InstOffset) +
      static_cast<std::uint32_t>(InstOffset));
  std::uint32_t V[4] = {0, 0, 0, 0};
  for (unsigned I = 0; I != N; ++I)
    V[I] = __builtin_amdgcn_raw_buffer_load_b32(
        Rsrc, static_cast<int>(VOffset + 4 * I), SOff, 0);
  writeDwords<N>(D0, D1, D2, D3, V);
}

template <unsigned N>
__attribute__((device, used)) void
bufferStoreDwords(
    std::uint32_t V0, std::uint32_t V1, std::uint32_t V2, std::uint32_t V3,
    __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo, std::uint32_t RsrcHi,
    std::uint32_t VOffset, std::uint32_t SOffset, std::int32_t InstOffset) {
  const int SOff = static_cast<int>(
      displacedBufferSOffset(RsrcLo, RsrcHi, VOffset, SOffset,
                             InstOffset) +
      static_cast<std::uint32_t>(InstOffset));
  const std::uint32_t V[4] = {V0, V1, V2, V3};
  for (unsigned I = 0; I != N; ++I)
    __builtin_amdgcn_raw_buffer_store_b32(
        V[I], Rsrc, static_cast<int>(VOffset + 4 * I), SOff, 0);
}

template <typename T>
__attribute__((device, used)) void
bufferLoadSubDword(
    llvm::MCRegister Dst, __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo,
    std::uint32_t RsrcHi, std::uint32_t VOffset, std::uint32_t SOffset,
    std::int32_t InstOffset) {
  const int SOff = static_cast<int>(
      displacedBufferSOffset(RsrcLo, RsrcHi, VOffset, SOffset,
                             InstOffset) +
      static_cast<std::uint32_t>(InstOffset));
  T Val;
  if constexpr (sizeof(T) == 1)
    Val = static_cast<T>(__builtin_amdgcn_raw_buffer_load_b8(
        Rsrc, static_cast<int>(VOffset), SOff, 0));
  else
    Val = static_cast<T>(__builtin_amdgcn_raw_buffer_load_b16(
        Rsrc, static_cast<int>(VOffset), SOff, 0));
  luthier::writeReg<std::uint32_t>(
      Dst, static_cast<std::uint32_t>(static_cast<std::int32_t>(Val)));
}

template <typename T>
__attribute__((device, used)) void
bufferStoreSubDword(
    std::uint32_t Value, __amdgpu_buffer_rsrc_t Rsrc, std::uint32_t RsrcLo,
    std::uint32_t RsrcHi, std::uint32_t VOffset, std::uint32_t SOffset,
    std::int32_t InstOffset) {
  const int SOff = static_cast<int>(
      displacedBufferSOffset(RsrcLo, RsrcHi, VOffset, SOffset,
                             InstOffset) +
      static_cast<std::uint32_t>(InstOffset));
  if constexpr (sizeof(T) == 1)
    __builtin_amdgcn_raw_buffer_store_b8(static_cast<unsigned char>(Value),
                                         Rsrc, static_cast<int>(VOffset), SOff,
                                         0);
  else
    __builtin_amdgcn_raw_buffer_store_b16(static_cast<unsigned short>(Value),
                                          Rsrc, static_cast<int>(VOffset),
                                          SOff, 0);
}

//===----------------------------------------------------------------------===//
// Explicit instantiations
//===----------------------------------------------------------------------===//
//
// Device pass only. On the host pass the plugin that synthesizes host handles
// for `__device__` functions clones a template *specialization* when host code
// takes its address --- which `hookFor` below does --- and an explicit
// instantiation here would pre-empt that with a device-only definition the host
// then cannot link against.
#ifdef __HIP_DEVICE_COMPILE__
//
// Every payload the host side can reach through `lookupHandleName` has to be
// emitted into this translation unit's offload bundle, which means naming each
// instantiation here.

template __attribute__((device, used)) void scratchLoadDwords<1>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void scratchLoadDwords<2>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void scratchLoadDwords<3>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void scratchLoadDwords<4>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint32_t, std::uint32_t, std::int32_t);

template __attribute__((device, used)) void scratchStoreDwords<1>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void scratchStoreDwords<2>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void scratchStoreDwords<3>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void scratchStoreDwords<4>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);

template __attribute__((device, used)) void
scratchLoadSubDword<std::uint8_t>(
    llvm::MCRegister, std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
scratchLoadSubDword<std::int8_t>(
    llvm::MCRegister, std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
scratchLoadSubDword<std::uint16_t>(
    llvm::MCRegister, std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
scratchLoadSubDword<std::int16_t>(
    llvm::MCRegister, std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
scratchStoreSubDword<std::uint8_t>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
scratchStoreSubDword<std::uint16_t>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t);

template __attribute__((device, used)) void flatLoadDwords<1>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint64_t, std::int32_t);
template __attribute__((device, used)) void flatLoadDwords<2>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint64_t, std::int32_t);
template __attribute__((device, used)) void flatLoadDwords<3>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint64_t, std::int32_t);
template __attribute__((device, used)) void flatLoadDwords<4>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    std::uint64_t, std::int32_t);

template __attribute__((device, used)) void flatStoreDwords<1>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t,
    std::int32_t);
template __attribute__((device, used)) void flatStoreDwords<2>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t,
    std::int32_t);
template __attribute__((device, used)) void flatStoreDwords<3>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t,
    std::int32_t);
template __attribute__((device, used)) void flatStoreDwords<4>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t,
    std::int32_t);

template __attribute__((device, used)) void
flatLoadSubDword<std::uint8_t>(
    llvm::MCRegister, std::uint64_t, std::int32_t);
template __attribute__((device, used)) void
flatLoadSubDword<std::int8_t>(
    llvm::MCRegister, std::uint64_t, std::int32_t);
template __attribute__((device, used)) void
flatLoadSubDword<std::uint16_t>(
    llvm::MCRegister, std::uint64_t, std::int32_t);
template __attribute__((device, used)) void
flatLoadSubDword<std::int16_t>(
    llvm::MCRegister, std::uint64_t, std::int32_t);
template __attribute__((device, used)) void
flatStoreSubDword<std::uint8_t>(
    std::uint32_t, std::uint64_t, std::int32_t);
template __attribute__((device, used)) void
flatStoreSubDword<std::uint16_t>(
    std::uint32_t, std::uint64_t, std::int32_t);

template __attribute__((device, used)) void bufferLoadDwords<1>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void bufferLoadDwords<2>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void bufferLoadDwords<3>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void bufferLoadDwords<4>(
    llvm::MCRegister, llvm::MCRegister, llvm::MCRegister, llvm::MCRegister,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);

template __attribute__((device, used)) void bufferStoreDwords<1>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void bufferStoreDwords<2>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void bufferStoreDwords<3>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);
template __attribute__((device, used)) void bufferStoreDwords<4>(
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::int32_t);

template __attribute__((device, used)) void
bufferLoadSubDword<std::uint8_t>(
    llvm::MCRegister, __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
bufferLoadSubDword<std::int8_t>(
    llvm::MCRegister, __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
bufferLoadSubDword<std::uint16_t>(
    llvm::MCRegister, __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
bufferLoadSubDword<std::int16_t>(
    llvm::MCRegister, __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
bufferStoreSubDword<std::uint8_t>(
    std::uint32_t, __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::int32_t);
template __attribute__((device, used)) void
bufferStoreSubDword<std::uint16_t>(
    std::uint32_t, __amdgpu_buffer_rsrc_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::int32_t);

#endif // __HIP_DEVICE_COMPILE__

} // namespace luthier::payloads

//===----------------------------------------------------------------------===//
// Host side
//===----------------------------------------------------------------------===//

#ifndef __HIP_DEVICE_COMPILE__

namespace Payloads = luthier::payloads;

namespace luthier {

namespace {

/// What an application memory instruction does, in the terms the payloads are
/// written in.
struct AccessDesc {
  enum class Kind { Scratch, Flat, Buffer } Class;
  bool IsStore{false};
  /// Number of 32-bit words moved, 1..4. Zero means a sub-dword access, whose
  /// width is \c SubDwordBytes .
  unsigned NumDwords{0};
  /// 1 or 2 when \c NumDwords is zero.
  unsigned SubDwordBytes{0};
  /// Whether a sub-dword load sign-extends into its 32-bit destination.
  bool SignExtend{false};
};

/// Classify \p Name --- an AMDGPU opcode name such as
/// <tt>SCRATCH_LOAD_DWORDX2_SADDR</tt> --- into an \c AccessDesc .
///
/// This reads the opcode's name rather than switching over several hundred
/// enumerators. The width token is the same across every addressing form of a
/// family, so one match covers all of \c _SADDR / \c _ST / \c _SVS / vaddr for
/// flat scratch and \c _OFFEN / \c _OFFSET for buffers, and a family this pass
/// has never seen (\c _D16_HI , \c _LDS , \c _FORMAT_ , ...) falls out as
/// \c nullopt rather than being silently mis-widened.
std::optional<AccessDesc> classifyByName(llvm::StringRef Name,
                                         AccessDesc::Kind Class) {
  AccessDesc D;
  D.Class = Class;
  llvm::StringRef Rest;
  if (auto [Head, Tail] = Name.split("_LOAD_"); !Tail.empty()) {
    (void)Head;
    D.IsStore = false;
    Rest = Tail;
  } else if (auto [Head2, Tail2] = Name.split("_STORE_"); !Tail2.empty()) {
    (void)Head2;
    D.IsStore = true;
    Rest = Tail2;
  } else {
    return std::nullopt;
  }
  // The width token runs to the next underscore; whatever follows is the
  // addressing form, which the operand lookups below handle generically.
  llvm::StringRef Width = Rest.take_until([](char C) { return C == '_'; });
  if (Width == "DWORD")
    D.NumDwords = 1;
  else if (Width == "DWORDX2")
    D.NumDwords = 2;
  else if (Width == "DWORDX3")
    D.NumDwords = 3;
  else if (Width == "DWORDX4")
    D.NumDwords = 4;
  else if (Width == "UBYTE" || Width == "BYTE")
    D.SubDwordBytes = 1;
  else if (Width == "SBYTE") {
    D.SubDwordBytes = 1;
    D.SignExtend = true;
  } else if (Width == "USHORT" || Width == "SHORT")
    D.SubDwordBytes = 2;
  else if (Width == "SSHORT") {
    D.SubDwordBytes = 2;
    D.SignExtend = true;
  } else
    return std::nullopt;

  // Anything past the width token that is not a pure addressing-form suffix
  // changes the semantics in a way the payloads do not model --- `_D16` and
  // `_D16_HI` preserve the other half of the destination, `_LDS` writes LDS
  // instead of a VGPR.
  llvm::StringRef Suffix = Rest.drop_front(Width.size());
  if (Suffix.contains("D16") || Suffix.contains("LDS"))
    return std::nullopt;
  return D;
}

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

  // Linking the payload module in is not free, so hold it off until an
  // application instruction that can reach the private segment turns up.
  /// Names of every payload the linked module contributed.
  ///
  /// Each payload is `__attribute__((used))`, which is what makes the device
  /// compile emit it into the bundle at all --- nothing in that translation unit
  /// calls it. The flip side is that the linked module pins all of them via
  /// `llvm.compiler.used`, so their out-of-line bodies survive into
  /// `ProcessIntrinsicsAtIRLevelPass`, which rejects them: a payload names the
  /// register it writes back through its own parameter, and the intrinsic
  /// lowering needs a literal. `createInjectedPayload` force-inlines the
  /// payloads this pass actually uses, which fixes them up at the call site,
  /// but the ones it does not use stay behind. So all of them get unpinned and
  /// erased once every payload has been built.
  llvm::SmallVector<std::string, 64> LinkedPayloadNames;

  bool Linked = false;
  // The subtarget comes from a target MachineFunction rather than from the
  // instrumentation module's MachineModuleAnalysis: the two always agree (a
  // heterogeneous prototype is not supported), and a standalone
  // `luthier-llc -passes=` run does not register the IModule's module-level
  // analysis-manager proxy.
  const llvm::TargetMachine *TM = nullptr;
  auto ensureLinked = [&]() -> llvm::Error {
    if (Linked)
      return llvm::Error::success();
    assert(TM && "no target machine to pick a bitcode slice with");
    llvm::SubtargetFeatures Features(TM->getTargetFeatureString());
    auto SubOrErr = OffloadParser->parseModule(
        TM->getTargetTriple(), TM->getTargetCPU(), Features, Ctx);
    LUTHIER_RETURN_ON_ERROR(SubOrErr.takeError());
    std::unique_ptr<llvm::Module> Sub = std::move(*SubOrErr);
    Sub->setTargetTriple(IModule.getTargetTriple());
    Sub->setDataLayout(IModule.getDataLayout());
    // HIP gives a `__device__` function internal linkage, and `Linker` skips a
    // locally-linked definition nothing else in the module already refers to
    // (`ModuleLinker::linkIfNeeded`), so the payloads would be dropped on the
    // floor. Promoting them is safe and precise: `llvm.used` /
    // `llvm.compiler.used` is exactly the set the `used` attribute on each
    // payload put there, and their names are namespace-qualified template
    // specializations that cannot collide with the tool's own module.
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
    if (llvm::Linker(IModule).linkInModule(std::move(Sub)))
      return LUTHIER_MAKE_GENERIC_ERROR(
          "RebaseAppScratchAccessesPass: failed to link the pass's device-side "
          "payload module into the prototype's instrumentation module.");
    Linked = true;
    return llvm::Error::success();
  };


  /// Resolve one of the payload hooks to its \c llvm::Function in the IModule.
  auto hookFor = [&](auto *HostHandle) -> llvm::Expected<llvm::Function *> {
    LUTHIER_RETURN_ON_ERROR(ensureLinked());
    auto NameOrErr = OffloadParser->lookupHandleName(HostHandle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    llvm::Function *F = IModule.getFunction(*NameOrErr);
    if (!F || F->isDeclaration())
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "RebaseAppScratchAccessesPass: payload hook '{0}' is missing from "
          "the instrumentation module after linking.",
          *NameOrErr));
    return F;
  };

  /// Pick the payload for \p D . The width is a template parameter on the
  /// device side, so it has to be spelled out per case here.
  auto selectHook = [&](const AccessDesc &D) -> llvm::Expected<llvm::Function *> {
    switch (D.Class) {
    case AccessDesc::Kind::Scratch:
      if (D.NumDwords) {
        switch (D.NumDwords) {
        case 1:
          return D.IsStore ? hookFor(&Payloads::scratchStoreDwords<1>)
                           : hookFor(&Payloads::scratchLoadDwords<1>);
        case 2:
          return D.IsStore ? hookFor(&Payloads::scratchStoreDwords<2>)
                           : hookFor(&Payloads::scratchLoadDwords<2>);
        case 3:
          return D.IsStore ? hookFor(&Payloads::scratchStoreDwords<3>)
                           : hookFor(&Payloads::scratchLoadDwords<3>);
        default:
          return D.IsStore ? hookFor(&Payloads::scratchStoreDwords<4>)
                           : hookFor(&Payloads::scratchLoadDwords<4>);
        }
      }
      if (D.IsStore)
        return D.SubDwordBytes == 1
                   ? hookFor(&Payloads::scratchStoreSubDword<std::uint8_t>)
                   : hookFor(&Payloads::scratchStoreSubDword<std::uint16_t>);
      if (D.SubDwordBytes == 1)
        return D.SignExtend
                   ? hookFor(&Payloads::scratchLoadSubDword<std::int8_t>)
                   : hookFor(&Payloads::scratchLoadSubDword<std::uint8_t>);
      return D.SignExtend
                 ? hookFor(&Payloads::scratchLoadSubDword<std::int16_t>)
                 : hookFor(&Payloads::scratchLoadSubDword<std::uint16_t>);
    case AccessDesc::Kind::Flat:
      if (D.NumDwords) {
        switch (D.NumDwords) {
        case 1:
          return D.IsStore ? hookFor(&Payloads::flatStoreDwords<1>)
                           : hookFor(&Payloads::flatLoadDwords<1>);
        case 2:
          return D.IsStore ? hookFor(&Payloads::flatStoreDwords<2>)
                           : hookFor(&Payloads::flatLoadDwords<2>);
        case 3:
          return D.IsStore ? hookFor(&Payloads::flatStoreDwords<3>)
                           : hookFor(&Payloads::flatLoadDwords<3>);
        default:
          return D.IsStore ? hookFor(&Payloads::flatStoreDwords<4>)
                           : hookFor(&Payloads::flatLoadDwords<4>);
        }
      }
      if (D.IsStore)
        return D.SubDwordBytes == 1
                   ? hookFor(&Payloads::flatStoreSubDword<std::uint8_t>)
                   : hookFor(&Payloads::flatStoreSubDword<std::uint16_t>);
      if (D.SubDwordBytes == 1)
        return D.SignExtend ? hookFor(&Payloads::flatLoadSubDword<std::int8_t>)
                            : hookFor(&Payloads::flatLoadSubDword<std::uint8_t>);
      return D.SignExtend
                 ? hookFor(&Payloads::flatLoadSubDword<std::int16_t>)
                 : hookFor(&Payloads::flatLoadSubDword<std::uint16_t>);
    case AccessDesc::Kind::Buffer:
      if (D.NumDwords) {
        switch (D.NumDwords) {
        case 1:
          return D.IsStore ? hookFor(&Payloads::bufferStoreDwords<1>)
                           : hookFor(&Payloads::bufferLoadDwords<1>);
        case 2:
          return D.IsStore ? hookFor(&Payloads::bufferStoreDwords<2>)
                           : hookFor(&Payloads::bufferLoadDwords<2>);
        case 3:
          return D.IsStore ? hookFor(&Payloads::bufferStoreDwords<3>)
                           : hookFor(&Payloads::bufferLoadDwords<3>);
        default:
          return D.IsStore ? hookFor(&Payloads::bufferStoreDwords<4>)
                           : hookFor(&Payloads::bufferLoadDwords<4>);
        }
      }
      if (D.IsStore)
        return D.SubDwordBytes == 1
                   ? hookFor(&Payloads::bufferStoreSubDword<std::uint8_t>)
                   : hookFor(&Payloads::bufferStoreSubDword<std::uint16_t>);
      if (D.SubDwordBytes == 1)
        return D.SignExtend
                   ? hookFor(&Payloads::bufferLoadSubDword<std::int8_t>)
                   : hookFor(&Payloads::bufferLoadSubDword<std::uint8_t>);
      return D.SignExtend
                 ? hookFor(&Payloads::bufferLoadSubDword<std::int16_t>)
                 : hookFor(&Payloads::bufferLoadSubDword<std::uint16_t>);
    }
    llvm_unreachable("unhandled access class");
  };

  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *I64 = llvm::Type::getInt64Ty(Ctx);
  /// The LLVM type behind HIP's \c __amdgpu_buffer_rsrc_t .
  llvm::Type *RsrcTy = llvm::PointerType::get(Ctx, 8);
  auto constI32 = [&](uint64_t V) -> PayloadArg {
    return llvm::ConstantInt::get(I32, V);
  };

  llvm::Error Err = llvm::Error::success();
  llvm::SmallVector<llvm::MachineInstr *, 16> ToErase;

  P.forEachTargetMF(PAM, [&](llvm::MachineFunction &MF) {
    if (Err)
      return;
    const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
    const auto &TII = *ST.getInstrInfo();
    const auto &TRI = *ST.getRegisterInfo();
    TM = &MF.getTarget();

    for (llvm::MachineBasicBlock &MBB : MF) {
      for (llvm::MachineInstr &MI : MBB) {
        if (Err)
          return;
        std::optional<AccessDesc::Kind> Class;
        if (TII.isFLATScratch(MI))
          Class = AccessDesc::Kind::Scratch;
        else if (TII.isFLAT(MI) && !TII.isFLATGlobal(MI))
          Class = AccessDesc::Kind::Flat;
        // A buffer access can only reach the private segment on a target whose
        // stack *is* a buffer, i.e. one that still has a private segment
        // buffer. On architected-flat-scratch parts (gfx940+, gfx11+, gfx12+)
        // there is no such descriptor to query at all --- the wave gets
        // FLAT_SCRATCH from hardware and scratch is reached through
        // SCRATCH_*/FLAT --- and the same holds wherever
        // `enableFlatScratch()` moves the stack onto SCRATCH_*. In both cases
        // every MUBUF is global by construction, so it must be left alone.
        //
        // This is also what keeps the payload buildable there: the state value
        // array only reserves lanes for WAVEFRONT_PRIVATE_SEGMENT_BUFFER under
        // exactly this predicate (see `findLowestFreeLanes` in
        // StateValueArraySpecs.cpp), so a buffer payload emitted on an
        // architected-flat-scratch target would read a slot that has no lane
        // assignment and fail to lower.
        else if (TII.isMUBUF(MI) && !ST.hasArchitectedFlatScratch() &&
                 !ST.enableFlatScratch())
          Class = AccessDesc::Kind::Buffer;
        if (!Class)
          continue;

        llvm::StringRef Name = TII.getName(MI.getOpcode());
        std::optional<AccessDesc> D = classifyByName(Name, *Class);

        if (!D && *Class != AccessDesc::Kind::Scratch) {
          // A buffer access reaches scratch only through the application's
          // scratch V#, which requires idxen == 0; and a flat atomic on private
          // memory is not something the hardware supports, so LLVM expands
          // private atomics into a load and a store rather than emitting one.
          // Everything in these two classes that this pass does not recognize
          // is therefore global by construction and needs no rebasing.
          continue;
        }
        if (!D) {
          Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
              "RebaseAppScratchAccessesPass: '{0}' in '{1}' addresses the "
              "wavefront's private segment but is not a form this pass knows "
              "how to re-issue. Add a payload for it in "
              "RebaseAppScratchAccessesPass.hip.",
              Name, MF.getName()));
          return;
        }
        // A scratch buffer requires idxen == 0, so the indexed addressing forms
        // are global by construction. The index is folded into `vaddr` rather
        // than carried in an operand of its own, so the opcode's own name is
        // what distinguishes them.
        if (*Class == AccessDesc::Kind::Buffer &&
            (Name.ends_with("_IDXEN") || Name.ends_with("_BOTHEN")))
          continue;

        // Cache-policy bits are a performance hint on private memory --- it is
        // wave-private and coherent with nothing --- so they are dropped on the
        // scratch path. On the flat and buffer paths the access may be global,
        // where they are not droppable, and the re-issued access has no way to
        // spell them: the buffer builtins take a compile-time-constant aux
        // operand and a plain pointer dereference carries no policy at all.
        const int64_t CPol = namedImmOr0(MI, llvm::AMDGPU::OpName::cpol);
        if (CPol != 0 && *Class != AccessDesc::Kind::Scratch) {
          Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
              "RebaseAppScratchAccessesPass: '{0}' in '{1}' carries cache "
              "policy {2}, which the re-issued access cannot reproduce.",
              Name, MF.getName(), CPol));
          return;
        }

        const int64_t InstOffset =
            namedImmOr0(MI, llvm::AMDGPU::OpName::offset);
        const llvm::MachineOperand *VAddr =
            findNamedOp(MI, llvm::AMDGPU::OpName::vaddr);
        const llvm::MachineOperand *SAddr =
            findNamedOp(MI, llvm::AMDGPU::OpName::saddr);
        const llvm::MachineOperand *SOffset =
            findNamedOp(MI, llvm::AMDGPU::OpName::soffset);
        const llvm::MachineOperand *SRsrc =
            findNamedOp(MI, llvm::AMDGPU::OpName::srsrc);
        // The data operand's name is not uniform across the three classes: a
        // FLAT or SCRATCH load defines `vdst`, while a MUBUF load defines
        // `vdata` --- the same name its store reads. Try both rather than
        // keying off the direction.
        const llvm::MachineOperand *Data =
            findNamedOp(MI, llvm::AMDGPU::OpName::vdst);
        if (!Data)
          Data = findNamedOp(MI, llvm::AMDGPU::OpName::vdata);
        if (!Data || !Data->isReg() || !Data->getReg()) {
          Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
              "RebaseAppScratchAccessesPass: '{0}' in '{1}' has no data "
              "register operand to re-issue through.",
              Name, MF.getName()));
          return;
        }

        /// An address operand that may be absent, an immediate, or a register.
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

        /// The \p I -th 32-bit channel of the data operand.
        const llvm::MCRegister DataReg = Data->getReg().asMCReg();
        auto dataChannel = [&](unsigned I) -> llvm::MCRegister {
          if (D->NumDwords <= 1)
            return DataReg;
          return TRI.getSubReg(DataReg,
                               llvm::SIRegisterInfo::getSubRegFromChannel(I));
        };

        // Every payload takes its data operands first --- the destination
        // register ids a load writes back into, or the 32-bit channel values a
        // store reads --- and its address operands after. The dword payloads
        // always take four data slots; slots past the access width are unused
        // and get a null register.
        llvm::SmallVector<PayloadArg, 10> Args;
        if (D->NumDwords) {
          for (unsigned I = 0; I != 4; ++I) {
            if (I >= D->NumDwords) {
              Args.push_back(constI32(0));
              continue;
            }
            Args.push_back(D->IsStore
                               ? PayloadArg{RegArg{dataChannel(I), I32}}
                               : constI32(dataChannel(I).id()));
          }
        } else {
          Args.push_back(D->IsStore ? PayloadArg{RegArg{DataReg, I32}}
                                    : constI32(DataReg.id()));
        }

        switch (D->Class) {
        case AccessDesc::Kind::Scratch:
          Args.push_back(addrArg(VAddr, I32));
          Args.push_back(addrArg(SAddr, I32));
          Args.push_back(constI32(InstOffset));
          break;
        case AccessDesc::Kind::Flat:
          Args.push_back(addrArg(VAddr, I64));
          Args.push_back(constI32(InstOffset));
          break;
        case AccessDesc::Kind::Buffer: {
          if (!SRsrc || !SRsrc->isReg() || !SRsrc->getReg()) {
            Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                "RebaseAppScratchAccessesPass: '{0}' in '{1}' has no resource "
                "descriptor operand.",
                Name, MF.getName()));
            return;
          }
          const llvm::MCRegister Rsrc = SRsrc->getReg().asMCReg();
          Args.push_back(RegArg{Rsrc, RsrcTy});
          // Words 0 and 1 again, as plain integers: the runtime scratch test
          // needs the V#'s 48-bit base and __amdgpu_buffer_rsrc_t is opaque.
          Args.push_back(
              RegArg{TRI.getSubReg(Rsrc, llvm::AMDGPU::sub0), I32});
          Args.push_back(
              RegArg{TRI.getSubReg(Rsrc, llvm::AMDGPU::sub1), I32});
          Args.push_back(addrArg(VAddr, I32));
          Args.push_back(addrArg(SOffset, I32));
          Args.push_back(constI32(InstOffset));
          break;
        }
        }

        auto HookOrErr = selectHook(*D);
        if (!HookOrErr) {
          Err = HookOrErr.takeError();
          return;
        }
        auto PayloadOrErr =
            P.createInjectedPayload(**HookOrErr, MI, IFAM, Args);
        if (!PayloadOrErr) {
          Err = PayloadOrErr.takeError();
          return;
        }
        ToErase.push_back(&MI);
      }
    }
  });

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
