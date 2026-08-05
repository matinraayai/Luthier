//===-- MIRToIRTranslator.cpp ---------------------------------------------===//
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
/// \file MIRToIRTranslator.cpp
/// Implements a set of APIs used to translate machine functions and
/// individual machine instructions to LLVM IR.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MIRToIRTranslator.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/MIInlineAsmEmitter.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/Metadata.h"
#include "luthier/ToolCodeGen/RegValueMetadata.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include <GCNSubtarget.h>
#include <SIDefines.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <SIModeRegisterDefaults.h>
#include <SIRegisterInfo.h>
#include <Utils/AMDGPUBaseInfo.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/Analysis/CallGraph.h>
#include <llvm/Analysis/InstSimplifyFolder.h>
#include <llvm/Analysis/InstructionSimplify.h>
#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/ValueHandle.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Transforms/Utils/Local.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-mir-to-ir"

namespace {
template <typename Tag, typename Tag::type MemPtr> struct Access {
  friend typename Tag::type get(Tag) { return MemPtr; }
};

/// Friend ADL trick to allow access to the private basic block field of
/// machine basic block
/// Unlike what LLVM assumes (IR comes after MIR), we have to construct the
/// IR basic block after we have the machine basic block
struct TagBB {
  using type = const llvm::BasicBlock *llvm::MachineBasicBlock::*;

  friend type get(TagBB);
};

template struct Access<TagBB, &llvm::MachineBasicBlock::BB>;

} // namespace

namespace luthier {

/// If \p Reg is not a VGPR/AGPR (i.e. SGPR, SCC, etc.)
/// attach \c !amdgpu.uniform metadata to \p I to mark it as uniform.
void inline annotateUniformIfNeeded(llvm::Instruction *I,
                                    const llvm::SIRegisterInfo &TRI,
                                    llvm::MCRegister Reg) {
  if (const llvm::TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Reg);
      RC && !llvm::SIRegisterInfo::isAGPRClass(RC) &&
      !llvm::SIRegisterInfo::isVGPRClass(RC))
    I->setMetadata("amdgpu.uniform", llvm::MDNode::get(I->getContext(), {}));
}

static void
getRegisterFileArgOrder(const llvm::GCNSubtarget &ST,
                        llvm::SmallVector<llvm::MCRegister> &ABIRegFileIdx) {
  ABIRegFileIdx.push_back(llvm::AMDGPU::SGPR0);
  ABIRegFileIdx.push_back(llvm::AMDGPU::isGFX9Plus(ST) ? llvm::AMDGPU::TTMP0
                                                       : llvm::AMDGPU::TBA_LO);
  ABIRegFileIdx.push_back(llvm::AMDGPU::isNotGFX10Plus(ST)
                              ? llvm::AMDGPU::M0
                              : llvm::AMDGPU::SGPR_NULL);
  if (llvm::AMDGPU::isNotGFX9Plus(ST))
    ABIRegFileIdx.push_back(llvm::AMDGPU::SRC_SHARED_BASE);
  ABIRegFileIdx.push_back(llvm::AMDGPU::SRC_VCCZ);
  ABIRegFileIdx.push_back(llvm::AMDGPU::VGPR0);
  if (ST.hasMAIInsts())
    ABIRegFileIdx.push_back(llvm::AMDGPU::AGPR0);
}

/// Populates \p RegFileSize with the canonical half-slot size of each register
/// file base used by the translator's IR ABI, derived from \p ST and the
/// declared \p NumSGPRs / \p NumVGPRs. Also reports the subtarget-dependent
/// TTMP and EXEC region base registers via \p TTMPBaseReg / \p ExecBaseReg.
/// Shared by \c initRegFileLayouts (instance setup) and
/// \c computeStandardDeviceFunctionType (stateless prototype factory) so both
/// paths agree on the same table.
static void computeRegFileSizes(
    const llvm::GCNSubtarget &ST, unsigned NumSGPRs, unsigned NumVGPRs,
    llvm::SmallDenseMap<llvm::MCRegister, unsigned> &RegFileSize,
    llvm::MCRegister &TTMPBaseReg, unsigned &ExecBaseReg) {
  TTMPBaseReg =
      llvm::AMDGPU::isGFX9Plus(ST) ? llvm::AMDGPU::TTMP0 : llvm::AMDGPU::TBA_LO;
  ExecBaseReg = llvm::AMDGPU::isNotGFX10Plus(ST) ? llvm::AMDGPU::M0
                                                 : llvm::AMDGPU::SGPR_NULL;
  unsigned NumApertureSregs = llvm::AMDGPU::isGFX9_GFX10(ST)  ? 10
                              : llvm::AMDGPU::isGFX11Plus(ST) ? 8
                                                              : 0;
  RegFileSize[llvm::AMDGPU::SGPR0] = 2u * NumSGPRs;
  /// TTMP region has 16 registers across all targets; if a new generation
  /// comes with a different encoding, this must be updated
  RegFileSize[TTMPBaseReg] = 2u * 16;
  /// There are 4 slots in the exec mask reg file; we keep SGPR_NULL even on
  /// targets that don't support it
  RegFileSize[ExecBaseReg] = 2u * 4;
  RegFileSize[llvm::AMDGPU::SRC_VCCZ] = 6;
  RegFileSize[llvm::AMDGPU::SRC_SHARED_BASE] = NumApertureSregs;
  RegFileSize[llvm::AMDGPU::VGPR0] = 2u * NumVGPRs;
  RegFileSize[llvm::AMDGPU::AGPR0] = ST.hasMAIInsts() ? 2u * NumVGPRs : 0u;
  RegFileSize[llvm::AMDGPU::MODE] = 1 << 7;
}

/// Decode the "denormal-fp-math[-f32]" attribute value (e.g.
/// "preserve-sign,ieee" or "ieee,preserve-sign") into the AMDGPU
/// \c FP_DENORM_* encoding used by the MODE register.
///
/// LLVM encodes the attribute as "<output>,<input>" (NOT input-first); an
/// empty or absent input component defaults to the output component. See
/// \c llvm::parseDenormalFPAttribute in llvm/ADT/FloatingPointMode.h.
static uint32_t decodeDenormAttr(llvm::StringRef AttrVal) {
  auto [OutStr, InStr] = AttrVal.split(',');
  OutStr = OutStr.trim();
  InStr = InStr.trim();
  bool OutFlush = (OutStr == "preserve-sign");
  bool InFlush = InStr.empty() ? OutFlush : (InStr == "preserve-sign");
  if (InFlush && OutFlush)
    return FP_DENORM_FLUSH_IN_FLUSH_OUT;
  if (OutFlush)
    return FP_DENORM_FLUSH_OUT;
  if (InFlush)
    return FP_DENORM_FLUSH_IN;
  return FP_DENORM_FLUSH_NONE;
}

static llvm::Value *getOrCreateIntOrPtrTypeForReg(
    llvm::DenseMap<llvm::Type *, llvm::Value *> &ValueEntries,
    llvm::IRBuilderBase &Builder) {
  assert(!ValueEntries.empty() && "Value entry map is empty");
  llvm::Value *VecIntOrPtrVal{nullptr};
  for (auto &[T, V] : ValueEntries) {
    if (T->isIntOrPtrTy())
      return V;
    if (T->isIntOrIntVectorTy() || T->isPtrOrPtrVectorTy())
      VecIntOrPtrVal = V;
  }
  /// If we couldn't find a pointer or an int type, do a bitcast on the first
  /// value in the map
  if (!VecIntOrPtrVal) {
    auto &[T, V] = *ValueEntries.begin();
    llvm::Type *OutTy = Builder.getIntNTy(T->getPrimitiveSizeInBits());
    VecIntOrPtrVal = Builder.CreateBitOrPointerCast(V, OutTy);
    ValueEntries[OutTy] = VecIntOrPtrVal;
  }
  return VecIntOrPtrVal;
}

static llvm::Value *getOrCreateIntOrFloatTypeForReg(
    llvm::DenseMap<llvm::Type *, llvm::Value *> &ValueEntries,
    llvm::IRBuilderBase &Builder) {
  assert(!ValueEntries.empty() && "Value entry map is empty");
  for (auto &[T, V] : ValueEntries) {
    if (T->isIntOrIntVectorTy() || T->isFPOrFPVectorTy())
      return V;
  }
  /// No int/FP entry exists (e.g. only pointer-typed values). Bitcast the
  /// first entry to an integer of the same bit width. Use
  /// \c getPrimitiveSizeInBits (NOT \c getIntegerBitWidth, which asserts on
  /// non-integer types), matching \c getOrCreateIntOrPtrTypeForReg.
  auto &[T, V] = *ValueEntries.begin();
  llvm::Type *OutTy = Builder.getIntNTy(T->getPrimitiveSizeInBits());
  llvm::Value *Cast = Builder.CreateBitOrPointerCast(V, OutTy);
  ValueEntries[OutTy] = Cast;
  return Cast;
}

/// Given a non-empty set of values mapped to the same register and their
/// types, manifests a vector type that breaks down the register into
/// scalar integer elements with \p ElemWidth as their width
/// Useful for 'extractelement'/'insertelement' indexing
///
/// \p TotalWidth is the authoritative register-slot width in bits, as
/// known by the caller (= NumHalves * RegGranule). Stored entries whose
/// primitive size disagrees with \p TotalWidth (e.g. a stray bitcast left
/// behind by a semantic that wrote a wider-than-the-slot value) are
/// ignored when materializing the requested vector — we always synthesize
/// from an entry that matches the slot width.
static llvm::Value *breakdownToVecTyFromAvailableValues(
    llvm::DenseMap<llvm::Type *, llvm::Value *> &ValueEntries,
    unsigned TotalWidth, unsigned ElemWidth, llvm::IRBuilderBase &Builder) {
  assert(!ValueEntries.empty() && "Empty value entry map");
  assert(TotalWidth != 0 && "TotalWidth must be provided by caller");
  assert(TotalWidth % ElemWidth == 0);
  unsigned NumElems = TotalWidth / ElemWidth;
  auto *VecTy =
      llvm::FixedVectorType::get(Builder.getIntNTy(ElemWidth), NumElems);
  if (auto ValueEntryIt = ValueEntries.find(VecTy);
      ValueEntryIt != ValueEntries.end()) {
    return ValueEntryIt->second;
  }
  // Find a width-matching entry to bitcast from. Pointer types report 0
  // from getPrimitiveSizeInBits without a DataLayout, so they are
  // skipped. If no entry matches TotalWidth, fall back to the
  // int-or-float helper (which itself bitcasts the first entry to its
  // own width and may produce a wrong-width pivot — preserved as a
  // last-resort path).
  llvm::Value *Pivot = nullptr;
  for (auto &[T, V] : ValueEntries) {
    if (T->getPrimitiveSizeInBits() == TotalWidth &&
        (T->isIntOrIntVectorTy() || T->isFPOrFPVectorTy())) {
      Pivot = V;
      break;
    }
  }
  if (!Pivot)
    Pivot = getOrCreateIntOrFloatTypeForReg(ValueEntries, Builder);
  /// The fallback pivot may not match the requested slot width. Catch that
  /// here with a clear message rather than letting it surface as an opaque
  /// "invalid bitcast" assertion deep inside IRBuilder.
  assert(Pivot->getType()->getPrimitiveSizeInBits() == TotalWidth &&
         "no register-value entry matches the requested slot width; cannot "
         "materialize the vector without corrupting the value");
  llvm::Value *Out = Builder.CreateBitOrPointerCast(Pivot, VecTy);
  ValueEntries[VecTy] = Out;
  return Out;
}

void MIRToIRTranslator::invalidateOverlaps(RegValueMap &State,
                                           const RegFileKey &WrittenRegKey,
                                           llvm::IRBuilderBase &Builder) {
  llvm::MCRegister BaseReg = std::get<0>(WrittenRegKey);
  const unsigned WStart = std::get<1>(WrittenRegKey);
  const unsigned WNumHalves = std::get<2>(WrittenRegKey);
  const unsigned WEnd = WStart + WNumHalves;
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] invalidateOverlaps: "
             << "base=" << TRI.getName(BaseReg) << " offset=" << WStart
             << " halves=" << WNumHalves << " end=" << WEnd << "\n");

  struct Preserve {
    uint32_t Offset;
    uint32_t NumHalves;
    llvm::Value *Val;
  };
  llvm::SmallVector<RegFileKey, 8> ToErase;
  llvm::SmallVector<Preserve, 4> ToPreserve;

  for (auto &[StoredKey, Entry] : State) {
    if (std::get<0>(StoredKey) != BaseReg)
      continue;
    const uint32_t SStart = std::get<1>(StoredKey);
    const uint32_t SNumHalves = std::get<2>(StoredKey);
    const uint32_t SEnd = SStart + SNumHalves;

    /// No overlap.
    if (SEnd <= WStart || SStart >= WEnd)
      continue;

    /// Skip the exact slot we're about to write — \c setRegOperandValue
    /// will overwrite it.
    if (SStart == WStart && SNumHalves == WNumHalves)
      continue;

    /// Stored ⊂ Written: fully covered, drop it.
    if (SStart >= WStart && SEnd <= WEnd) {
      LLVM_DEBUG(
          luthier::dbgs()
          << "  Fully covered (Stored ⊂ Written), erasing stored key: base="
          << TRI.getName(std::get<0>(StoredKey)) << " offset=" << SStart
          << " halves=" << SNumHalves << "\n");
      ToErase.push_back(StoredKey);
      continue;
    }

    /// Written ⊂ Stored: partial overwrite of a super-register. Preserve
    /// the non-overlapping regions as the largest uniform chunk size that
    /// divides both regions, so a later read can re-compose.
    if (SStart <= WStart && SEnd >= WEnd) {
      LLVM_DEBUG(luthier::dbgs()
                 << "  Partial overwrite (written ⊂ stored), preserving "
                    "non-overlapping parts of stored key: base="
                 << TRI.getName(std::get<0>(StoredKey)) << " offset=" << SStart
                 << " halves=" << SNumHalves << "\n");

      // Compute optimal chunk size as the GCD of the two preserved region
      // sizes and the written region size. Including \c WNumHalves is
      // required: the chunk size has to divide the full stored width
      // (\c LeftSize + \c WNumHalves + \c RightSize), otherwise
      // \c breakdownToVecTyFromAvailableValues cannot evenly partition the
      // stored vector.
      const uint32_t LeftSize = WStart - SStart; // may be 0
      const uint32_t RightSize = SEnd - WEnd;    // may be 0
      // std::gcd treats 0 as the identity, so this works whether LeftSize or
      // RightSize is zero.
      uint32_t OptHalves = std::gcd(std::gcd(LeftSize, RightSize), WNumHalves);

      const unsigned ElemWidth = OptHalves * RegGranule;
      const unsigned StoredTotalWidth = SNumHalves * RegGranule;
      llvm::Value *Vec = breakdownToVecTyFromAvailableValues(
          Entry, StoredTotalWidth, ElemWidth, Builder);

      auto preserveRegion = [&](uint32_t RegionStart, uint32_t RegionEnd) {
        const uint32_t NumChunks = (RegionEnd - RegionStart) / OptHalves;
        for (uint32_t CI = 0; CI < NumChunks; ++CI) {
          uint32_t AbsOffset = RegionStart + CI * OptHalves;
          uint32_t SrcIdx = (AbsOffset - SStart) / OptHalves;
          LLVM_DEBUG(luthier::dbgs()
                     << "  Preserving " << OptHalves << " halves at offset "
                     << AbsOffset << "\n");
          llvm::Value *Elem = Builder.CreateExtractElement(Vec, SrcIdx);
          ToPreserve.push_back({AbsOffset, OptHalves, Elem});
        }
      };
      if (LeftSize)
        preserveRegion(SStart, WStart);
      if (RightSize)
        preserveRegion(WEnd, SEnd);
      ToErase.push_back(StoredKey);
      continue;
    }
    LLVM_DEBUG(luthier::dbgs()
               << "  Partial overlap, erasing stored key: base="
               << TRI.getName(std::get<0>(StoredKey)) << " offset=" << SStart
               << " halves=" << SNumHalves << "\n");
    ToErase.push_back(StoredKey);
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] invalidateOverlaps: Erasing "
             << ToErase.size() << " entries, preserving " << ToPreserve.size()
             << " partial entries\n");

  for (auto &K : ToErase) {
    LLVM_DEBUG(luthier::dbgs()
               << "  Deleting entry: [" << TRI.getName(std::get<0>(K)) << ", "
               << std::get<1>(K) << ", " << std::get<2>(K) << "]\n");
    State.erase(K);
  }
  for (const Preserve &P : ToPreserve) {
    LLVM_DEBUG(luthier::dbgs()
               << "  Restoring preserved entry at offset " << P.Offset
               << ", halves: " << P.NumHalves << "\n");
    State[std::make_tuple(BaseReg, P.Offset, P.NumHalves)][P.Val->getType()] =
        P.Val;
  }
}

llvm::Value *MIRToIRTranslator::extractChunkFromSource(
    RegValueMap &State, const RegFileKey &RegKey, unsigned VecChunkSize,
    unsigned Idx, unsigned NumChunks, llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] extractChunkFromSource: "
                                "base="
                             << TRI.getName(std::get<0>(RegKey))
                             << " offset=" << std::get<1>(RegKey)
                             << " Idx=" << Idx << " NumChunks=" << NumChunks
                             << " chunkSize=" << VecChunkSize << "\n";);
  auto &RegValueMap = State[RegKey];
  const unsigned KeyTotalWidth = std::get<2>(RegKey) * RegGranule;
  unsigned VecChunkRegGranMul = VecChunkSize / RegGranule;
  unsigned ChunkSizeInRegGran = NumChunks * VecChunkRegGranMul;
  llvm::Type *ChunkIntTy = Builder.getIntNTy(VecChunkSize);

  // Fast path: source width tiles cleanly into VecChunkSize lanes. Use
  // vector extractelement.
  if (KeyTotalWidth % VecChunkSize == 0) {
    llvm::Value *TheVec = breakdownToVecTyFromAvailableValues(
        RegValueMap, KeyTotalWidth, VecChunkSize, Builder);

    if (NumChunks == 1)
      return Builder.CreateExtractElement(TheVec, Idx);

    auto *ChunkTy = llvm::FixedVectorType::get(ChunkIntTy, NumChunks);
    llvm::Value *Chunk = llvm::PoisonValue::get(ChunkTy);
    for (uint32_t I = 0; I < NumChunks; ++I) {
      llvm::Value *E = Builder.CreateExtractElement(TheVec, Idx + I);
      State[std::make_tuple(std::get<0>(RegKey), I * VecChunkRegGranMul,
                            ChunkSizeInRegGran)][E->getType()] = E;
      Chunk = Builder.CreateInsertElement(Chunk, E, I);
    }
    return Chunk;
  }

  // Slow path: source width is coprime with VecChunkSize (e.g. source
  // has 3 halves, callers want 2-half lanes). The vector path can't
  // represent that. Fall back to a flat integer view and lshr+trunc.
  llvm::Type *FlatTy = Builder.getIntNTy(KeyTotalWidth);
  llvm::Value *Flat = nullptr;
  if (auto It = RegValueMap.find(FlatTy); It != RegValueMap.end()) {
    Flat = It->second;
  } else {
    llvm::Value *Pivot = getOrCreateIntOrFloatTypeForReg(RegValueMap, Builder);
    Flat = Builder.CreateBitOrPointerCast(Pivot, FlatTy);
    RegValueMap[FlatTy] = Flat;
  }

  auto ExtractOne = [&](unsigned ElemIdx) -> llvm::Value * {
    llvm::Value *Shifted = Builder.CreateLShr(
        Flat, llvm::ConstantInt::get(FlatTy, ElemIdx * VecChunkSize));
    return Builder.CreateTrunc(Shifted, ChunkIntTy);
  };

  if (NumChunks == 1)
    return ExtractOne(Idx);

  auto *ChunkTy = llvm::FixedVectorType::get(ChunkIntTy, NumChunks);
  llvm::Value *Chunk = llvm::PoisonValue::get(ChunkTy);
  for (uint32_t I = 0; I < NumChunks; ++I) {
    llvm::Value *E = ExtractOne(Idx + I);
    State[std::make_tuple(std::get<0>(RegKey), I * VecChunkRegGranMul,
                          ChunkSizeInRegGran)][E->getType()] = E;
    Chunk = Builder.CreateInsertElement(Chunk, E, I);
  }
  return Chunk;
}

llvm::Value *MIRToIRTranslator::materializeFromOverlapping(
    RegValueMap &State, const llvm::MachineBasicBlock &MBB,
    const RegFileKey &ReadKeyReg, llvm::IRBuilderBase &Builder,
    llvm::Type &RegType) {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] materializeFromOverlapping\n");

  llvm::MCRegister BaseReg = std::get<0>(ReadKeyReg);
  const uint32_t RStart = std::get<1>(ReadKeyReg);
  const uint32_t RNumHalves = std::get<2>(ReadKeyReg);
  /// Note: End is exclusive
  const uint32_t REnd = RStart + RNumHalves;

  // Step 1: Collect all overlapping entries
  struct OverlapInfo {
    uint32_t SrcOffset;
    uint32_t SrcNumHalves;
    RegFileKey RegKey;
    uint32_t OverlapStart;
    uint32_t OverlapEnd;
  };

  llvm::SmallVector<OverlapInfo, 8> Overlaps;
  for (auto &[Key, Entry] : State) {
    if (std::get<0>(Key) != BaseReg)
      continue;
    const uint32_t SOffset = std::get<1>(Key);
    const uint32_t SEnd = SOffset + std::get<2>(Key);
    const uint32_t OStart = std::max(SOffset, RStart);
    const uint32_t OEnd = std::min(SEnd, REnd);
    if (OStart < OEnd) {
      Overlaps.push_back({SOffset, SEnd - SOffset, Key, OStart, OEnd});
    }
  }

  // Step 2: Handle no overlaps case
  if (Overlaps.empty()) {
    if (!MBB.pred_empty()) {
      // Create PHI for entire register
      llvm::PHINode *Phi = Builder.CreatePHI(&RegType, MBB.pred_size());
      ToBeFixedPhis.emplace_back(&MBB, ReadKeyReg, Phi);
      State[ReadKeyReg][&RegType] = Phi;
      return Phi;
    } else {
      // Entry block - freeze(poison)
      llvm::Value *InitVal =
          Builder.CreateFreeze(llvm::PoisonValue::get(&RegType));
      State[ReadKeyReg][&RegType] = InitVal;
      return InitVal;
    }
  }

  // Step 3: Sort overlaps by the size of overlap with the target register
  llvm::sort(Overlaps, [](const OverlapInfo &A, const OverlapInfo &B) {
    return (A.OverlapEnd - A.OverlapStart) > (B.OverlapEnd - B.OverlapStart);
  });

  // Step 4: Build coverage map and identify overlapping chunks
  llvm::BitVector Covered(REnd - RStart, false);

  struct OverlapChunkInfo {
    OverlapInfo *Src;
    uint32_t SrcChunkStart; // Position in source (in halves)
    uint32_t ChunkStart; // Position in result (in halves, relative to WStart)
    uint32_t ChunkEnd;   // End position in result (exclusive)
  };
  llvm::SmallVector<OverlapChunkInfo, 8> OverlapChunks;

  for (OverlapInfo &Overlap : Overlaps) {
    for (uint32_t H = Overlap.OverlapStart; H < Overlap.OverlapEnd;) {
      if (Covered[H - RStart]) {
        H++;
        continue;
      }
      uint32_t ChunkStart = H;
      while (H < Overlap.OverlapEnd && !Covered[H - RStart]) {
        Covered[H - RStart] = true;
        H++;
      }
      uint32_t ChunkEnd = H;
      OverlapChunks.push_back({&Overlap, ChunkStart - Overlap.SrcOffset,
                               ChunkStart - RStart, ChunkEnd - RStart});
    }
  }

  // Step 5: Handle uncovered chunks and add them to the coverage map
  struct NonOverlapChunkInfo {
    RegFileKey KeyReg;
    unsigned ChunkStart;
    uint32_t ChunkEnd;
  };
  llvm::SmallVector<NonOverlapChunkInfo, 8> NonOverlapChunks;

  for (uint32_t H = RStart; H < REnd;) {
    if (Covered[H - RStart]) {
      H++;
      continue;
    }
    uint32_t RangeStart = H;
    while (H < REnd && !Covered[H - RStart]) {
      H++;
    }
    uint32_t RangeEnd = H;
    uint32_t RangeNumHalves = RangeEnd - RangeStart;

    llvm::Type *ValTy = Builder.getIntNTy(RangeNumHalves * 16u);
    llvm::Value *DefaultVal = nullptr;
    RegFileKey NonOverlappingSubKey =
        std::make_tuple(BaseReg, RangeStart, RangeNumHalves);
    if (MBB.pred_empty()) {
      // Entry block - freeze(poison) for the missing value
      DefaultVal = Builder.CreateFreeze(llvm::PoisonValue::get(ValTy));
    } else {
      // Has predecessors - create PHI for the missing value
      llvm::PHINode *Phi = Builder.CreatePHI(ValTy, MBB.pred_size());
      ToBeFixedPhis.emplace_back(&MBB, NonOverlappingSubKey, Phi);
      DefaultVal = Phi;
    }
    State[NonOverlappingSubKey][ValTy] = DefaultVal;
    NonOverlapChunks.push_back({NonOverlappingSubKey, RangeStart, RangeEnd});
  }

  unsigned OptimalNumHalves =
      std::accumulate(OverlapChunks.begin(), OverlapChunks.end(),
                      OverlapChunks[0].ChunkEnd - OverlapChunks[0].ChunkStart,
                      [](unsigned A, OverlapChunkInfo &B) {
                        return std::gcd(A, B.ChunkEnd - B.ChunkStart);
                      });
  OptimalNumHalves =
      std::accumulate(NonOverlapChunks.begin(), NonOverlapChunks.end(),
                      OptimalNumHalves, [](unsigned A, NonOverlapChunkInfo &B) {
                        return std::gcd(A, B.ChunkEnd - B.ChunkStart);
                      });

  // Step 6: Construct a vector type to materialize chunks
  auto *WorkingTy = llvm::FixedVectorType::get(
      Builder.getIntNTy(OptimalNumHalves * RegGranule),
      RNumHalves / OptimalNumHalves);
  llvm::Value *Result = llvm::PoisonValue::get(WorkingTy);

  unsigned OptimalChunkSizeInBits = RegGranule * OptimalNumHalves;

  // InsertChunkFn inserts chunks into Result.
  // SrcChunkStart: source-relative start offset (in halves) within K.
  // ChunkStart / ChunkEnd: RStart-relative offsets (in halves) in the result.
  auto InsertChunkFn = [&](unsigned SrcChunkStart, unsigned ChunkStart,
                           unsigned ChunkEnd, const RegFileKey &K) {
    unsigned NumChunks = ChunkEnd - ChunkStart;
    for (unsigned CI = 0; CI < NumChunks; CI += OptimalNumHalves) {
      // Cache key must use the absolute position in the register file.
      RegFileKey SubRegKey =
          std::make_tuple(BaseReg, RStart + ChunkStart + CI, OptimalNumHalves);
      unsigned SrcElIdx = (SrcChunkStart + CI) / OptimalNumHalves;
      // DestElIdx indexes into WorkingTy which starts at RStart.
      unsigned DestElIdx = (ChunkStart + CI) / OptimalNumHalves;
      llvm::Value *ChunkVal = extractChunkFromSource(
          State, K, OptimalChunkSizeInBits, SrcElIdx, 1, Builder);
      State[SubRegKey][ChunkVal->getType()] = ChunkVal;
      ChunkVal = Builder.CreateBitOrPointerCast(
          ChunkVal, Builder.getIntNTy(OptimalChunkSizeInBits));
      State[SubRegKey][ChunkVal->getType()] = ChunkVal;
      Result = Builder.CreateInsertElement(Result, ChunkVal, DestElIdx);
    }
  };

  // Step 7: Extract and insert chunks.
  // NonOverlapChunks store absolute offsets; normalize to RStart-relative
  // before calling InsertChunkFn (which expects RStart-relative ChunkStart).
  for (auto &C : NonOverlapChunks) {
    InsertChunkFn(0, C.ChunkStart - RStart, C.ChunkEnd - RStart, C.KeyReg);
  }

  for (auto &C : OverlapChunks) {
    InsertChunkFn(C.SrcChunkStart, C.ChunkStart, C.ChunkEnd, C.Src->RegKey);
  }

  // Step 8: Final bitcast to requested type
  return Builder.CreateBitOrPointerCast(Result, &RegType);
}

llvm::Value &
MIRToIRTranslator::getOperandAsValue(const llvm::MachineBasicBlock &MBB,
                                     llvm::MCRegister Reg,
                                     llvm::Type *OutRegType) {
  llvm::StringRef RegName = TRI.getName(Reg);
  std::string RegValName = getRegValueName(Reg);

  LLVM_DEBUG(luthier::dbgs()
             << llvm::formatv("[MIRToIRTranslator] Materializing register {0} "
                              "in MBB {1}\n",
                              RegName, MBB.getNumber()));
  (void)RegName;

  auto *BB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock());
  assert(BB && "MBB does not have an IR basic block");

  llvm::Instruction *TermInst = BB->getTerminatorOrNull();

  llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
      Builder(BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
              llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                annotateUniformIfNeeded(I, TRI, Reg);
                LLVM_DEBUG(
                    luthier::dbgs()
                    << "[MIRToIRTranslator] Inserting reg read instruction "
                    << *I << "\n");
              }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  return getOperandAsValue(MBB, getRegFileKey(Reg), Builder, OutRegType);
}

llvm::Value &MIRToIRTranslator::getOperandAsValue(
    const llvm::MachineBasicBlock &MBB, const RegFileKey &Key,
    llvm::IRBuilderBase &Builder, llvm::Type *OutRegType) {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] getOperandAsValue: MBB " << MBB.getNumber()
             << " base=" << TRI.getName(std::get<0>(Key)) << " offset="
             << std::get<1>(Key) << " halves=" << std::get<2>(Key) << "\n");
  RegValueMap &State = VM[MBB];
  /// ---- Bounds check -------------------------------------------------
  /// Out-of-range access returns the file's base register value (s0/v0/
  /// a0). Hardware semantics: each 32-bit slot of an OOR multi-slot read
  /// returns base-reg's value; writes are dropped.
  llvm::MCRegister BaseReg = std::get<0>(Key);
  unsigned Offset = std::get<1>(Key);
  unsigned NumHalves = std::get<2>(Key);
  unsigned Allocated = RegFileSize.at(BaseReg);
  if (Offset + NumHalves > Allocated) {
    assert(Offset != 0 &&
           "offset 0 is not in range of the register file allocation");
    Offset = 0;
  }

  if (!OutRegType)
    OutRegType = Builder.getIntNTy(std::get<2>(Key) * RegGranule);

  /// ---- Normal file-keyed lookup -------------------------------------

  // Exact match.
  if (auto It = State.find(Key); It != State.end()) {
    auto &VTM = It->second;
    if (auto V = VTM.find(OutRegType); V != VTM.end())
      return *V->getSecond();
    llvm::Value *CastVal = getOrCreateIntOrPtrTypeForReg(VTM, Builder);
    llvm::Value *Out = Builder.CreateBitOrPointerCast(CastVal, OutRegType);
    VTM[OutRegType] = Out;
    return *Out;
  }

  // Materialize from overlapping registers
  llvm::Value *V =
      materializeFromOverlapping(State, MBB, Key, Builder, *OutRegType);
  State[Key][OutRegType] = V;
  return *V;
}

/// Build the i32 MODE register value that mirrors the kernel-entry state
/// implied by the function's FP attributes (lifted from the kernel
/// descriptor by \c CodeDiscoveryPass). Fields whose attribute is missing
/// fall back to \c SIModeRegisterDefaults so the subtarget-specific
/// defaults stay authoritative. Target-divergent bits (IEEE, DX10_CLAMP)
/// are guarded with subtarget predicates.
static llvm::Value *buildInitialModeValue(const llvm::Function &F,
                                          const llvm::GCNSubtarget &ST,
                                          llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] Building initial MODE "
                                "register value\n");
  llvm::SIModeRegisterDefaults Defaults(F, ST);

  uint32_t Mode = 0;

  /// FP_ROUND (bits 0..3): the backend emits round-to-nearest at kernel
  /// entry on every supported target and there is no function attribute
  /// that overrides this, so both halves stay zero.

  /// FP_DENORM_F32 (bits 4..5).
  uint32_t Denorm32 =
      F.hasFnAttribute("denormal-fp-math-f32")
          ? decodeDenormAttr(
                F.getFnAttribute("denormal-fp-math-f32").getValueAsString())
          : Defaults.fpDenormModeSPValue();
  Mode |= (Denorm32 & 0x3u) << 4;

  /// FP_DENORM_F64/F16 (bits 6..7).
  uint32_t Denorm1664 =
      F.hasFnAttribute("denormal-fp-math")
          ? decodeDenormAttr(
                F.getFnAttribute("denormal-fp-math").getValueAsString())
          : Defaults.fpDenormModeDPValue();
  Mode |= (Denorm1664 & 0x3u) << 6;

  /// DX10_CLAMP (bit 8) — pre-GFX12 only. On GFX12 the bit moved out of
  /// WAVE_MODE; we leave it cleared here.
  if (!llvm::AMDGPU::isGFX12Plus(ST)) {
    bool DX10Clamp =
        F.hasFnAttribute("amdgpu-dx10-clamp")
            ? F.getFnAttribute("amdgpu-dx10-clamp").getValueAsString() == "true"
            : Defaults.DX10Clamp;
    if (DX10Clamp)
      Mode |= llvm::AMDGPU::Hwreg::DX10_CLAMP_MASK;
  }

  /// IEEE (bit 9) — pre-GFX12 only. Moved out of WAVE_MODE on GFX12.
  if (!llvm::AMDGPU::isGFX12Plus(ST)) {
    bool IEEE =
        F.hasFnAttribute("amdgpu-ieee")
            ? F.getFnAttribute("amdgpu-ieee").getValueAsString() == "true"
            : Defaults.IEEE;
    if (IEEE)
      Mode |= (1u << 9);
  }

  /// GPR_IDX_EN, VSKIP, CSP — GFX9-and-earlier only. These MODE bits were
  /// removed / repurposed on GFX10+. On the targets that do have them,
  /// they are guaranteed zero on kernel entry; the masked AND-NOT keeps
  /// the invariant tied to the canonical SIDefines names.
  if (!llvm::AMDGPU::isGFX10Plus(ST)) {
    Mode &= ~llvm::AMDGPU::Hwreg::GPR_IDX_EN_MASK;
    Mode &= ~llvm::AMDGPU::Hwreg::VSKIP_MASK;
    Mode &= ~llvm::AMDGPU::Hwreg::CSP_MASK;
  }

  return Builder.getInt32(Mode);
}

void MIRToIRTranslator::initKernelEntryRegs(llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] Initializing kernel entry "
                                "registers for '"
                             << MF.getName() << "'\n");
  const auto &Info = *MF.getInfo<llvm::SIMachineFunctionInfo>();

  using PV = llvm::AMDGPUFunctionArgInfo::PreloadedValue;

  auto seedRegValue = [&](const llvm::MachineBasicBlock &MBB,
                          llvm::MCRegister Reg, llvm::Value *Val) {
    RegValueMap &State = VM[MBB];
    RegFileKey Key = getRegFileKey(Reg);
    State[Key][Val->getType()] = Val;
    RegValueDesc Desc{std::get<0>(Key), std::get<1>(Key), std::get<2>(Key)};
    std::string Name = formatRegValueDescName(Desc, TRI.getName(Reg));
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(Val))
      attachRegValue(*I, Desc, Name);
    else
      addEntryRegMapping(const_cast<llvm::Function &>(MF.getFunction()), Val,
                         Desc, Name);
  };

  /// Seed a single preloaded register with \p Val.
  auto seed = [&](PV Which, llvm::Value *Val) {
    llvm::MCRegister Reg = Info.getPreloadedReg(Which);
    if (!Reg)
      return;
    seedRegValue(MF.front(), Reg, Val);
  };

  /// Create a frozen-poison placeholder for values with no intrinsic.
  auto makePlaceholder = [&](PV Which) -> llvm::Value * {
    llvm::MCRegister Reg = Info.getPreloadedReg(Which);
    if (!Reg)
      return nullptr;
    unsigned BitWidth = getPhysRegisterSize(Reg);
    return Builder.CreateFreeze(
        llvm::PoisonValue::get(Builder.getIntNTy(BitWidth)));
  };

  /// Emit a void-returning intrinsic whose result is a pointer, then
  /// ptrtoint it to match the register's integer type.
  auto ptrIntrinsic = [&](PV Which, llvm::Intrinsic::ID IID) {
    llvm::MCRegister Reg = Info.getPreloadedReg(Which);
    if (!Reg)
      return;
    llvm::Value *Ptr =
        Builder.CreateIntrinsic(Builder.getPtrTy(4), IID, {}, nullptr);
    // Store the pointer form for consumers that address through this register.
    seed(Which, Ptr);
    // Also store the integer form: getPrimitiveSizeInBits() returns 0 for
    // pointer types, which causes breakdownToVecTyFromAvailableValues to
    // produce a zero-element vector if this register is later split.
    unsigned BitWidth = getPhysRegisterSize(Reg);
    seed(Which, Builder.CreatePtrToInt(Ptr, Builder.getIntNTy(BitWidth)));
  };

  /// Emit a scalar-returning intrinsic (i32 or i64).
  auto scalarIntrinsic = [&](PV Which, llvm::Intrinsic::ID IID,
                             llvm::Type *RetTy) {
    const llvm::ArgDescriptor *ArgDesc =
        std::get<0>(Info.getArgInfo().getPreloadedValue(Which));
    if (!ArgDesc)
      return;
    llvm::MCRegister Reg = ArgDesc->getRegister();
    if (!Reg)
      return;
    unsigned Mask = ArgDesc->getMask();
    llvm::Value *Val = Builder.CreateIntrinsic(RetTy, IID, {}, nullptr);
    /// If the input argument has a mask (e.g. in case of packed workitem ID),
    /// construct the value from the mask first before materializing the final
    /// register value; Otherwise, just assign the register name to the
    /// intrinsic value
    if (Mask != ~0u) {
      unsigned NumRZeros = std::countr_zero(Mask);
      unsigned MaskNoRZeros = Mask >> NumRZeros;
      Val = Builder.CreateAnd(Val, Builder.getInt32(MaskNoRZeros));
      if (NumRZeros)
        Val = Builder.CreateShl(Val, Builder.getInt32(NumRZeros));
      /// Packed work-item IDs (\c FeaturePackedTID) place X/Y/Z in disjoint
      /// bitfields of the SAME VGPR. Each dimension is seeded separately.
      /// OR the repositioned field into whatever has already been seeded for
      /// this register.
      RegValueMap &State = VM[MF.front()];
      RegFileKey Key = getRegFileKey(Reg);
      if (auto It = State.find(Key); It != State.end())
        if (auto VIt = It->second.find(Val->getType()); VIt != It->second.end())
          Val = Builder.CreateOr(VIt->second, Val);
    }

    seed(Which, Val);
  };

  // ---- User SGPRs (allocated in HSA ABI order) ----

  // PrivateSegmentBuffer: no intrinsic — use placeholder.
  if (llvm::Value *V = makePlaceholder(PV::PRIVATE_SEGMENT_BUFFER))
    seed(PV::PRIVATE_SEGMENT_BUFFER, V);

  // DispatchPtr → llvm.amdgcn.dispatch.ptr() : ptr addrspace(4)
  ptrIntrinsic(PV::DISPATCH_PTR, llvm::Intrinsic::amdgcn_dispatch_ptr);

  // QueuePtr → llvm.amdgcn.queue.ptr() : ptr addrspace(4)
  ptrIntrinsic(PV::QUEUE_PTR, llvm::Intrinsic::amdgcn_queue_ptr);

  // KernargSegmentPtr → llvm.amdgcn.kernarg.segment.ptr() : ptr addrspace(4)
  ptrIntrinsic(PV::KERNARG_SEGMENT_PTR,
               llvm::Intrinsic::amdgcn_kernarg_segment_ptr);

  // DispatchID → llvm.amdgcn.dispatch.id() : i64
  scalarIntrinsic(PV::DISPATCH_ID, llvm::Intrinsic::amdgcn_dispatch_id,
                  Builder.getInt64Ty());

  // FlatScratchInit: no intrinsic — use placeholder.
  if (llvm::Value *V = makePlaceholder(PV::FLAT_SCRATCH_INIT))
    seed(PV::FLAT_SCRATCH_INIT, V);

  // PrivateSegmentSize: no intrinsic — use placeholder.
  if (llvm::Value *V = makePlaceholder(PV::PRIVATE_SEGMENT_SIZE))
    seed(PV::PRIVATE_SEGMENT_SIZE, V);

  // ---- Preloaded kernel arguments ----
  //
  // On targets with the kernarg-preload feature the CP loads `Length` dwords
  // of the kernarg segment (starting `Offset` dwords in) into the SGPRs
  // immediately after the user SGPRs, and the kernel reads its arguments from
  // those SGPRs instead of issuing an s_load. The lifted trace reads them
  // directly, so seed each preload SGPR with the equivalent kernarg load.
  //
  // Machine Function reserves the preload registers as user SGPRs, so the
  // first preload SGPR is SGPR0 + (NumUserSGPRs - NumKernargPreloadedSGPRs).
  if (unsigned PreloadLen = Info.getNumKernargPreloadedSGPRs()) {
    unsigned FirstPreloadSGPR =
        llvm::AMDGPU::SGPR0 + Info.getNumUserSGPRs() - PreloadLen;
    unsigned OffsetDwords = MF.getFunction().getFnAttributeAsParsedInteger(
        "amdgpu.kd.kernarg_preload_offset");
    llvm::Type *I32 = Builder.getInt32Ty();
    llvm::Value *KernargPtr = Builder.CreateIntrinsic(
        Builder.getPtrTy(4), llvm::Intrinsic::amdgcn_kernarg_segment_ptr, {},
        nullptr);
    for (unsigned I = 0; I < PreloadLen; ++I) {
      llvm::Value *Slot =
          Builder.CreateConstInBoundsGEP1_64(I32, KernargPtr, OffsetDwords + I);
      auto *Load = Builder.CreateAlignedLoad(I32, Slot, llvm::Align(4));
      // Kernarg memory is constant for the kernel's lifetime.
      Load->setMetadata(llvm::LLVMContext::MD_invariant_load,
                        llvm::MDNode::get(Load->getContext(), {}));
      seedRegValue(MF.front(), llvm::MCRegister(FirstPreloadSGPR + I), Load);
    }
  }

  // ---- System SGPRs ----

  // WorkgroupID X/Y/Z → llvm.amdgcn.workgroup.id.{x,y,z}() : i32
  // On non-architected targets these resolve to the system SGPRs allocated by
  // CodeDiscoveryPass. On architected-SGPRs targets those SGPRs are not
  // allocated (the seeds below are no-ops) and the values come from TTMPs —
  // seeded just after this block.
  scalarIntrinsic(PV::WORKGROUP_ID_X, llvm::Intrinsic::amdgcn_workgroup_id_x,
                  Builder.getInt32Ty());
  scalarIntrinsic(PV::WORKGROUP_ID_Y, llvm::Intrinsic::amdgcn_workgroup_id_y,
                  Builder.getInt32Ty());
  scalarIntrinsic(PV::WORKGROUP_ID_Z, llvm::Intrinsic::amdgcn_workgroup_id_z,
                  Builder.getInt32Ty());

  // On architected-SGPRs targets (GFX12+, or any part built with
  // +architected-sgprs) the workgroup IDs live in fixed TTMP registers that
  // the backend reads directly (see SITargetLowering::getPreloadedValue):
  //   TTMP9        = workgroup id X
  //   TTMP7[15:0]  = workgroup id Y, TTMP7[31:16] = workgroup id Z
  // The lifted trace reads those TTMPs, so seed them. (The Y/Z pack into one
  // register — same OR-merge shape as packed work-item IDs.)
  if (ST.hasArchitectedSGPRs()) {
    llvm::Type *I32 = Builder.getInt32Ty();
    if (Info.hasWorkGroupIDX())
      seedRegValue(
          MF.front(), llvm::AMDGPU::TTMP9,
          Builder.CreateIntrinsic(I32, llvm::Intrinsic::amdgcn_workgroup_id_x,
                                  {}, nullptr));
    llvm::Value *Ttmp7 = nullptr;
    const bool HasZ = Info.hasWorkGroupIDZ();
    if (Info.hasWorkGroupIDY()) {
      llvm::Value *Y = Builder.CreateIntrinsic(
          I32, llvm::Intrinsic::amdgcn_workgroup_id_y, {}, nullptr);
      // With Z present, Y occupies only TTMP7[15:0]; otherwise it spans the
      // whole register (the HW guarantees the high half is zero).
      Ttmp7 = HasZ ? Builder.CreateAnd(Y, Builder.getInt32(0xFFFF)) : Y;
    }
    if (HasZ) {
      llvm::Value *Z = Builder.CreateIntrinsic(
          I32, llvm::Intrinsic::amdgcn_workgroup_id_z, {}, nullptr);
      llvm::Value *ZHi = Builder.CreateShl(Z, Builder.getInt32(16));
      Ttmp7 = Ttmp7 ? Builder.CreateOr(Ttmp7, ZHi) : ZHi;
    }
    if (Ttmp7)
      seedRegValue(MF.front(), llvm::AMDGPU::TTMP7, Ttmp7);
  }

  // WorkGroupInfo has no PreloadedValue enum entry, so seed it directly from
  // ArgInfo (CodeDiscoveryPass allocates it when the KD enables it). No
  // intrinsic — a frozen-poison placeholder, but carrying register provenance.
  if (Info.getArgInfo().WorkGroupInfo.isRegister()) {
    llvm::MCRegister WGInfo = Info.getArgInfo().WorkGroupInfo.getRegister();
    seedRegValue(MF.front(), WGInfo,
                 Builder.CreateFreeze(llvm::PoisonValue::get(
                     Builder.getIntNTy(getPhysRegisterSize(WGInfo)))));
  }

  // PrivateSegmentWaveByteOffset: no intrinsic — use placeholder.
  if (llvm::Value *V = makePlaceholder(PV::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET))
    seed(PV::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, V);

  // ---- VGPRs (work-item IDs) ----

  // WorkitemID X/Y/Z → llvm.amdgcn.workitem.id.{x,y,z}() : i32
  scalarIntrinsic(PV::WORKITEM_ID_X, llvm::Intrinsic::amdgcn_workitem_id_x,
                  Builder.getInt32Ty());
  scalarIntrinsic(PV::WORKITEM_ID_Y, llvm::Intrinsic::amdgcn_workitem_id_y,
                  Builder.getInt32Ty());
  scalarIntrinsic(PV::WORKITEM_ID_Z, llvm::Intrinsic::amdgcn_workitem_id_z,
                  Builder.getInt32Ty());

  // ---- Writable specials ----

  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();

  /// EXEC is all-ones on kernel entry (every lane active). Width matches
  /// the wavefront size — use ~0ULL so wave64 sets all 64 bits, not just
  /// the low 32.
  llvm::MCRegister Exec = TRI.getExec();
  unsigned ExecWidth = TRI.getRegSizeInBits(Exec, MF.getRegInfo());
  llvm::Value *ExecInit = Builder.getInt(llvm::APInt::getAllOnes(ExecWidth));
  seedRegValue(MF.front(), Exec, ExecInit);

  /// SCC is zero on kernel entry.
  seedRegValue(MF.front(), llvm::AMDGPU::SRC_SCC, Builder.getInt32(false));

  /// MODE: constant assembled from the kernel-descriptor-derived attrs.
  llvm::Value *ModeInit = buildInitialModeValue(MF.getFunction(), ST, Builder);
  seedRegValue(MF.front(), llvm::AMDGPU::MODE, ModeInit);

  /// VCC is zero on kernel entry. \c TRI.getVCC() returns VCC_LO on
  /// wave32 and the full VCC pair on wave64.
  if (llvm::MCRegister VccReg = TRI.getVCC()) {
    unsigned VccWidth = TRI.getRegSizeInBits(VccReg, MF.getRegInfo());
    llvm::Value *VccInit = Builder.getIntN(VccWidth, 0);
    seedRegValue(MF.front(), VccReg, VccInit);
  }
}

MIRToIRTranslator::MIRToIRTranslator(llvm::MachineFunction &MF,
                                     llvm::Error &Err)
    : MF(MF), TRI(*MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo()),
      TII(*MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo()),
      ST(MF.getSubtarget<llvm::GCNSubtarget>()) {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] Creating translator for '" << MF.getName()
             << "' with " << MF.size() << " MBBs\n");
  llvm::ErrorAsOutParameter EAO(Err);
  for (const llvm::MachineBasicBlock &MBB : MF)
    VM.try_emplace(std::ref(MBB));

  Err = initRegFileLayouts();
  if (Err)
    return;

  Err =
      MIInlineAsmEmitter::get(const_cast<llvm::TargetMachine &>(MF.getTarget()))
          .moveInto(InlineAsmEmitter);
  if (Err) {
    return;
  }
}

llvm::MCRegister MIRToIRTranslator::getPhysReg(llvm::MCRegister Reg) const {
  switch (Reg) {
  case llvm::AMDGPU::SCC:
    return llvm::AMDGPU::SRC_SCC;
  default:
    return llvm::AMDGPU::getMCReg(Reg, ST);
  }
}

unsigned MIRToIRTranslator::getPhysRegisterSize(llvm::MCRegister Reg) const {
  if (Reg == llvm::AMDGPU::MODE)
    return 32;
  else if (Reg == llvm::AMDGPU::SCC)
    return 32; /// Return SRC_SCC's size instead
  const llvm::TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
  if (RC) {
    return TRI.getRegSizeInBits(*RC);
  }
  llvm_unreachable(
      llvm::formatv("Register {0} does not have any register class and its "
                    "size must be explicitly provided",
                    TRI.getName(Reg))
          .str()
          .c_str());
}

llvm::Error MIRToIRTranslator::initRegFileLayouts() {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] Initializing register file "
                "layouts for '"
             << MF.getName() << "'\n");
  const llvm::Function &F = MF.getFunction();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();

  unsigned NumSGPRs = F.getFnAttributeAsParsedInteger("amdgpu-num-sgpr");
  unsigned NumVGPRs = F.getFnAttributeAsParsedInteger("amdgpu-num-vgpr");

  if (NumSGPRs == 0) {
    return LUTHIER_MAKE_GENERIC_ERROR("amdgpu-num-sgpr must be non-zero.");
  }
  if (NumVGPRs == 0) {
    return LUTHIER_MAKE_GENERIC_ERROR("amdgpu-num-vgpr must be a non-zero.");
  }

  getRegisterFileArgOrder(ST, FunctionCallArgOrder);
  computeRegFileSizes(ST, NumSGPRs, NumVGPRs, RegFileSize, TTMPBaseReg,
                      ExecBaseReg);

  /// Reserve two SGPR slots at the top of the kernel SGPR allocation for each
  /// SGPR-aliased special on pre-GFX10, in the order the GPU allocates them:
  /// VCC, then XNACK_MASK, then FLAT_SCR. We store only
  /// the LO-half SGPR MCRegister; HI-half is \c LO + 1 because the SGPR
  /// enum is contiguous. The kernel is guaranteed to carry at least
  /// enough SGPRs for VCC (the SGPR granule on every supported target is
  /// >= 8), so VCC always fits.
  assert(NumSGPRs >= 2 && "kernel must have at least two SGPRs for VCC");
  unsigned NextSlot = NumSGPRs;
  auto reserveLoPair = [&]() -> llvm::MCRegister {
    if (NextSlot < 2)
      return llvm::MCRegister{};
    NextSlot -= 2;
    return llvm::MCRegister(llvm::AMDGPU::SGPR0 + NextSlot);
  };
  /// VCC is reserved first; route it through the same guarded helper as the
  /// other specials so a (asserted-against) NumSGPRs < 2 can't underflow the
  /// unsigned slot counter in release builds.
  VccLoSgpr = reserveLoPair();
  if (llvm::AMDGPU::isNotGFX10Plus(ST)) {
    if (ST.getTargetID().isXnackSupported())
      XnackMaskLoSgpr = reserveLoPair();
    if (ST.hasFlatScratchInsts())
      FlatScrLoSgpr = reserveLoPair();
  }
  return llvm::Error::success();
}

MIRToIRTranslator::RegFileKey
MIRToIRTranslator::getRegFileKey(llvm::MCRegister Reg) const {
  LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] getRegFileKey for reg "
                             << TRI.getName(Reg) << "\n");
  llvm::MCRegister MCReg = getPhysReg(Reg);
  if (MCReg == llvm::AMDGPU::MODE)
    return std::make_tuple(Reg, 0, 2);

  unsigned Enc = TRI.getEncodingValue(MCReg);
  unsigned HwIdx = Enc & llvm::AMDGPU::HWEncoding::REG_IDX_MASK;
  unsigned IsHi16 = (Enc & llvm::AMDGPU::HWEncoding::IS_HI16) ? 1u : 0u;

  llvm::MCRegister BaseReg;
  if (Enc & llvm::AMDGPU::HWEncoding::IS_AGPR)
    BaseReg = llvm::AMDGPU::AGPR0;
  else if (Enc & llvm::AMDGPU::HWEncoding::IS_VGPR)
    BaseReg = llvm::AMDGPU::VGPR0;
  else
    BaseReg = llvm::AMDGPU::SGPR0;

  if (BaseReg == llvm::AMDGPU::SGPR0) {
    /// Pre-GFX10 alias translation: VCC / XNACK_MASK / FLAT_SCR are
    /// reserved at the top of the kernel SGPR allocation, so route them to
    /// the logical SGPR they alias before the encoding lookup. The cache
    /// then naturally shares the slot between VCC-named and SGPR-named
    /// access to the same physical pair.
    if (llvm::AMDGPU::isNotGFX10Plus(ST)) {
      auto rewriteAlias = [&](llvm::MCRegister AliasReg,
                              llvm::MCRegister LogicalBase) -> bool {
        if (!TRI.regsOverlap(Reg, AliasReg))
          return false;
        HwIdx = TRI.getEncodingValue(LogicalBase) &
                llvm::AMDGPU::HWEncoding::REG_IDX_MASK;
        return true;
      };
      for (auto [AliasReg, LogicalBase] :
           std::initializer_list<std::pair<llvm::MCRegister, llvm::MCRegister>>{
               {llvm::AMDGPU::VCC, VccLoSgpr},
               {llvm::AMDGPU::XNACK_MASK, XnackMaskLoSgpr},
               {llvm::AMDGPU::FLAT_SCR, FlatScrLoSgpr}}) {
        if (rewriteAlias(AliasReg, LogicalBase))
          break;
      }
    }
    /// Take care of special SGPR registers
    if (HwIdx >= RegFileSize.at(llvm::AMDGPU::SGPR0) / 2) {
      /// We sort the checks based on the frequency of the register files
      /// accessed

      unsigned VCCZBaseIdx =
          TRI.getEncodingValue(getPhysReg(llvm::AMDGPU::SRC_VCCZ)) &
          llvm::AMDGPU::HWEncoding::REG_IDX_MASK;
      unsigned ExecBaseIdx = TRI.getEncodingValue(getPhysReg(ExecBaseReg)) &
                             llvm::AMDGPU::HWEncoding::REG_IDX_MASK;
      unsigned TTmpBaseIdx = TRI.getEncodingValue(getPhysReg(TTMPBaseReg)) &
                             llvm::AMDGPU::HWEncoding::REG_IDX_MASK;
      unsigned SharedBaseIdx =
          TRI.getEncodingValue(getPhysReg(llvm::AMDGPU::SRC_SHARED_BASE)) &
          llvm::AMDGPU::HWEncoding::REG_IDX_MASK;

      if (HwIdx >= VCCZBaseIdx && HwIdx < VCCZBaseIdx + 6) {
        BaseReg = llvm::AMDGPU::SRC_VCCZ;
      } else if (HwIdx >= ExecBaseIdx && HwIdx < ExecBaseIdx + 4) {
        BaseReg = ExecBaseReg;
      } else if (HwIdx >= TTmpBaseIdx && HwIdx < TTmpBaseIdx + 16) {
        BaseReg = TTMPBaseReg;
      } else if (llvm::AMDGPU::isGFX9Plus(ST) && HwIdx >= SharedBaseIdx &&
                 HwIdx < SharedBaseIdx + RegFileSize.at(getPhysReg(
                                             llvm::AMDGPU::SRC_SHARED_BASE)) /
                                             2) {
        BaseReg = getPhysReg(llvm::AMDGPU::SRC_SHARED_BASE);
      } else
        llvm_unreachable("SGPR is not contained in any register file");
    }
  }

  unsigned BaseHwIdx =
      TRI.getEncodingValue(BaseReg) & llvm::AMDGPU::HWEncoding::REG_IDX_MASK;

  unsigned Offset = (HwIdx - BaseHwIdx) * 2 + IsHi16;

  unsigned RegSizeBits = getPhysRegisterSize(Reg);

  auto Key = std::make_tuple(BaseReg, Offset, RegSizeBits / RegGranule);
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] -> Key: base=" << TRI.getName(BaseReg)
             << " offset=" << Offset << " halves=" << std::get<2>(Key) << "\n");
  return Key;
}

std::string MIRToIRTranslator::getRegfileValueName(llvm::MCRegister BaseReg) {
  switch (BaseReg) {
  case llvm::AMDGPU::SGPR0:
    return "sgpr_file";
  case llvm::AMDGPU::VGPR0:
    return "vgpr_file";
  case llvm::AMDGPU::AGPR0:
    return "agpr_file";
  default:
    if (BaseReg == TTMPBaseReg)
      return "ttmp_file";
    else if (BaseReg == ExecBaseReg)
      return "exec_file";
    else if (BaseReg == llvm::AMDGPU::SRC_SHARED_BASE)
      return "apreture_file";
    else if (BaseReg == llvm::AMDGPU::SRC_VCCZ)
      return "vccz_file";
    else
      assert(BaseReg == llvm::AMDGPU::MODE && "Invalid register file base");
    return "hw_reg_file";
  }
}

llvm::Value *MIRToIRTranslator::getRegisterFile(
    const llvm::MachineBasicBlock &MBB, llvm::MCRegister Reg,
    llvm::IRBuilderBase &Builder, llvm::Type *LaneTy) {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] getRegisterFile: MBB " << MBB.getNumber()
             << " reg=" << TRI.getName(Reg) << "\n");
  /// Always materialize the FULL register file (offset=0..total) under a
  /// single canonical key, then return a shufflevector of just the
  /// requested slice. Earlier versions materialized each slice under its
  /// own key, which polluted `RegValueMap` with mismatched-width entries
  /// (e.g. `<7 x i32>` for a slice from `v0` vs `<5 x i32>` for a slice
  /// from `v2`) and tripped the width-divisibility check in
  /// `breakdownToVecTyFromAvailableValues` when one query needed to
  /// rebuild from another's cached value. With a single full-file key,
  /// every consumer shares the same cache entry; the slice returned to
  /// the caller is a cheap `shufflevector` lane-pick the optimizer
  /// folds away when the index is constant.
  auto Key = getRegFileKey(Reg);
  llvm::MCRegister RegFileBaseReg = std::get<0>(Key);
  unsigned StartHalves = std::get<1>(Key);
  unsigned TotalHalves = RegFileSize.at(RegFileBaseReg);
  assert(TotalHalves != 0 &&
         "register file is not modeled for the current target");
  assert(StartHalves <= TotalHalves && "register offset exceeds file size");

  if (!LaneTy)
    LaneTy = Builder.getInt32Ty();
  assert(LaneTy->isIntegerTy() && !LaneTy->isVectorTy() &&
         "Lane type is not a scalar integer type");

  unsigned LaneSize = LaneTy->getPrimitiveSizeInBits();
  assert(LaneSize % RegGranule == 0 && "Lane size is not divisible by 16");

  unsigned FullNumLanes = TotalHalves * RegGranule / LaneSize;
  auto *FullVecTy = llvm::FixedVectorType::get(LaneTy, FullNumLanes);
  RegFileKey FullKey = std::make_tuple(RegFileBaseReg, 0u, TotalHalves);
  llvm::Value *FullVec = &getOperandAsValue(MBB, FullKey, Builder, FullVecTy);

  unsigned StartLane = StartHalves * RegGranule / LaneSize;
  unsigned SliceNumLanes = FullNumLanes - StartLane;
  if (StartLane == 0)
    return FullVec;

  llvm::SmallVector<int, 32> Mask;
  Mask.reserve(SliceNumLanes);
  for (unsigned I = 0; I < SliceNumLanes; ++I)
    Mask.push_back(static_cast<int>(StartLane + I));
  return Builder.CreateShuffleVector(FullVec, Mask);
}

llvm::Value *MIRToIRTranslator::getRegisterFile(const llvm::MachineInstr &MI,
                                                llvm::MCRegister Register,
                                                llvm::Type *LaneTy) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");
  auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
  assert(BB && "MBB has no IR basic block");
  llvm::Instruction *TermInst = BB->getTerminatorOrNull();

  llvm::MCRegister BaseReg = std::get<0>(getRegFileKey(Register));

  std::string ValueName = getRegfileValueName(BaseReg);
  llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
      Builder(BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
              llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                annotateUniformIfNeeded(I, TRI, Register);
                LLVM_DEBUG(
                    luthier::dbgs()
                    << "[MIRToIRTranslator] Inserting read reg instruction "
                    << *I << "\n");
              }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  return getRegisterFile(*MBB, Register, Builder, LaneTy);
}

void MIRToIRTranslator::setRegisterFile(const llvm::MachineInstr &MI,
                                        llvm::MCRegister Reg,
                                        llvm::Value *NewVec) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");
  auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
  assert(BB && "MBB has no IR basic block");
  llvm::Instruction *TermInst = BB->getTerminatorOrNull();

  llvm::MCRegister BaseReg = std::get<0>(getRegFileKey(Reg));

  std::string ValueName = getRegfileValueName(BaseReg);
  llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
      Builder(BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
              llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                annotateUniformIfNeeded(I, TRI, Reg);
                LLVM_DEBUG(
                    luthier::dbgs()
                    << "[MIRToIRTranslator] Inserting read reg instruction "
                    << *I << "\n");
              }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  setRegisterFile(*MBB, Reg, Builder, NewVec);
}

llvm::Value *MIRToIRTranslator::getRegisterFile(const llvm::MachineInstr &MI,
                                                llvm::AMDGPU::OpName OpName,
                                                llvm::Type *LaneTy) {
  const llvm::MachineOperand *Op = TII.getNamedOperand(MI, OpName);
  assert(Op && Op->isReg() &&
         "GetRegisterFile target operand is not a register operand");
  return getRegisterFile(MI, Op->getReg(), LaneTy);
}

void MIRToIRTranslator::setRegisterFile(const llvm::MachineInstr &MI,
                                        llvm::AMDGPU::OpName OpName,
                                        llvm::Value *NewVec) {
  const llvm::MachineOperand *Op = TII.getNamedOperand(MI, OpName);
  assert(Op && Op->isReg() &&
         "SetRegisterFile target operand is not a register operand");
  setRegisterFile(MI, Op->getReg(), NewVec);
}

void MIRToIRTranslator::setRegisterFile(const llvm::MachineBasicBlock &MBB,
                                        llvm::MCRegister Reg,
                                        llvm::IRBuilderBase &Builder,
                                        llvm::Value *Val) {
  /// Symmetric with `getRegisterFile`: write the FULL file under the
  /// canonical (BaseReg, 0, TotalHalves) key. `Val` is a vector covering
  /// the slice `[Reg..end]`; splice its lanes back into the full file
  /// via insertelement at the appropriate absolute lane indices.
  auto Key = getRegFileKey(Reg);
  llvm::MCRegister RegFileBaseReg = std::get<0>(Key);
  unsigned StartHalves = std::get<1>(Key);
  unsigned TotalHalves = RegFileSize.at(RegFileBaseReg);
  assert(StartHalves <= TotalHalves && "register offset exceeds file size");

  auto *SliceVecTy = llvm::cast<llvm::FixedVectorType>(Val->getType());
  llvm::Type *LaneTy = SliceVecTy->getElementType();
  unsigned LaneSize = LaneTy->getPrimitiveSizeInBits();
  unsigned FullNumLanes = TotalHalves * RegGranule / LaneSize;
  auto *FullVecTy = llvm::FixedVectorType::get(LaneTy, FullNumLanes);
  RegFileKey FullKey = std::make_tuple(RegFileBaseReg, 0u, TotalHalves);

  llvm::Value *NewFull;
  unsigned StartLane = StartHalves * RegGranule / LaneSize;
  if (StartLane == 0 && SliceVecTy->getNumElements() == FullNumLanes) {
    /// Slice spans the whole file — caller already produced the full
    /// vector, no splicing needed.
    NewFull = Val;
  } else {
    /// Read the current full file, then insertelement each slice lane
    /// at its absolute position. Adjacent insertelements collapse
    /// cleanly under InstCombine when many lanes are unchanged.
    llvm::Value *OldFull = &getOperandAsValue(MBB, FullKey, Builder, FullVecTy);
    NewFull = OldFull;
    unsigned SliceLanes = SliceVecTy->getNumElements();
    for (unsigned I = 0; I < SliceLanes; ++I) {
      llvm::Value *Lane = Builder.CreateExtractElement(Val, I);
      NewFull = Builder.CreateInsertElement(NewFull, Lane, StartLane + I);
    }
  }

  setRegOperandValue(MBB, FullKey, Builder, NewFull);
}

llvm::FunctionType *MIRToIRTranslator::getStandardDeviceFunctionType() const {
  LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] Getting standard device "
                                "function type for '"
                             << MF.getName() << "'\n");
  const llvm::Function &F = MF.getFunction();
  if (F.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
    return F.getFunctionType();
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  /// The translator only exists once \c initRegFileLayouts has validated
  /// non-zero \c amdgpu-num-sgpr / \c amdgpu-num-vgpr, so the static call
  /// here is infallible for any well-formed instance.
  return llvm::cantFail(computeStandardDeviceFunctionType(
      F.getContext(), ST, F.getFnAttributeAsParsedInteger("amdgpu-num-sgpr"),
      F.getFnAttributeAsParsedInteger("amdgpu-num-vgpr")));
}

llvm::Expected<llvm::FunctionType *>
MIRToIRTranslator::computeStandardDeviceFunctionType(
    llvm::LLVMContext &Ctx, const llvm::GCNSubtarget &ST, unsigned NumSGPRs,
    unsigned NumVGPRs) {
  if (NumSGPRs == 0)
    return LUTHIER_MAKE_GENERIC_ERROR("amdgpu-num-sgpr must be non-zero.");
  if (NumVGPRs == 0)
    return LUTHIER_MAKE_GENERIC_ERROR("amdgpu-num-vgpr must be non-zero.");

  llvm::SmallVector<llvm::MCRegister> ArgOrder;
  getRegisterFileArgOrder(ST, ArgOrder);

  llvm::SmallDenseMap<llvm::MCRegister, unsigned> RegFileSize;
  llvm::MCRegister TTMPBaseReg;
  unsigned ExecBaseReg = 0;
  computeRegFileSizes(ST, NumSGPRs, NumVGPRs, RegFileSize, TTMPBaseReg,
                      ExecBaseReg);

  unsigned TotalNumArgs = 0;
  for (llvm::MCRegister RegFileBase : ArgOrder)
    TotalNumArgs += RegFileSize.at(RegFileBase) / 2;

  auto *I32 = llvm::Type::getInt32Ty(Ctx);
  llvm::SmallVector<llvm::Type *> Fields(TotalNumArgs, I32);
  llvm::FunctionType *FuncTy = llvm::FunctionType::get(
      llvm::Type::getVoidTy(Ctx), Fields, /*isVarArg=*/false);

  LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] device function type: "
                             << *FuncTy << "\n");
  return FuncTy;
}

void MIRToIRTranslator::initDeviceFunctionEntryRegs(
    llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] Initializing device function "
                "entry registers for '"
             << MF.getName() << "' with " << MF.getFunction().arg_size()
             << " arguments\n");
  llvm::Function &F = const_cast<llvm::Function &>(MF.getFunction());

  const llvm::MachineBasicBlock &EntryMBB = MF.front();
  RegValueMap &State = VM[EntryMBB];

  unsigned CurrentArgPos = 0;
  llvm::Type *I32 = Builder.getInt32Ty();
  for (llvm::MCRegister RegFileBase : FunctionCallArgOrder) {
    /// store register file entries
    unsigned NumLanes32 = RegFileSize.at(RegFileBase) / 2u;
    llvm::StringRef BaseName = TRI.getName(RegFileBase);
    for (unsigned I = 0; I < NumLanes32; ++I) {
      // Each 32-bit GPR spans 2 halves (RegGranule = 16 bits), so SGPR_N lives
      // at offset 2*N in the half-indexed register file.
      llvm::Argument *Arg = F.getArg(CurrentArgPos + I);
      State[std::make_tuple(RegFileBase, I * 2, 2)][I32] = Arg;
      RegValueDesc Desc{RegFileBase, I * 2u, 2u};
      addEntryRegMapping(F, Arg, Desc, formatRegValueDescName(Desc, BaseName));
    }
    CurrentArgPos += NumLanes32;
  }
}

void MIRToIRTranslator::emitDirectTailCall(const llvm::MachineInstr &MI,
                                           llvm::IRBuilderBase &Builder,
                                           llvm::Value *InstAddr,
                                           llvm::Value *Target) {
  llvm::Value *FinalTarget{nullptr};
  if (auto *TargetConst = dyn_cast<llvm::ConstantInt>(Target);
      TargetConst && MI.getOpcode() == llvm::AMDGPU::S_CALL_B64) {
    FinalTarget = Builder.CreateAdd(
        InstAddr, Builder.getInt64(4 * TargetConst->getSExtValue()));
  } else if (llvm::isa<llvm::Function>(Target)) {
    FinalTarget = Target;
  } else {
    llvm_unreachable("Unsupported direct call target operand");
  }
  assert(FinalTarget && "Target does not have a called target");

  emitIndirectTailCall(MI, Builder, FinalTarget);
}

void MIRToIRTranslator::emitIndirectTailCall(const llvm::MachineInstr &MI,
                                             llvm::IRBuilderBase &Builder,
                                             llvm::Value *Target) {
  if (!Target) {
    // CodeDiscoveryPass couldn't resolve the call target (e.g. S_CALL_B64
    // with an unresolved address). Skip emission rather than crash — the
    // MIR still records the call site for downstream analysis.
    LLVM_DEBUG(luthier::dbgs()
               << "[MIRToIRTranslator] Skipping call emission in MBB "
               << MI.getParent()->getNumber() << ": target is nullptr\n");
    return;
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] Emitting indirect tail call "
                "in MBB "
             << MI.getParent()->getNumber() << " target=" << *Target << "\n");
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");

  llvm::Value *FuncPtr =
      Builder.CreateBitOrPointerCast(Target, Builder.getPtrTy());

  llvm::FunctionType *FTy = getStandardDeviceFunctionType();
  std::vector<llvm::Value *> CallArgs;
  CallArgs.reserve(FTy->getNumParams());

  for (llvm::MCRegister RegFileBase : FunctionCallArgOrder) {
    unsigned NumLanes32 = RegFileSize.at(RegFileBase) / 2;
    for (unsigned PI = 0; PI < NumLanes32; ++PI) {
      // Each 32-bit GPR spans 2 halves, so SGPR_N lives at offset 2*N.
      CallArgs.push_back(&getOperandAsValue(
          *MBB, std::make_tuple(RegFileBase, PI * 2, 2), Builder));
    }
  }

  llvm::CallInst *Call = Builder.CreateCall(FTy, FuncPtr, CallArgs);
  Call->setTailCallKind(llvm::CallInst::TCK_Tail);

  /// Create an unreachable instruction to end the control flow graph
  Builder.CreateUnreachable();
}

llvm::Value &MIRToIRTranslator::getOperandAsValue(const llvm::MachineInstr &MI,
                                                  llvm::AMDGPU::OpName OpName,
                                                  llvm::Type *OutType) {
  return getOperandAsValue(*TII.getNamedOperand(MI, OpName), OutType);
}

llvm::Value &MIRToIRTranslator::getOperandAsValue(const llvm::MachineInstr &MI,
                                                  llvm::MCRegister Reg,
                                                  llvm::Type *RegType) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI does not have a machine basic block");
  if (shouldEmitGPRIndexAccess(MI, Reg))
    return emitIndexedVGPRSrc(MI, Reg, RegType);
  return getOperandAsValue(*MBB, Reg, RegType);
}

llvm::Value &
MIRToIRTranslator::getOperandAsValue(const llvm::MachineOperand &Op,
                                     llvm::Type *OutType) {
  switch (Op.getType()) {
  case llvm::MachineOperand::MO_Register: {
    const llvm::MachineInstr *MI = Op.getParent();
    assert(MI && "Operand does not have a machine instruction");
    return getOperandAsValue(*MI, Op.getReg(), OutType);
  }
  case llvm::MachineOperand::MO_Immediate: {
    const llvm::MachineInstr *MI = Op.getParent();
    assert(MI && "Operand does not have a machine instruction");
    llvm::LLVMContext &Ctx = MF.getFunction().getContext();
    if (!OutType) {
      // Default to the natural width of this operand slot. AMDGPU semantics
      // routines (e.g. S_CMP_LT_U32) call \c getOperandAsValue without an
      // explicit type for both the register and the immediate operands of a
      // comparison; if the immediate defaults to i64 while the register
      // defaults to the register's size (i32 for an SGPR), the resulting
      // ICmp/binop sees mismatched operand types.
      unsigned OpIdx = MI->getOperandNo(&Op);
      const llvm::MCInstrDesc &Desc = MI->getDesc();
      // Default to a 32-bit literal — the canonical VALU literal slot. This
      // also covers an immediate that lands at an implicit/appended operand
      // index (beyond the described operands), where \c operands()[OpIdx]
      // would read out of bounds.
      unsigned SizeInBytes = 4;
      if (OpIdx < Desc.getNumOperands()) {
        const llvm::MCOperandInfo &OpInfo = Desc.operands()[OpIdx];
        if (OpInfo.RegClass == -1) {
          // Pure-immediate slot. \c SIInstrInfo::getOpSize asserts that
          // \c OperandType is the generic \c MCOI::OPERAND_IMMEDIATE,
          // which excludes AMDGPU's own immediate operand kinds (K
          // immediates, encoded modifiers, split-barrier int32, …).
          // Decode the size directly from the operand kind so callers
          // that don't pass an explicit \c OutType still get a properly
          // sized constant.
          switch (OpInfo.OperandType) {
          case llvm::AMDGPU::OPERAND_KIMM64:
            SizeInBytes = 8;
            break;
          case llvm::AMDGPU::OPERAND_KIMM16:
            SizeInBytes = 2;
            break;
          case llvm::AMDGPU::OPERAND_KIMM32:
          case llvm::AMDGPU::OPERAND_INLINE_SPLIT_BARRIER_INT32:
          case llvm::AMDGPU::OPERAND_INPUT_MODS:
          case llvm::MCOI::OPERAND_IMMEDIATE:
            SizeInBytes = 4;
            break;
          default:
            // A wrong size here would surface immediately as a type mismatch
            // at the use site rather than corrupt silently.
            SizeInBytes = 4;
            break;
          }
        } else {
          // Register-or-immediate slot: fall back to the register
          // class's width.
          SizeInBytes = TII.getOpSize(MI->getOpcode(), OpIdx);
        }
      }
      OutType = llvm::IntegerType::get(Ctx, SizeInBytes * 8);
    }
    if (OutType->isIntegerTy())
      return *llvm::ConstantInt::getSigned(OutType, Op.getImm(),
                                           /*ImplicitTrunc=*/true);
    // Non-integer destination type (e.g. the <2 x float>/<2 x i16> packed
    // operand types requested by VOP3P semantics such as V_PK_MUL_F32):
    // materialize the raw immediate bits as an integer of the same width and
    // bitcast to the requested type. NB: inline-constant broadcast to packed
    // lanes is not modeled here; the literal bit pattern is taken as-is, which
    // matches how the integer path interprets Op.getImm().
    unsigned Bits = OutType->getPrimitiveSizeInBits();
    auto *RawInt = llvm::ConstantInt::getSigned(
        llvm::IntegerType::get(Ctx, Bits), Op.getImm(), /*ImplicitTrunc=*/true);
    return *llvm::ConstantExpr::getBitCast(RawInt, OutType);
  }
  case llvm::MachineOperand::MO_GlobalAddress:
    return *const_cast<llvm::GlobalValue *>(Op.getGlobal());
  default:
    llvm_unreachable("Unhandled operand type");
  }
}

llvm::BasicBlock &
MIRToIRTranslator::getOperandAsBasicBlock(const llvm::MachineInstr &MI,
                                          llvm::AMDGPU::OpName OpName) {
  return getOperandAsBasicBlock(*TII.getNamedOperand(MI, OpName));
}

llvm::BasicBlock &
MIRToIRTranslator::getOperandAsBasicBlock(const llvm::MachineOperand &Op) {
  auto *BB = const_cast<llvm::BasicBlock *>(Op.getMBB()->getBasicBlock());
  assert(BB && "MBB operand has no IR BasicBlock");
  return *BB;
}

llvm::Function *
MIRToIRTranslator::getOperandAsFunction(const llvm::MachineInstr &MI,
                                        llvm::AMDGPU::OpName OpName) {
  const llvm::MachineOperand *Op = TII.getNamedOperand(MI, OpName);
  if (!Op || !Op->isGlobal())
    return nullptr;
  return const_cast<llvm::Function *>(
      llvm::dyn_cast<llvm::Function>(Op->getGlobal()));
}

void MIRToIRTranslator::setRegOperandValue(const llvm::MachineInstr &MI,
                                           llvm::MCRegister Reg,
                                           llvm::Value *Val) {
  assert(Val && "Val is nullptr");
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");

  if (shouldEmitGPRIndexAccess(MI, Reg)) {
    emitIndexedVGPRDst(MI, Reg, Val);
    return;
  }

  auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
  assert(BB && "MBB has no IR basic block");

  llvm::Instruction *TermInst = BB->getTerminatorOrNull();
  std::string ValueName = getRegValueName(Reg);
  llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
      Builder(BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
              llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                annotateUniformIfNeeded(I, TRI, Reg);
                LLVM_DEBUG(
                    luthier::dbgs()
                    << "[MIRToIRTranslator] Inserting reg write instruction "
                    << *I << "\n");
              }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  unsigned RegSize = getPhysRegisterSize(Reg);
  unsigned ValBits = Val->getType()->getPrimitiveSizeInBits();
  // Zero-extend integer values narrower than the destination register.
  // This is the AMDGPU wave32 ballot → VCC case: ballot returns
  // `iWavefrontSize` (i32 on wave32) but VCC is always 64-bit. The
  // hardware leaves the inactive upper half at zero, which matches a
  // zext semantically. We only allow widening — a value larger than
  // the register would be a real bug in the .td semantics.
  if (ValBits < RegSize && Val->getType()->isIntegerTy())
    Val = Builder.CreateZExt(Val, Builder.getIntNTy(RegSize));
  assert(Val->getType()->getPrimitiveSizeInBits() == RegSize &&
         "Value type's size is not the same as the type of the register");
  (void)RegSize;

  LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                 "[MIRToIRTranslator] Setting register {0} to value {3} for "
                 "MBB {1} (type: {2})\n",
                 TRI.getName(Reg), MBB->getNumber(),
                 *Val->getType()->getScalarType(), *Val));

  setRegOperandValue(*MBB, getRegFileKey(Reg), Builder, Val);
}

void MIRToIRTranslator::setRegOperandValue(const llvm::MachineOperand &Op,
                                           llvm::Value *Val) {
  assert(Val && "Val is nullptr");
  assert(Op.isReg() && "Operand is not a register");
  assert(Op.getReg().isPhysical() && "Operand is not a physical register");
  const llvm::MachineInstr *MI = Op.getParent();
  assert(MI && "Machine operand has no parent MI");
  setRegOperandValue(*MI, Op.getReg(), Val);
}

void MIRToIRTranslator::setRegOperandValue(const llvm::MachineBasicBlock &MBB,
                                           const RegFileKey &Key,
                                           llvm::IRBuilderBase &Builder,
                                           llvm::Value *Val) {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] setRegOperandValue: MBB "
             << MBB.getNumber() << " base=" << TRI.getName(std::get<0>(Key))
             << " offset=" << std::get<1>(Key) << " halves=" << std::get<2>(Key)
             << " val=" << *Val << " (type=" << *Val->getType() << ")\n");
  RegValueMap &State = VM[MBB];
  llvm::MCRegister BaseReg = std::get<0>(Key);
  unsigned Offset = std::get<1>(Key);
  unsigned Size = std::get<2>(Key);

  /// Bounds check: silently drop writes that target an unallocated plain
  /// GPR slot. Specials (TTMP/M0/EXEC/NULL/VCC-on-GFX10+) bypass the
  /// check and are always written through.
  unsigned Allocated = RegFileSize.at(BaseReg);
  if (Offset + Size > Allocated) {
    LLVM_DEBUG(luthier::dbgs()
               << "[MIRToIRTranslator] Dropping out-of-range write to "
               << " (offset=" << Offset << " halves=" << Size
               << " allocated=" << Allocated << ")\n");
    return;
  }

  /// Preserve non-overlapping portions of partially-overwritten
  /// super-registers, then erase fully-covered entries.
  invalidateOverlaps(State, Key, Builder);
  State[Key][Val->getType()] = Val;

  /// Tag the value with the (BaseReg, HalfWordOffset, NumHalves) it
  /// represents so downstream passes can trace register provenance.
  /// Instructions carry per-instruction \c !luthier.reg metadata;
  /// non-Instruction values (function arguments / constants) flow into
  /// the function-level \c !luthier.entry_reg_map slot.
  RegValueDesc Desc{BaseReg, Offset, Size};
  std::string Name = formatRegValueDescName(Desc, TRI.getName(BaseReg));
  if (auto *I = llvm::dyn_cast<llvm::Instruction>(Val)) {
    attachRegValue(*I, Desc, Name);
  } else {
    addEntryRegMapping(const_cast<llvm::Function &>(MF.getFunction()), Val,
                       Desc, Name);
  }
}

void MIRToIRTranslator::setRegOperandValue(const llvm::MachineInstr &MI,
                                           llvm::AMDGPU::OpName OpName,
                                           llvm::Value *Val) {
  setRegOperandValue(*TII.getNamedOperand(MI, OpName), Val);
}

llvm::BasicBlock *MIRToIRTranslator::getNextBB(const llvm::MachineInstr &MI) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI does not have a basic block");
  const llvm::MachineBasicBlock *NextMBB = MBB->getNextNode();
  assert(NextMBB && "MI doesn't have a fall-through block");

  return const_cast<llvm::BasicBlock *>(NextMBB->getBasicBlock());
}

llvm::SyncScope::ID
MIRToIRTranslator::getSyncScope(const llvm::Value *CPolVal) const {
  llvm::LLVMContext &Ctx = MF.getFunction().getContext();
  const auto *CI = llvm::dyn_cast_or_null<llvm::ConstantInt>(CPolVal);
  if (!CI)
    return llvm::SyncScope::System;
  uint64_t CPol = CI->getZExtValue();

  // Pick the encoding by subtarget.
  if (llvm::AMDGPU::isGFX12Plus(ST)) {
    // gfx12: bits[4:3] = scope. 00=CU, 01=SE, 10=DEV, 11=SYS.
    unsigned Scope = (CPol >> 3) & 0x3;
    switch (Scope) {
    case 0:
      return Ctx.getOrInsertSyncScopeID("wavefront");
    case 1:
      return Ctx.getOrInsertSyncScopeID("workgroup");
    case 2:
      return Ctx.getOrInsertSyncScopeID("agent");
    case 3:
    default:
      return llvm::SyncScope::System;
    }
  }
  if (llvm::AMDGPU::isGFX940(ST)) {
    // CDNA3: bits[1:0] = SC1:SC0. 00=wavefront, 01=workgroup, 10=agent, 11=sys.
    unsigned Scope = CPol & 0x3;
    switch (Scope) {
    case 0:
      return Ctx.getOrInsertSyncScopeID("wavefront");
    case 1:
      return Ctx.getOrInsertSyncScopeID("workgroup");
    case 2:
      return Ctx.getOrInsertSyncScopeID("agent");
    case 3:
    default:
      return llvm::SyncScope::System;
    }
  }
  // gfx7-gfx11 (pre-CDNA3, pre-gfx12): bit 0 = GLC, bit 1 = SLC.
  //   SLC=1                ⇒ system
  //   GLC=1 SLC=0         ⇒ agent
  //   GLC=0               ⇒ workgroup
  bool GLC = CPol & 0x1;
  bool SLC = CPol & 0x2;
  if (SLC)
    return llvm::SyncScope::System;
  if (GLC)
    return Ctx.getOrInsertSyncScopeID("agent");
  return Ctx.getOrInsertSyncScopeID("workgroup");
}

llvm::AtomicOrdering
MIRToIRTranslator::getOrdering(const llvm::Value * /*CPolVal*/) const {
  // AMDGPU atomics are monotonic at the HW level. Higher orderings are
  // expressed by surrounding barrier instructions inserted by
  // SIMemoryLegalizer at lowering time, not by the atomic op itself.
  return llvm::AtomicOrdering::Monotonic;
}

void MIRToIRTranslator::fixupPhis() {
  LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] Fixing up "
                             << ToBeFixedPhis.size() << " PHI nodes\n");
  llvm::SmallVector<llvm::PHINode *> SingleValuePhis{};

  /// Resolving a per-register PHI may cause \c materializeReg on a
  /// predecessor to emit a new placeholder PHI (there, or in one of its
  /// own predecessors). Those get appended to \c ToBeFixedPhis while we
  /// iterate, so keep draining until the list is empty.
  while (!ToBeFixedPhis.empty()) {
    // Pop the back entry by value. `getOperandAsValue` below may append new
    // entries to `ToBeFixedPhis` (via materializeReg's placeholder PHIs),
    // which can grow the SmallVector and invalidate any iterator we kept into
    // it. Draining from the back is O(1) per pop (order is irrelevant — the
    // loop runs until the worklist is empty).
    ToBeFixedRegValuePhiInfo Cur = ToBeFixedPhis.pop_back_val();
    for (const llvm::MachineBasicBlock *PredMBB : Cur.MBB->predecessors()) {
      auto *PredBB = const_cast<llvm::BasicBlock *>(PredMBB->getBasicBlock());
      if (!llvm::is_contained(Cur.Phi->blocks(), PredBB)) {
        llvm::IRBuilder<llvm::InstSimplifyFolder,
                        llvm::IRBuilderCallbackInserter>
            Builder(Cur.Phi->getContext(),
                    llvm::InstSimplifyFolder{MF.getDataLayout()},
                    llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                      if (Cur.Phi->hasMetadata("amdgpu.uniform"))
                        I->setMetadata("amdgpu.uniform",
                                       llvm::MDNode::get(I->getContext(), {}));
                      LLVM_DEBUG(
                          luthier::dbgs()
                          << "[MIRToIRTranslator] Inserting instruction to "
                             "resolve phi: "
                          << *I << "\n");
                    }});
        // Insert just before the predecessor's terminator so all value-defining
        // instructions (asm calls, loads, etc.) already appear above this
        // point
        Builder.SetInsertPoint(PredBB->getTerminator());
        Cur.Phi->addIncoming(&getOperandAsValue(*PredMBB, Cur.RegKey, Builder,
                                                Cur.Phi->getType()),
                             PredBB);
      }
    }
    if (Cur.Phi->getNumIncomingValues() == 1)
      SingleValuePhis.push_back(Cur.Phi);
  }

  /// Remove single edge
  for (llvm::PHINode *P : SingleValuePhis) {
    llvm::Value *V = P->getIncomingValue(0);
    P->replaceAllUsesWith(V);
    P->eraseFromParent();
  }
}

#define GET_SI_INSTR_SEMANTIC_FUNCTIONS
#include "SIInstrSemantics.inc"

#define GET_SI_INSTR_SEMANTIC_DISPATCH
#define HANDLE_INST_SEMANTIC(OPCODE)                                           \
  case llvm::AMDGPU::OPCODE:                                                   \
    return luthier::raiseMachineInstr<llvm::AMDGPU::OPCODE>(MI, Builder, *this);

void MIRToIRTranslator::raiseMachineInstr(const llvm::MachineInstr &MI,
                                          llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] raiseMachineInstr: " << MI);

  switch (MI.getOpcode()) {

#include "SIInstrSemantics.inc"

  default: {
    LLVM_DEBUG(luthier::dbgs()
               << "[MIRToIRTranslator] Unmodelled instruction " << MI << "\n");

    InlineAsmEmitter->emitInlineAsm(
        Builder, MI,
        [&](llvm::MCRegister Reg) -> llvm::Value & {
          return getOperandAsValue(MI, Reg);
        },
        [&](llvm::MCRegister Reg, llvm::Value &Val) {
          setRegOperandValue(MI, Reg, &Val);
        });
  }
  }
}

void MIRToIRTranslator::translate() {
  auto &F = const_cast<llvm::Function &>(MF.getFunction());
  llvm::LLVMContext &Ctx = F.getContext();
  /// Early exit if there are no basic blocks in the machine function
  if (MF.empty())
    return;

  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] Translating machine function '"
             << MF.getName() << "' with " << MF.size() << " MBBs\n");

  /// Delete any basic blocks already present in the IR Function
  if (!F.empty())
    (void)F.erase(F.begin(), F.end());

  /// Create BBs associated with every MBB in the MF
  for (llvm::MachineBasicBlock &MBB : MF) {
    MBB.*get(TagBB()) = llvm::BasicBlock::Create(Ctx, "", &F);
  }

  /// If this is a kernel entry function, seed the register tracker with the
  /// hardware pre-loaded SGPR/VGPR values. Otherwise the function is a
  /// device function with the standard prototype — seed from its arguments.

  auto *EntryBB = const_cast<llvm::BasicBlock *>(MF.front().getBasicBlock());
  assert(EntryBB && "Entry MBB has no IR basic block");

  llvm::IRBuilder EntryBuilder{EntryBB};
  if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
    initKernelEntryRegs(EntryBuilder);
  else
    initDeviceFunctionEntryRegs(EntryBuilder);

  /// Iterate over the MBBs and raise the machine instructions in each MBB to
  /// LLVM IR
  for (llvm::MachineBasicBlock &MBB : MF) {
    LLVM_DEBUG(luthier::dbgs()
               << "[MIRToIRTranslator] Processing MBB " << MBB.getNumber()
               << " with " << MBB.size() << " instructions\n");
    auto *BB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock());
    for (llvm::MachineInstr &MI : MBB) {
      LLVM_DEBUG(luthier::dbgs() << "[MIRToIRTranslator] Translating MI: ";
                 MI.print(luthier::dbgs()););
      llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
          Builder(Ctx, llvm::InstSimplifyFolder{MF.getDataLayout()},
                  llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                    if (MI.getPCSections())
                      I->setMetadata(llvm::LLVMContext::MD_pcsections,
                                     MI.getPCSections());
                    if (MI.getDebugLoc())
                      I->setDebugLoc(MI.getDebugLoc());
                    LLVM_DEBUG(llvm::dbgs()
                               << "[MIRToIRTranslator] Inserting translated "
                                  "instruction "
                               << *I << "\n");
                  }});
      Builder.SetInsertPoint(BB);
      raiseMachineInstr(MI, Builder);
    }
    /// An empty MBB has no terminator instruction, so it trivially "ends in"
    /// no branch — guard \c MBB.back() against the empty case to avoid
    /// dereferencing \c --end().
    bool EndsInBranch = !MBB.empty() && MBB.back().isBranch();
    if (MBB.canFallThrough() && !EndsInBranch && !BB->getTerminatorOrNull()) {
      if (const llvm::MachineBasicBlock *NextMBB = MBB.getNextNode()) {
        auto *NextBB = const_cast<llvm::BasicBlock *>(NextMBB->getBasicBlock());
        llvm::IRBuilder{BB}.CreateBr(NextBB);
      }
    }
    /// Safety net: a well-formed trace block ends in a terminator (branch,
    /// return/tail-call, or endpgm) or falls through to a successor. If a block
    /// is still terminator-less here — e.g. a truncated trace whose last MBB
    /// ends in a non-terminator with no fall-through successor — close it with
    /// \c unreachable so the translated function stays valid IR rather than
    /// tripping the verifier in a downstream pass.
    if (!BB->getTerminatorOrNull())
      llvm::IRBuilder<>{BB}.CreateUnreachable();
  }

  /// Pass 3: insert an EXEC-mask predicate check before every vector MBB's
  /// BodyBB. The check BB receives all of the BodyBB's existing predecessor
  /// edges and dispatches to either the BodyBB (lane active) or a synthetic
  /// skip block (lane inactive). Existing per-register placeholder PHIs that
  /// were placed in the BodyBB during Pass 2 are hoisted to the CheckBB so
  /// their incoming-block list (the vector MBB's MIR predecessors) stays
  /// consistent with the IR predecessor list after the redirect.
  for (llvm::MachineBasicBlock &MBB : MF) {
    if (!luthier::isVectorMBB(MBB))
      continue;
    auto *BodyBB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock());
    auto *CheckBB = llvm::BasicBlock::Create(
        Ctx, BodyBB->hasName() ? BodyBB->getName() + ".check" : "check", &F,
        BodyBB);
    auto *SkipBB = llvm::BasicBlock::Create(
        Ctx, BodyBB->hasName() ? BodyBB->getName() + ".skip" : "skip", &F,
        BodyBB);
    VectorCheckBBs[&MBB] = CheckBB;
    ExecScaffoldBBs.insert(CheckBB);
    ExecScaffoldBBs.insert(SkipBB);

    /// Redirect every external predecessor edge from BodyBB to CheckBB.
    /// CheckBB and SkipBB are empty at this point, so any user of BodyBB
    /// outside CheckBB qualifies as "external" — the condBr/br we add
    /// below will recreate the CheckBB→BodyBB and SkipBB→BodyBB edges
    BodyBB->replaceUsesWithIf(CheckBB, [&](llvm::Use &U) {
      auto *I = llvm::dyn_cast<llvm::Instruction>(U.getUser());
      return I && I->getParent() != CheckBB && I->getParent() != SkipBB;
    });

    /// Hoist any placeholder PHI nodes from BodyBB to CheckBB. After the
    /// redirect, CheckBB's IR predecessors are exactly the MIR predecessor
    /// IR BBs, which is what the placeholder PHIs (anchored on the vector
    /// MBB in ToBeFixedPhis) expect during fixup
    while (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&BodyBB->front()))
      Phi->moveBefore(*CheckBB, CheckBB->begin());

    emitExecPredicateCheck(MBB, CheckBB, BodyBB, SkipBB);

    /// SkipBB is a degenerate pass-through to BodyBB: even for inactive
    /// lanes the body still executes (matching the hardware semantics that
    /// VALU instructions are no-ops when EXEC is zero for that lane). The
    /// IR shape exposes the per-lane predicate so downstream analyses can
    /// reason about it
    llvm::IRBuilder<>{SkipBB}.CreateBr(BodyBB);
  }

  /// Fixup all dangeling PHIs
  fixupPhis();

  /// Rewrite @llvm.amdgcn.s.{get,set}reg with a constant hwreg encoding
  /// naming a tracked register (MODE today) into direct read/write of
  /// the tracked SSA value with explicit bitfield ops, so the kernel-
  /// entry MODE constant flows through to the optimizer.
  foldHwregIntrinsics();

  /// Final cleanup: simplify and remove dead non-trace IR. Trace
  /// instructions (those whose pcsections carry a trace instruction
  /// address) are preserved verbatim
  optimizeNonTraceInsts();

  LLVM_DEBUG(luthier::dbgs()
             << "[MIRToIRTranslator] Translation complete for '" << F.getName()
             << "': " << F.size() << " basic blocks\n");
}

void MIRToIRTranslator::emitExecPredicateCheck(
    const llvm::MachineBasicBlock &VectorMBB, llvm::BasicBlock *CheckBB,
    llvm::BasicBlock *BodyBB, llvm::BasicBlock *SkipBB) {
  llvm::IRBuilder<> Builder(CheckBB);

  /// EXEC value at the entry of the CheckBB. If the vector MBB has MIR
  /// predecessors, place a placeholder PHI that fixupPhis will resolve from
  /// each predecessor's EXEC state. If it has none, the vector MBB is the
  /// function's entry block: use the EXEC value seeded at function entry —
  /// all-ones for a kernel (every lane active), or the incoming EXEC for a
  /// device function (passed in via the register-file arguments). A vector
  /// MBB never modifies EXEC (CodeDiscoveryPass splits on EXEC writes), so
  /// the entry value map still holds the entry EXEC, and the materialized
  /// value (a constant or an argument-derived value) dominates CheckBB.
  llvm::MCRegister ExecReg = TRI.getExec();
  unsigned ExecWidth = TRI.getRegSizeInBits(ExecReg, MF.getRegInfo());
  llvm::Type *ExecTy = Builder.getIntNTy(ExecWidth);
  unsigned NumMIRPreds = VectorMBB.pred_size();
  llvm::Value *ExecVal;
  if (NumMIRPreds == 0) {
    ExecVal =
        &getOperandAsValue(VectorMBB, getRegFileKey(ExecReg), Builder, ExecTy);
  } else {
    llvm::PHINode *ExecPhi =
        Builder.CreatePHI(ExecTy, NumMIRPreds, "exec.check");
    ToBeFixedPhis.push_back({&VectorMBB, getRegFileKey(ExecReg), ExecPhi});
    ExecVal = ExecPhi;
  }

  /// laneId = mbcnt.hi(-1, mbcnt.lo(-1, 0)) on wave64; on wave32 only
  /// mbcnt.lo is needed
  llvm::Type *I32 = Builder.getInt32Ty();
  llvm::Value *LaneId = Builder.CreateIntrinsic(
      I32, llvm::Intrinsic::amdgcn_mbcnt_lo,
      {Builder.getInt32(-1), Builder.getInt32(0)}, nullptr, "mbcnt.lo");
  if (ExecWidth == 64)
    LaneId = Builder.CreateIntrinsic(I32, llvm::Intrinsic::amdgcn_mbcnt_hi,
                                     {Builder.getInt32(-1), LaneId}, nullptr,
                                     "mbcnt.hi");

  llvm::Value *LaneIdExt = Builder.CreateZExtOrTrunc(LaneId, ExecTy);
  llvm::Value *Shifted = Builder.CreateLShr(ExecVal, LaneIdExt, "exec.shifted");
  llvm::Value *Bit =
      Builder.CreateAnd(Shifted, llvm::ConstantInt::get(ExecTy, 1), "exec.bit");
  llvm::Value *IsActive =
      Builder.CreateTrunc(Bit, Builder.getInt1Ty(), "exec.is.active");
  Builder.CreateCondBr(IsActive, BodyBB, SkipBB);
}

bool MIRToIRTranslator::shouldEmitGPRIndexAccess(const llvm::MachineInstr &MI,
                                                 llvm::MCRegister Reg) const {
  if (!ST.hasVGPRIndexMode())
    return false;
  if (!llvm::SIInstrInfo::isVALU(MI))
    return false;
  /// Limit Phase B's first cut to 32-bit single-VGPR operands. For
  /// wider VGPR operands (e.g. v_*_b64 reading a pair) the indexed
  /// read would need to extract two adjacent lanes — left as future
  /// work; meanwhile those fall through to the direct path.
  if (getPhysRegisterSize(Reg) != 32)
    return false;
  return TRI.isVGPR(MF.getRegInfo(), Reg);
}

llvm::Value &MIRToIRTranslator::emitIndexedVGPRSrc(const llvm::MachineInstr &MI,
                                                   llvm::MCRegister Reg,
                                                   llvm::Type *OutType) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");
  auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
  assert(BB && "MBB has no IR basic block");
  llvm::Instruction *TermInst = BB->getTerminatorOrNull();

  llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
      Builder(BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
              llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                annotateUniformIfNeeded(I, TRI, Reg);
              }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  llvm::Type *I32 = Builder.getInt32Ty();
  if (!OutType)
    OutType = I32;

  /// MODE.GPR_IDX_EN is bit 27. M0[7:0] is the index.
  llvm::Value *Mode = &getOperandAsValue(*MBB, llvm::AMDGPU::MODE, I32);
  llvm::Value *EnBit = Builder.CreateAnd(
      Builder.CreateLShr(Mode, llvm::ConstantInt::get(I32, 27)),
      llvm::ConstantInt::get(I32, 1));
  llvm::Value *En = Builder.CreateTrunc(EnBit, Builder.getInt1Ty());

  llvm::Value *M0 = &getOperandAsValue(*MBB, llvm::AMDGPU::M0, I32);
  llvm::Value *Idx = Builder.CreateAnd(M0, llvm::ConstantInt::get(I32, 0xFF));

  /// Per-slot select chain across the VGPR file from Reg to end-of-file:
  /// start with the direct read (`Reg + 0`) and, for each subsequent
  /// slot k, fold in `select(En && Idx==k, slot_k, acc)`. This avoids
  /// materializing a full-file vector via `getRegisterFile`, which would
  /// trip width-alignment assertions in the tracker's invalidation path
  /// for VGPR allocations whose total halves don't divide every cached
  /// query width. When En folds to 0 every per-slot select collapses to
  /// `acc` and the final result reduces to the direct read.
  auto Key = getRegFileKey(Reg);
  llvm::MCRegister BaseReg = std::get<0>(Key);
  unsigned BaseHalves = std::get<1>(Key);
  unsigned TotalHalves = RegFileSize.at(BaseReg);
  assert(BaseHalves <= TotalHalves && "Base offset exceeds file allocation");

  /// Direct read at base register — first slot of the chain.
  llvm::Value *Acc = &getOperandAsValue(*MBB, Reg, OutType);
  llvm::Value *AccI32 = Acc;
  if (AccI32->getType() != I32)
    AccI32 = Builder.CreateBitOrPointerCast(AccI32, I32);

  unsigned NumSlots = (TotalHalves - BaseHalves) / 2;
  for (unsigned k = 1; k < NumSlots; ++k) {
    RegFileKey SlotKey = std::make_tuple(BaseReg, BaseHalves + 2 * k, 2u);
    llvm::Value *Slot = &getOperandAsValue(*MBB, SlotKey, Builder, I32);
    llvm::Value *KEq =
        Builder.CreateICmpEQ(Idx, llvm::ConstantInt::get(I32, k));
    llvm::Value *Pick = Builder.CreateAnd(En, KEq);
    AccI32 = Builder.CreateSelect(Pick, Slot, AccI32);
  }
  if (AccI32->getType() != OutType)
    AccI32 = Builder.CreateBitOrPointerCast(AccI32, OutType);
  return *AccI32;
}

void MIRToIRTranslator::emitIndexedVGPRDst(const llvm::MachineInstr &MI,
                                           llvm::MCRegister Reg,
                                           llvm::Value *Val) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");
  auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
  assert(BB && "MBB has no IR basic block");
  llvm::Instruction *TermInst = BB->getTerminatorOrNull();

  llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
      Builder(BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
              llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                annotateUniformIfNeeded(I, TRI, Reg);
              }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  llvm::Type *I32 = Builder.getInt32Ty();
  llvm::Value *Mode = &getOperandAsValue(*MBB, llvm::AMDGPU::MODE, I32);
  llvm::Value *EnBit = Builder.CreateAnd(
      Builder.CreateLShr(Mode, llvm::ConstantInt::get(I32, 27)),
      llvm::ConstantInt::get(I32, 1));
  llvm::Value *En = Builder.CreateTrunc(EnBit, Builder.getInt1Ty());

  llvm::Value *M0 = &getOperandAsValue(*MBB, llvm::AMDGPU::M0, I32);
  llvm::Value *Idx = Builder.CreateAnd(M0, llvm::ConstantInt::get(I32, 0xFF));
  /// final_idx = en ? Idx : 0
  llvm::Value *FinalIdx =
      Builder.CreateSelect(En, Idx, llvm::ConstantInt::get(I32, 0));

  llvm::Value *ValI32 = Val;
  if (ValI32->getType() != I32) {
    unsigned ValBits = ValI32->getType()->getPrimitiveSizeInBits();
    if (ValBits < 32) {
      // Sub-32-bit writes (e.g. the f16 result of V_FMA_MIXLO_F16, which
      // targets the low 16 bits of the 32-bit vdst slot) cannot be bitcast
      // straight to i32. Mirror the non-indexed path: reinterpret as a
      // same-width integer if needed, then zero-extend into the i32 slot.
      if (!ValI32->getType()->isIntegerTy())
        ValI32 = Builder.CreateBitCast(ValI32, Builder.getIntNTy(ValBits));
      ValI32 = Builder.CreateZExt(ValI32, I32);
    } else {
      ValI32 = Builder.CreateBitOrPointerCast(ValI32, I32);
    }
  }

  /// Per-slot conditional write across the VGPR file from Reg to the end
  /// of the allocated VGPR space:
  ///
  ///   for k in [0, NumSlots):
  ///     slot_k = (FinalIdx == k) ? Val : slot_k_old
  ///
  /// Each setRegOperandValue invalidates only the single slot's overlap
  /// (the existing per-slot path), avoiding the slice-wide invalidation
  /// in `setRegisterFile` that trips `breakdownToVecTyFromAvailableValues`
  /// when mixed-width VGPR entries are already in the tracker. When
  /// FinalIdx folds to 0 (the common case: MODE.GPR_IDX_EN=0), every
  /// slot k>0's select collapses to old_k and DCE removes those writes
  /// after `optimizeNonTraceInsts`; slot 0's select collapses to Val,
  /// matching the direct write path.
  auto Key = getRegFileKey(Reg);
  llvm::MCRegister BaseReg = std::get<0>(Key);
  unsigned BaseHalves = std::get<1>(Key);
  unsigned TotalHalves = RegFileSize.at(BaseReg);
  assert(BaseHalves <= TotalHalves && "Base offset exceeds file allocation");
  /// 32-bit slots: each occupies 2 halves.
  unsigned NumSlots = (TotalHalves - BaseHalves) / 2;
  for (unsigned k = 0; k < NumSlots; ++k) {
    RegFileKey SlotKey = std::make_tuple(BaseReg, BaseHalves + 2 * k, 2u);
    llvm::Value *OldVal = &getOperandAsValue(*MBB, SlotKey, Builder, I32);
    llvm::Value *Cond =
        Builder.CreateICmpEQ(FinalIdx, llvm::ConstantInt::get(I32, k));
    llvm::Value *NewVal = Builder.CreateSelect(Cond, ValI32, OldVal);
    setRegOperandValue(*MBB, SlotKey, Builder, NewVal);
  }
}

/// Maps an AMDGPU hwreg ID (the ID field of an `s_getreg`/`s_setreg`
/// 16-bit encoding) to the MCRegister we track in the register-value map.
/// Returns std::nullopt for IDs the translator does not model — those
/// calls stay as opaque intrinsics. AMDGPU only exposes a register enum
/// entry for MODE; STATUS / TRAPSTS / HW_ID / etc. have no MCRegister
/// counterpart and so cannot be folded today.
static std::optional<llvm::MCRegister> mapHwregIDToReg(unsigned Id) {
  switch (Id) {
  case llvm::AMDGPU::Hwreg::ID_MODE:
    return llvm::MCRegister(llvm::AMDGPU::MODE);
  default:
    return std::nullopt;
  }
}

/// Decode a constant hwreg encoding into (ID, offset, width).
struct DecodedHwreg {
  unsigned Id;
  unsigned Offset;
  unsigned Width;
};
static DecodedHwreg decodeHwregEncoding(uint64_t Enc) {
  /// Encoding layout (see SIDefines.h):
  ///   bits 0..5    : ID       (6 bits)
  ///   bits 6..10   : offset   (5 bits)
  ///   bits 11..15  : size - 1 (5 bits, stored as size-1; decoded width
  ///                  = stored value + 1).
  unsigned Id = static_cast<unsigned>(Enc & 0x3F);
  unsigned Offset = static_cast<unsigned>((Enc >> 6) & 0x1F);
  unsigned Width = static_cast<unsigned>(((Enc >> 11) & 0x1F) + 1);
  return {Id, Offset, Width};
}

void MIRToIRTranslator::foldHwregIntrinsics() {
  auto &F = const_cast<llvm::Function &>(MF.getFunction());

  /// MBB lookup: each translated BB is tagged with its source MBB via
  /// `MBB.getBasicBlock()`. Build the reverse map once.
  llvm::DenseMap<const llvm::BasicBlock *, const llvm::MachineBasicBlock *>
      BBToMBB;
  for (const llvm::MachineBasicBlock &MBB : MF)
    if (const auto *BB = MBB.getBasicBlock())
      BBToMBB[BB] = &MBB;

  llvm::SmallVector<llvm::CallInst *, 16> Worklist;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);
      if (!Call)
        continue;
      llvm::Intrinsic::ID IID = Call->getIntrinsicID();
      if (IID == llvm::Intrinsic::amdgcn_s_getreg ||
          IID == llvm::Intrinsic::amdgcn_s_setreg)
        Worklist.push_back(Call);
    }
  }

  for (llvm::CallInst *Call : Worklist) {
    /// hwreg encoding is arg 0 (i32 constant).
    auto *EncC = llvm::dyn_cast<llvm::ConstantInt>(Call->getArgOperand(0));
    if (!EncC)
      continue;
    DecodedHwreg D = decodeHwregEncoding(EncC->getZExtValue());
    std::optional<llvm::MCRegister> RegOpt = mapHwregIDToReg(D.Id);
    if (!RegOpt)
      continue;
    llvm::MCRegister HwReg = *RegOpt;

    const llvm::BasicBlock *BB = Call->getParent();
    auto It = BBToMBB.find(BB);
    if (It == BBToMBB.end())
      continue;
    const llvm::MachineBasicBlock &MBB = *It->second;

    llvm::MDNode *PCS = Call->getMetadata(llvm::LLVMContext::MD_pcsections);
    llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
        Builder(F.getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
                llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                  if (PCS)
                    I->setMetadata(llvm::LLVMContext::MD_pcsections, PCS);
                }});
    Builder.SetInsertPoint(Call);

    /// Materialize the current value of the tracked register at this BB
    /// boundary as an i32. The register-tracking machinery may need to
    /// emit PHIs to plumb the value from predecessors.
    auto Key = getRegFileKey(HwReg);
    llvm::Value *RegVal = &getOperandAsValue(
        MBB, Key, Builder, llvm::Type::getInt32Ty(F.getContext()));

    llvm::Intrinsic::ID IID = Call->getIntrinsicID();

    uint32_t FieldMask = (D.Width >= 32 ? ~0u : ((1u << D.Width) - 1u));

    if (IID == llvm::Intrinsic::amdgcn_s_getreg) {
      /// result = (reg_val >> offset) & field_mask
      llvm::Value *Shifted = Builder.CreateLShr(
          RegVal, llvm::ConstantInt::get(Builder.getInt32Ty(), D.Offset));
      llvm::Value *Masked = Builder.CreateAnd(
          Shifted, llvm::ConstantInt::get(Builder.getInt32Ty(), FieldMask));
      Call->replaceAllUsesWith(Masked);
      Call->eraseFromParent();
    } else {
      /// new_reg_val = (reg_val & ~(field_mask << offset))
      ///             | ((src & field_mask) << offset)
      llvm::Value *Src = Call->getArgOperand(1);
      uint32_t SlotMask = FieldMask << D.Offset;
      llvm::Value *Cleared = Builder.CreateAnd(
          RegVal, llvm::ConstantInt::get(Builder.getInt32Ty(), ~SlotMask));
      llvm::Value *Field = Builder.CreateAnd(
          Src, llvm::ConstantInt::get(Builder.getInt32Ty(), FieldMask));
      llvm::Value *Shifted = Builder.CreateShl(
          Field, llvm::ConstantInt::get(Builder.getInt32Ty(), D.Offset));
      llvm::Value *NewReg = Builder.CreateOr(Cleared, Shifted);
      setRegOperandValue(MBB, Key, Builder, NewReg);
      Call->eraseFromParent();
    }
  }
}

void MIRToIRTranslator::optimizeNonTraceInsts() {
  auto &F = const_cast<llvm::Function &>(MF.getFunction());
  const llvm::DataLayout &DL = MF.getDataLayout();
  llvm::SimplifyQuery SQ(DL);

  auto IsTrace = [](const llvm::Instruction *I) -> bool {
    auto *MD = I->getMetadata(llvm::LLVMContext::MD_pcsections);
    if (!MD)
      return false;
    auto *TMD = llvm::dyn_cast<TargetMachineInstrMDNode>(MD);
    if (!TMD)
      return false;
    return TMD->getTraceInstrAddress().has_value();
  };

  llvm::SmallVector<llvm::WeakTrackingVH, 64> Worklist;
  for (llvm::BasicBlock &BB : F) {
    if (ExecScaffoldBBs.contains(&BB))
      continue;
    for (llvm::Instruction &I : BB)
      if (!IsTrace(&I))
        Worklist.emplace_back(&I);
  }

  while (!Worklist.empty()) {
    llvm::WeakTrackingVH WH = Worklist.pop_back_val();
    auto *I = llvm::dyn_cast_or_null<llvm::Instruction>(WH);
    if (!I)
      continue;
    if (IsTrace(I))
      continue;
    if (ExecScaffoldBBs.contains(I->getParent()))
      continue;
    if (llvm::Value *V = llvm::simplifyInstruction(I, SQ)) {
      for (llvm::User *U : I->users())
        if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U))
          if (!IsTrace(UI))
            Worklist.emplace_back(UI);
      /// Carry register-provenance tags from \p I onto its surviving
      /// replacement so downstream passes can still trace which physical
      /// register a folded value represents.
      if (I->hasMetadata(RegValueMDKindName)) {
        if (auto *RI = llvm::dyn_cast<llvm::Instruction>(V)) {
          mergeRegValues(*RI, *I);
        } else {
          llvm::SmallVector<RegValueDesc, 2> Descs;
          getRegValues(*I, Descs);
          auto *MD = I->getMetadata(RegValueMDKindName);
          auto &MF = F;
          for (unsigned K = 0; K < Descs.size(); ++K) {
            llvm::StringRef Name;
            if (auto *Entry = llvm::dyn_cast<llvm::MDNode>(MD->getOperand(K)))
              if (auto *S =
                      llvm::dyn_cast<llvm::MDString>(Entry->getOperand(0)))
                Name = S->getString();
            addEntryRegMapping(MF, V, Descs[K], Name);
          }
        }
      }
      I->replaceAllUsesWith(V);
    }
    if (llvm::isInstructionTriviallyDead(I)) {
      for (llvm::Use &Op : I->operands())
        if (auto *OI = llvm::dyn_cast<llvm::Instruction>(Op.get()))
          if (!IsTrace(OI))
            Worklist.emplace_back(OI);
      I->eraseFromParent();
    }
  }
}

} // namespace luthier
