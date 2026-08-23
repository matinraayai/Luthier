//===-- TraceFunctionTranslator.cpp ---------------------------------------===//
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
/// \file TraceFunctionTranslator.cpp
/// Implements a set of APIs used to translate machine functions and
/// individual machine instructions to LLVM IR.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TraceFunctionTranslator.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
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
#include <llvm/IR/CFG.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IRBuilderFolder.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/ValueHandle.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Transforms/Utils/Local.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-trace-function-translator"

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

/// Recursive worker for \c isProvablyAllOnesInt. \p Visited is used to
/// break PHI cycles: when we re-encounter a node we're currently
/// analyzing, we optimistically treat the back-edge as if it produces
/// all-ones. If every non-cycle leaf turns out to be a \c -1 constant,
/// this optimism holds; if any leaf is not, propagation up returns
/// false and the fold is rejected.
bool isProvablyAllOnesIntImpl(const llvm::Value *V,
                              llvm::SmallPtrSetImpl<const llvm::Value *>
                                  &Visited) {
  if (!V)
    return false;
  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V))
    return CI->isMinusOne();
  if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(V)) {
    if (Phi->getNumIncomingValues() == 0)
      return false;
    if (!Visited.insert(Phi).second)
      return true;
    for (const llvm::Value *In : Phi->incoming_values())
      if (!isProvablyAllOnesIntImpl(In, Visited))
        return false;
    return true;
  }
  return false;
}

/// True iff \p V is provably an all-ones bit pattern of its integer
/// type. Handles a constant \c -1, a PHI that transitively merges only
/// \c -1 constants (with arbitrary cycles among intermediate PHIs), and
/// combinations thereof. Non-PHI / non-constant values are treated
/// conservatively.
bool isProvablyAllOnesInt(const llvm::Value *V) {
  llvm::SmallPtrSet<const llvm::Value *, 8> Visited;
  return isProvablyAllOnesIntImpl(V, Visited);
}

/// AMDGPU-specific compile-time fold for target intrinsics whose result
/// can be computed statically from their operands.
///
/// Currently handles:
///   * <tt>llvm.amdgcn.readfirstlane(<Constant>)</tt> \c → \c <Constant>
///     — the value is uniform, so cross-lane broadcast is a no-op.
///   * <tt>llvm.amdgcn.readlane(<Constant>, <any>)</tt> \c → \c <Constant>
///     — same rationale, regardless of the lane index.
///   * <tt>llvm.amdgcn.mbcnt.lo(0, X)</tt> \c → \c X — with a zero mask,
///     no lane's bit is counted so the popcount contributes 0.
///   * <tt>llvm.amdgcn.mbcnt.hi(0, X)</tt> \c → \c X — same.
llvm::Value *tryFoldAMDGPUIntrinsic(llvm::Intrinsic::ID ID,
                                    llvm::ArrayRef<llvm::Value *> Args) {
  switch (ID) {
  case llvm::Intrinsic::amdgcn_readfirstlane:
    if (Args.size() >= 1 && llvm::isa<llvm::Constant>(Args[0]))
      return Args[0];
    return nullptr;
  case llvm::Intrinsic::amdgcn_readlane:
    if (Args.size() >= 1 && llvm::isa<llvm::Constant>(Args[0]))
      return Args[0];
    return nullptr;
  case llvm::Intrinsic::amdgcn_mbcnt_lo:
  case llvm::Intrinsic::amdgcn_mbcnt_hi:
    if (Args.size() >= 2)
      if (auto *M = llvm::dyn_cast<llvm::ConstantInt>(Args[0]);
          M && M->isZero())
        return Args[1];
    return nullptr;
  default:
    return nullptr;
  }
}

/// A folder for \c llvm::IRBuilder that layers AMDGPU-specific
/// intrinsic simplification on top of \c llvm::InstSimplifyFolder.
class LuthierAMDGPUFolder final : public llvm::IRBuilderFolder {
  llvm::InstSimplifyFolder Base;

public:
  explicit LuthierAMDGPUFolder(const llvm::DataLayout &DL) : Base(DL) {}

  llvm::Value *FoldBinOp(llvm::Instruction::BinaryOps Opc, llvm::Value *LHS,
                         llvm::Value *RHS) const override {
    return Base.FoldBinOp(Opc, LHS, RHS);
  }

  llvm::Value *FoldExactBinOp(llvm::Instruction::BinaryOps Opc,
                              llvm::Value *LHS, llvm::Value *RHS,
                              bool IsExact) const override {
    return Base.FoldExactBinOp(Opc, LHS, RHS, IsExact);
  }
  llvm::Value *FoldNoWrapBinOp(llvm::Instruction::BinaryOps Opc,
                               llvm::Value *LHS, llvm::Value *RHS, bool HasNUW,
                               bool HasNSW) const override {
    return Base.FoldNoWrapBinOp(Opc, LHS, RHS, HasNUW, HasNSW);
  }
  llvm::Value *FoldBinOpFMF(llvm::Instruction::BinaryOps Opc, llvm::Value *LHS,
                            llvm::Value *RHS,
                            llvm::FastMathFlags FMF) const override {
    return Base.FoldBinOpFMF(Opc, LHS, RHS, FMF);
  }
  llvm::Value *FoldUnOpFMF(llvm::Instruction::UnaryOps Opc, llvm::Value *V,
                           llvm::FastMathFlags FMF) const override {
    return Base.FoldUnOpFMF(Opc, V, FMF);
  }
  llvm::Value *FoldCmp(llvm::CmpInst::Predicate P, llvm::Value *LHS,
                       llvm::Value *RHS) const override {
    return Base.FoldCmp(P, LHS, RHS);
  }
  llvm::Value *FoldGEP(llvm::Type *Ty, llvm::Value *Ptr,
                       llvm::ArrayRef<llvm::Value *> IdxList,
                       llvm::GEPNoWrapFlags NW) const override {
    return Base.FoldGEP(Ty, Ptr, IdxList, NW);
  }
  llvm::Value *FoldSelect(llvm::Value *C, llvm::Value *True, llvm::Value *False,
                          llvm::FastMathFlags FMF =
                              llvm::FastMathFlags()) const override {
    return Base.FoldSelect(C, True, False, FMF);
  }
  llvm::Value *
  FoldExtractValue(llvm::Value *Agg,
                   llvm::ArrayRef<unsigned> IdxList) const override {
    return Base.FoldExtractValue(Agg, IdxList);
  }
  llvm::Value *
  FoldInsertValue(llvm::Value *Agg, llvm::Value *Val,
                  llvm::ArrayRef<unsigned> IdxList) const override {
    return Base.FoldInsertValue(Agg, Val, IdxList);
  }
  llvm::Value *FoldExtractElement(llvm::Value *Vec,
                                  llvm::Value *Idx) const override {
    return Base.FoldExtractElement(Vec, Idx);
  }
  llvm::Value *FoldInsertElement(llvm::Value *Vec, llvm::Value *NewElt,
                                 llvm::Value *Idx) const override {
    return Base.FoldInsertElement(Vec, NewElt, Idx);
  }
  llvm::Value *FoldShuffleVector(llvm::Value *V1, llvm::Value *V2,
                                 llvm::ArrayRef<int> Mask) const override {
    return Base.FoldShuffleVector(V1, V2, Mask);
  }
  llvm::Value *FoldCast(llvm::Instruction::CastOps Op, llvm::Value *V,
                        llvm::Type *DestTy) const override {
    return Base.FoldCast(Op, V, DestTy);
  }
  llvm::Value *FoldUnaryIntrinsic(
      llvm::Intrinsic::ID ID, llvm::Value *Op, llvm::Type *Ty,
      llvm::FastMathFlags FMF = llvm::FastMathFlags()) const override {
    if (llvm::Value *V = tryFoldAMDGPUIntrinsic(ID, {Op}))
      return V;
    return Base.FoldUnaryIntrinsic(ID, Op, Ty, FMF);
  }
  llvm::Value *FoldBinaryIntrinsic(
      llvm::Intrinsic::ID ID, llvm::Value *LHS, llvm::Value *RHS,
      llvm::Type *Ty,
      llvm::FastMathFlags FMF = llvm::FastMathFlags()) const override {
    if (llvm::Value *V = tryFoldAMDGPUIntrinsic(ID, {LHS, RHS}))
      return V;
    return Base.FoldBinaryIntrinsic(ID, LHS, RHS, Ty, FMF);
  }
  llvm::Value *CreatePointerCast(llvm::Constant *C,
                                 llvm::Type *DestTy) const override {
    return Base.CreatePointerCast(C, DestTy);
  }
  llvm::Value *
  CreatePointerBitCastOrAddrSpaceCast(llvm::Constant *C,
                                      llvm::Type *DestTy) const override {
    return Base.CreatePointerBitCastOrAddrSpaceCast(C, DestTy);
  }
};

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

void TraceFunctionTranslator::invalidateOverlaps(
    RegValueMap &State, const RegFileKey &WrittenRegKey,
    llvm::IRBuilderBase &Builder) {
  llvm::MCRegister BaseReg = std::get<0>(WrittenRegKey);
  const unsigned WStart = std::get<1>(WrittenRegKey);
  const unsigned WNumHalves = std::get<2>(WrittenRegKey);
  const unsigned WEnd = WStart + WNumHalves;
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] invalidateOverlaps: "
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
             << "[TraceFunctionTranslator] invalidateOverlaps: Erasing "
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

llvm::Value *TraceFunctionTranslator::extractChunkFromSource(
    RegValueMap &State, const RegFileKey &RegKey, unsigned VecChunkSize,
    unsigned Idx, unsigned NumChunks, llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs()
                 << "[TraceFunctionTranslator] extractChunkFromSource: "
                    "base="
                 << TRI.getName(std::get<0>(RegKey))
                 << " offset=" << std::get<1>(RegKey) << " Idx=" << Idx
                 << " NumChunks=" << NumChunks << " chunkSize=" << VecChunkSize
                 << "\n";);
  auto &RegValueMap = State[RegKey];
  const unsigned KeyTotalWidth = std::get<2>(RegKey) * RegGranule;
  unsigned VecChunkRegGranMul = VecChunkSize / RegGranule;
  unsigned ChunkSizeInRegGran = NumChunks * VecChunkRegGranMul;
  llvm::Type *ChunkIntTy = Builder.getIntNTy(VecChunkSize);

  // Fast path: source width tiles cleanly into VecChunkSize lanes. Use
  // vector extractelement.
  if (KeyTotalWidth % VecChunkSize == 0) {
    if (NumChunks == 1 && KeyTotalWidth == VecChunkSize) {
      assert(Idx == 0 && "single-lane source: only index 0 is valid");
      if (auto It = RegValueMap.find(ChunkIntTy); It != RegValueMap.end())
        return It->second;
      llvm::Value *Pivot = nullptr;
      for (auto &[T, V] : RegValueMap) {
        if (T->getPrimitiveSizeInBits() == VecChunkSize &&
            (T->isIntOrIntVectorTy() || T->isFPOrFPVectorTy())) {
          Pivot = V;
          break;
        }
      }
      if (!Pivot)
        Pivot = getOrCreateIntOrFloatTypeForReg(RegValueMap, Builder);
      llvm::Value *Out = Builder.CreateBitOrPointerCast(Pivot, ChunkIntTy);
      RegValueMap[ChunkIntTy] = Out;
      return Out;
    }

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

llvm::Value *TraceFunctionTranslator::materializeFromOverlapping(
    RegValueMap &State, const llvm::BasicBlock &BB,
    const RegFileKey &ReadKeyReg, llvm::IRBuilderBase &Builder,
    llvm::Type &RegType) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] materializeFromOverlapping\n");

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
    const unsigned NumIRPreds = llvm::pred_size(&BB);
    if (NumIRPreds != 0) {
      // Create PHI for entire register. Anchor at top of BB regardless of
      // where Builder's insertion point is.
      llvm::PHINode *Phi = llvm::PHINode::Create(
          &RegType, NumIRPreds, "",
          const_cast<llvm::BasicBlock &>(BB).begin());
      ToBeFixedPhis.emplace_back(&BB, ReadKeyReg, Phi);
      State[ReadKeyReg][&RegType] = Phi;
      return Phi;
    }
    // Entry block - freeze(poison)
    llvm::Value *InitVal =
        Builder.CreateFreeze(llvm::PoisonValue::get(&RegType));
    State[ReadKeyReg][&RegType] = InitVal;
    return InitVal;
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
    const unsigned NumIRPreds = llvm::pred_size(&BB);
    if (NumIRPreds == 0) {
      // Entry block - freeze(poison) for the missing value
      DefaultVal = Builder.CreateFreeze(llvm::PoisonValue::get(ValTy));
    } else {
      // Has predecessors - create PHI for the missing value at top of BB.
      llvm::PHINode *Phi = llvm::PHINode::Create(
          ValTy, NumIRPreds, "", const_cast<llvm::BasicBlock &>(BB).begin());
      ToBeFixedPhis.emplace_back(&BB, NonOverlappingSubKey, Phi);
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

  const unsigned NumLanes = RNumHalves / OptimalNumHalves;
  const unsigned OptimalChunkSizeInBits = RegGranule * OptimalNumHalves;

  // Materialize a single OptimalChunkSize-wide chunk from source K, cache
  // it under its absolute sub-key, and return the cast-to-IntN value.
  // SrcChunkStart is the source-relative offset (in halves) within K;
  // AbsChunkStart is the chunk's absolute offset in the register file.
  auto ExtractOneChunk = [&](unsigned SrcChunkStart, unsigned AbsChunkStart,
                             const RegFileKey &K) -> llvm::Value * {
    RegFileKey SubRegKey =
        std::make_tuple(BaseReg, AbsChunkStart, OptimalNumHalves);
    unsigned SrcElIdx = SrcChunkStart / OptimalNumHalves;
    llvm::Value *ChunkVal = extractChunkFromSource(
        State, K, OptimalChunkSizeInBits, SrcElIdx, 1, Builder);
    State[SubRegKey][ChunkVal->getType()] = ChunkVal;
    ChunkVal = Builder.CreateBitOrPointerCast(
        ChunkVal, Builder.getIntNTy(OptimalChunkSizeInBits));
    State[SubRegKey][ChunkVal->getType()] = ChunkVal;
    return ChunkVal;
  };

  // Fast path: the requested slot is exactly one chunk wide. Wrapping the
  // scalar in a <1 x T> and bitcasting back to T produces an
  // insertelement+bitcast pair that InstSimplify (run by
  // optimizeNonTraceInsts) does not fold, so return the chunk value
  // directly. Exactly one of {NonOverlap, Overlap} carries the chunk in
  // this case because they partition [RStart, REnd).
  if (NumLanes == 1) {
    llvm::Value *ChunkVal = nullptr;
    if (!NonOverlapChunks.empty()) {
      const auto &C = NonOverlapChunks[0];
      ChunkVal = ExtractOneChunk(0, C.ChunkStart, C.KeyReg);
    } else {
      const auto &C = OverlapChunks[0];
      ChunkVal =
          ExtractOneChunk(C.SrcChunkStart, RStart + C.ChunkStart, C.Src->RegKey);
    }
    return Builder.CreateBitOrPointerCast(ChunkVal, &RegType);
  }

  // Step 6: Construct a vector type to materialize chunks
  auto *WorkingTy = llvm::FixedVectorType::get(
      Builder.getIntNTy(OptimalChunkSizeInBits), NumLanes);
  llvm::Value *Result = llvm::PoisonValue::get(WorkingTy);

  // InsertChunkFn inserts chunks into Result.
  // SrcChunkStart: source-relative start offset (in halves) within K.
  // ChunkStart / ChunkEnd: RStart-relative offsets (in halves) in the result.
  auto InsertChunkFn = [&](unsigned SrcChunkStart, unsigned ChunkStart,
                           unsigned ChunkEnd, const RegFileKey &K) {
    unsigned NumChunks = ChunkEnd - ChunkStart;
    for (unsigned CI = 0; CI < NumChunks; CI += OptimalNumHalves) {
      unsigned DestElIdx = (ChunkStart + CI) / OptimalNumHalves;
      llvm::Value *ChunkVal = ExtractOneChunk(
          SrcChunkStart + CI, RStart + ChunkStart + CI, K);
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
TraceFunctionTranslator::getOperandAsValue(const llvm::BasicBlock &BB,
                                           llvm::MCRegister Reg,
                                           llvm::Type *OutRegType) {
  llvm::StringRef RegName = TRI.getName(Reg);
  std::string RegValName = getRegValueName(Reg);

  LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                 "[TraceFunctionTranslator] Materializing register {0} "
                 "in BB '{1}'\n",
                 RegName, BB.getName()));
  (void)RegName;

  auto *MutableBB = const_cast<llvm::BasicBlock *>(&BB);
  llvm::Instruction *TermInst = MutableBB->getTerminatorOrNull();

  llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
      Builder(
          MutableBB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
          llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
            annotateUniformIfNeeded(I, TRI, Reg);
            LLVM_DEBUG(
                luthier::dbgs()
                << "[TraceFunctionTranslator] Inserting reg read instruction "
                << *I << "\n");
          }});
  TermInst ? Builder.SetInsertPoint(TermInst)
           : Builder.SetInsertPoint(MutableBB);

  return getOperandAsValue(BB, getRegFileKey(Reg), Builder, OutRegType);
}

llvm::Value &TraceFunctionTranslator::getOperandAsValue(
    const llvm::BasicBlock &BB, const RegFileKey &Key,
    llvm::IRBuilderBase &Builder, llvm::Type *OutRegType) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] getOperandAsValue: BB '"
             << BB.getName() << "' base=" << TRI.getName(std::get<0>(Key))
             << " offset=" << std::get<1>(Key) << " halves=" << std::get<2>(Key)
             << "\n");
  RegValueMap &State = VM[&BB];
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
      materializeFromOverlapping(State, BB, Key, Builder, *OutRegType);
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
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Building initial MODE "
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

void TraceFunctionTranslator::initKernelEntryRegs(
    llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Initializing kernel entry "
                "registers for '"
             << MF.getName() << "'\n");
  const auto &Info = *MF.getInfo<llvm::SIMachineFunctionInfo>();

  using PV = llvm::AMDGPUFunctionArgInfo::PreloadedValue;

  auto seedRegValue = [&](const llvm::MachineBasicBlock &MBB,
                          llvm::MCRegister Reg, llvm::Value *Val) {
    const llvm::BasicBlock *BB = MBB.getBasicBlock();
    assert(BB && "MBB has no IR basic block");
    RegValueMap &State = VM[BB];
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
      const llvm::BasicBlock *EntryIRBB = MF.front().getBasicBlock();
      assert(EntryIRBB && "Entry MBB has no IR basic block");
      RegValueMap &State = VM[EntryIRBB];
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

TraceFunctionTranslator::TraceFunctionTranslator(llvm::MachineFunction &MF,
                                                 llvm::Error &Err)
    : MF(MF), TRI(*MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo()),
      TII(*MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo()),
      ST(MF.getSubtarget<llvm::GCNSubtarget>()) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Creating translator for '"
             << MF.getName() << "' with " << MF.size() << " MBBs\n");
  llvm::ErrorAsOutParameter EAO(Err);

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

llvm::MCRegister
TraceFunctionTranslator::getPhysReg(llvm::MCRegister Reg) const {
  switch (Reg) {
  case llvm::AMDGPU::SCC:
    return llvm::AMDGPU::SRC_SCC;
  default:
    return llvm::AMDGPU::getMCReg(Reg, ST);
  }
}

unsigned
TraceFunctionTranslator::getPhysRegisterSize(llvm::MCRegister Reg) const {
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

llvm::Error TraceFunctionTranslator::initRegFileLayouts() {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Initializing register file "
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

TraceFunctionTranslator::RegFileKey
TraceFunctionTranslator::getRegFileKey(llvm::MCRegister Reg) const {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] getRegFileKey for reg "
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
      TRI.getEncodingValue(getPhysReg(BaseReg)) &
                       llvm::AMDGPU::HWEncoding::REG_IDX_MASK;

  unsigned Offset = (HwIdx - BaseHwIdx) * 2 + IsHi16;

  unsigned RegSizeBits = getPhysRegisterSize(Reg);

  auto Key = std::make_tuple(BaseReg, Offset, RegSizeBits / RegGranule);
  LLVM_DEBUG(luthier::dbgs() << "[TraceFunctionTranslator] -> Key: base="
                             << TRI.getName(BaseReg) << " offset=" << Offset
                             << " halves=" << std::get<2>(Key) << "\n");
  return Key;
}

std::string
TraceFunctionTranslator::getRegfileValueName(llvm::MCRegister BaseReg) {
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

llvm::Value *TraceFunctionTranslator::getRegisterFile(
    const llvm::BasicBlock &BB, llvm::MCRegister Reg,
    llvm::IRBuilderBase &Builder, llvm::Type *LaneTy) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] getRegisterFile: BB '"
             << BB.getName() << "' reg=" << TRI.getName(Reg) << "\n");
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
  llvm::Value *FullVec = &getOperandAsValue(BB, FullKey, Builder, FullVecTy);

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

llvm::Value *
TraceFunctionTranslator::getRegisterFile(const llvm::MachineInstr &MI,
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
      Builder(
          BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
          llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
            annotateUniformIfNeeded(I, TRI, Register);
            LLVM_DEBUG(
                luthier::dbgs()
                << "[TraceFunctionTranslator] Inserting read reg instruction "
                << *I << "\n");
          }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  return getRegisterFile(*BB, Register, Builder, LaneTy);
}

void TraceFunctionTranslator::setRegisterFile(const llvm::MachineInstr &MI,
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
      Builder(
          BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
          llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
            annotateUniformIfNeeded(I, TRI, Reg);
            LLVM_DEBUG(
                luthier::dbgs()
                << "[TraceFunctionTranslator] Inserting read reg instruction "
                << *I << "\n");
          }});
  TermInst ? Builder.SetInsertPoint(TermInst) : Builder.SetInsertPoint(BB);

  setRegisterFile(*BB, Reg, Builder, NewVec);
}

llvm::Value *
TraceFunctionTranslator::getRegisterFile(const llvm::MachineInstr &MI,
                                         llvm::AMDGPU::OpName OpName,
                                         llvm::Type *LaneTy) {
  const llvm::MachineOperand *Op = TII.getNamedOperand(MI, OpName);
  assert(Op && Op->isReg() &&
         "GetRegisterFile target operand is not a register operand");
  return getRegisterFile(MI, Op->getReg(), LaneTy);
}

void TraceFunctionTranslator::setRegisterFile(const llvm::MachineInstr &MI,
                                              llvm::AMDGPU::OpName OpName,
                                              llvm::Value *NewVec) {
  const llvm::MachineOperand *Op = TII.getNamedOperand(MI, OpName);
  assert(Op && Op->isReg() &&
         "SetRegisterFile target operand is not a register operand");
  setRegisterFile(MI, Op->getReg(), NewVec);
}

void TraceFunctionTranslator::setRegisterFile(
    const llvm::BasicBlock &BB, llvm::MCRegister Reg,
    llvm::IRBuilderBase &Builder, llvm::Value *Val) {
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
    llvm::Value *OldFull = &getOperandAsValue(BB, FullKey, Builder, FullVecTy);
    NewFull = OldFull;
    unsigned SliceLanes = SliceVecTy->getNumElements();
    for (unsigned I = 0; I < SliceLanes; ++I) {
      llvm::Value *Lane = Builder.CreateExtractElement(Val, I);
      NewFull = Builder.CreateInsertElement(NewFull, Lane, StartLane + I);
    }
  }

  setRegOperandValue(BB, FullKey, Builder, NewFull);
}

llvm::FunctionType *
TraceFunctionTranslator::getStandardDeviceFunctionType() const {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Getting standard device "
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
TraceFunctionTranslator::computeStandardDeviceFunctionType(
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

  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] device function type: " << *FuncTy
             << "\n");
  return FuncTy;
}

void TraceFunctionTranslator::initDeviceFunctionEntryRegs(
    llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Initializing device function "
                "entry registers for '"
             << MF.getName() << "' with " << MF.getFunction().arg_size()
             << " arguments\n");
  llvm::Function &F = const_cast<llvm::Function &>(MF.getFunction());

  const llvm::MachineBasicBlock &EntryMBB = MF.front();
  const llvm::BasicBlock *EntryIRBB = EntryMBB.getBasicBlock();
  assert(EntryIRBB && "Entry MBB has no IR basic block");
  RegValueMap &State = VM[EntryIRBB];

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

void TraceFunctionTranslator::emitDirectTailCall(const llvm::MachineInstr &MI,
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

void TraceFunctionTranslator::emitIndirectTailCall(const llvm::MachineInstr &MI,
                                                   llvm::IRBuilderBase &Builder,
                                                   llvm::Value *Target) {
  if (!Target) {
    // CodeDiscoveryPass couldn't resolve the call target (e.g. S_CALL_B64
    // with an unresolved address). Skip emission rather than crash — the
    // MIR still records the call site for downstream analysis.
    LLVM_DEBUG(luthier::dbgs()
               << "[TraceFunctionTranslator] Skipping call emission in MBB "
               << MI.getParent()->getNumber() << ": target is nullptr\n");
    return;
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Emitting indirect tail call "
                "in MBB "
             << MI.getParent()->getNumber() << " target=" << *Target << "\n");
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");
  const llvm::BasicBlock *BB = MBB->getBasicBlock();
  assert(BB && "MBB has no IR basic block");

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
          *BB, std::make_tuple(RegFileBase, PI * 2, 2), Builder));
    }
  }

  llvm::CallInst *Call = Builder.CreateCall(FTy, FuncPtr, CallArgs);
  Call->setTailCallKind(llvm::CallInst::TCK_Tail);

}

llvm::Value &
TraceFunctionTranslator::getOperandAsValue(const llvm::MachineInstr &MI,
                                           llvm::AMDGPU::OpName OpName,
                                           llvm::Type *OutType) {
  return getOperandAsValue(*TII.getNamedOperand(MI, OpName), OutType);
}

llvm::Value &TraceFunctionTranslator::getOperandAsValue(
    const llvm::MachineInstr &MI, llvm::MCRegister Reg, llvm::Type *RegType) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI does not have a machine basic block");
  if (shouldEmitGPRIndexAccess(MI, Reg))
    return emitIndexedVGPRSrc(MI, Reg, RegType);
  const llvm::BasicBlock *BB = MBB->getBasicBlock();
  assert(BB && "MBB does not have an IR basic block");
  return getOperandAsValue(*BB, Reg, RegType);
}

llvm::Value &
TraceFunctionTranslator::getOperandAsValue(const llvm::MachineOperand &Op,
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
TraceFunctionTranslator::getOperandAsBasicBlock(const llvm::MachineInstr &MI,
                                                llvm::AMDGPU::OpName OpName) {
  return getOperandAsBasicBlock(*TII.getNamedOperand(MI, OpName));
}

llvm::BasicBlock &TraceFunctionTranslator::getOperandAsBasicBlock(
    const llvm::MachineOperand &Op) {
  auto *BB = const_cast<llvm::BasicBlock *>(Op.getMBB()->getBasicBlock());
  assert(BB && "MBB operand has no IR BasicBlock");
  return *BB;
}

llvm::Function *
TraceFunctionTranslator::getOperandAsFunction(const llvm::MachineInstr &MI,
                                              llvm::AMDGPU::OpName OpName) {
  const llvm::MachineOperand *Op = TII.getNamedOperand(MI, OpName);
  if (!Op || !Op->isGlobal())
    return nullptr;
  return const_cast<llvm::Function *>(
      llvm::dyn_cast<llvm::Function>(Op->getGlobal()));
}

void TraceFunctionTranslator::setRegOperandValue(const llvm::MachineInstr &MI,
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
      Builder(
          BB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()},
          llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
            annotateUniformIfNeeded(I, TRI, Reg);
            LLVM_DEBUG(
                luthier::dbgs()
                << "[TraceFunctionTranslator] Inserting reg write instruction "
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

  LLVM_DEBUG(
      luthier::dbgs() << llvm::formatv(
          "[TraceFunctionTranslator] Setting register {0} to value {3} for "
          "MBB {1} (type: {2})\n",
          TRI.getName(Reg), MBB->getNumber(), *Val->getType()->getScalarType(),
          *Val));

  setRegOperandValue(*BB, getRegFileKey(Reg), Builder, Val);
}

void TraceFunctionTranslator::setRegOperandValue(const llvm::MachineOperand &Op,
                                                 llvm::Value *Val) {
  assert(Val && "Val is nullptr");
  assert(Op.isReg() && "Operand is not a register");
  assert(Op.getReg().isPhysical() && "Operand is not a physical register");
  const llvm::MachineInstr *MI = Op.getParent();
  assert(MI && "Machine operand has no parent MI");
  setRegOperandValue(*MI, Op.getReg(), Val);
}

void TraceFunctionTranslator::setRegOperandValue(
    const llvm::BasicBlock &BB, const RegFileKey &Key,
    llvm::IRBuilderBase &Builder, llvm::Value *Val) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] setRegOperandValue: BB '"
             << BB.getName() << "' base=" << TRI.getName(std::get<0>(Key))
             << " offset=" << std::get<1>(Key) << " halves=" << std::get<2>(Key)
             << " val=" << *Val << " (type=" << *Val->getType() << ")\n");
  RegValueMap &State = VM[&BB];
  llvm::MCRegister BaseReg = std::get<0>(Key);
  unsigned Offset = std::get<1>(Key);
  unsigned Size = std::get<2>(Key);

  /// Bounds check: silently drop writes that target an unallocated plain
  /// GPR slot. Specials (TTMP/M0/EXEC/NULL/VCC-on-GFX10+) bypass the
  /// check and are always written through.
  unsigned Allocated = RegFileSize.at(BaseReg);
  if (Offset + Size > Allocated) {
    LLVM_DEBUG(luthier::dbgs()
               << "[TraceFunctionTranslator] Dropping out-of-range write to "
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

void TraceFunctionTranslator::setRegOperandValue(const llvm::MachineInstr &MI,
                                                 llvm::AMDGPU::OpName OpName,
                                                 llvm::Value *Val) {
  setRegOperandValue(*TII.getNamedOperand(MI, OpName), Val);
}

llvm::BasicBlock *
TraceFunctionTranslator::getNextBB(const llvm::MachineInstr &MI) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI does not have a basic block");
  const llvm::MachineBasicBlock *NextMBB = MBB->getNextNode();
  assert(NextMBB && "MI doesn't have a fall-through block");

  return const_cast<llvm::BasicBlock *>(NextMBB->getBasicBlock());
}

llvm::SyncScope::ID
TraceFunctionTranslator::getSyncScope(const llvm::Value *CPolVal) const {
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
TraceFunctionTranslator::getOrdering(const llvm::Value * /*CPolVal*/) const {
  // AMDGPU atomics are monotonic at the HW level. Higher orderings are
  // expressed by surrounding barrier instructions inserted by
  // SIMemoryLegalizer at lowering time, not by the atomic op itself.
  return llvm::AtomicOrdering::Monotonic;
}

void TraceFunctionTranslator::fixupPhis() {
  LLVM_DEBUG(luthier::dbgs() << "[TraceFunctionTranslator] Fixing up "
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
    // Walk the IR CFG for predecessors. Under the
    // diamond scaffold, a vector MBB's IR entry is CheckBB with two
    // IR successors (BodyBB, SkipBB) — both of which may be
    // predecessors of a downstream vector MBB's CheckBB. Iterating IR
    // predecessors naturally covers both edges; the recursive
    // \c getOperandAsValue resolves each predecessor's exit-state
    // materialization on demand.
    for (llvm::BasicBlock *PredBB :
         llvm::predecessors(const_cast<llvm::BasicBlock *>(Cur.BB))) {
      if (llvm::is_contained(Cur.Phi->blocks(), PredBB))
        continue;
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
                        << "[TraceFunctionTranslator] Inserting instruction to "
                           "resolve phi: "
                        << *I << "\n");
                  }});
      // Insert just before the predecessor's terminator so all value-
      // defining instructions (asm calls, loads, etc.) already appear
      // above this point.
      Builder.SetInsertPoint(PredBB->getTerminator());
      Cur.Phi->addIncoming(
          &getOperandAsValue(*PredBB, Cur.RegKey, Builder,
                             Cur.Phi->getType()),
          PredBB);
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

void TraceFunctionTranslator::raiseMachineInstr(const llvm::MachineInstr &MI,
                                                llvm::IRBuilderBase &Builder) {
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] raiseMachineInstr: " << MI);

  switch (MI.getOpcode()) {

#include "SIInstrSemantics.inc"

  case llvm::TargetOpcode::PATCHPOINT: {
    // Raise the PATCHPOINT injection marker into a distinct call to a
    // \c luthier.patchpoint function declaration. The call carries the
    // payload's extern handle as its first argument, followed by the
    // materialized values of the marker's implicit-use regs. Its return
    // type encodes the implicit-def regs: void if none, the reg's value
    // directly if there is exactly one, a literal struct of values if
    // there are two or more. Each site gets its own function decl —
    // LLVM auto-suffixes the shared "luthier.patchpoint" name — because
    // the signature is site-specific.
    llvm::Module *M = MF.getFunction().getParent();
    llvm::LLVMContext &Ctx = M->getContext();

    // PatchpointOpers::TargetPos = 2 holds the payload extern's
    // MO_GlobalAddress. Prototype::assignToInject built it as a Function
    // decl in the target module.
    auto *PayloadHandle = llvm::cast<llvm::Function>(
        const_cast<llvm::GlobalValue *>(MI.getOperand(2).getGlobal()));

    // Split the marker's implicit operands into use and def phys-regs.
    llvm::SmallVector<llvm::MCRegister, 4> UseRegs;
    llvm::SmallVector<llvm::MCRegister, 4> DefRegs;
    for (const llvm::MachineOperand &Op : MI.implicit_operands()) {
      if (!Op.isReg())
        continue;
      if (Op.isDef())
        DefRegs.push_back(Op.getReg().asMCReg());
      else
        UseRegs.push_back(Op.getReg().asMCReg());
    }

    // Argument list: {payload handle, use-reg values...}.
    llvm::SmallVector<llvm::Value *, 4> CallArgs;
    llvm::SmallVector<llvm::Type *, 4> ArgTypes;
    CallArgs.push_back(PayloadHandle);
    ArgTypes.push_back(PayloadHandle->getType());
    for (llvm::MCRegister R : UseRegs) {
      llvm::Value &V = getOperandAsValue(MI, R);
      CallArgs.push_back(&V);
      ArgTypes.push_back(V.getType());
    }

    // Return type: void / T / literal struct{T1, T2, ...}.
    llvm::SmallVector<llvm::Type *, 4> DefTypes;
    for (llvm::MCRegister R : DefRegs)
      DefTypes.push_back(llvm::IntegerType::get(Ctx, getPhysRegisterSize(R)));
    llvm::Type *RetTy = DefTypes.empty() ? llvm::Type::getVoidTy(Ctx)
                        : DefTypes.size() == 1
                            ? DefTypes.front()
                            : llvm::StructType::get(Ctx, DefTypes);

    auto *FnTy = llvm::FunctionType::get(RetTy, ArgTypes, /*isVarArg=*/false);
    auto *PatchpointFn = llvm::Function::Create(
        FnTy, llvm::GlobalValue::ExternalLinkage, "luthier.patchpoint", M);

    llvm::CallInst *Call = Builder.CreateCall(PatchpointFn, CallArgs);

    // Write back the implicit-def values into the register value map.
    if (DefRegs.size() == 1) {
      setRegOperandValue(MI, DefRegs.front(), Call);
    } else {
      for (unsigned I = 0, E = DefRegs.size(); I != E; ++I) {
        llvm::Value *V = Builder.CreateExtractValue(Call, {I});
        setRegOperandValue(MI, DefRegs[I], V);
      }
    }
    break;
  }

  default: {
    LLVM_DEBUG(luthier::dbgs()
               << "[TraceFunctionTranslator] Unmodelled instruction " << MI
               << "\n");

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

void TraceFunctionTranslator::translateMBBBody(llvm::MachineBasicBlock &MBB) {
  llvm::LLVMContext &Ctx = MF.getFunction().getContext();
  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Processing MBB " << MBB.getNumber()
             << " with " << MBB.size() << " instructions\n");
  auto *BB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock());
  for (llvm::MachineInstr &MI : MBB) {
    LLVM_DEBUG(luthier::dbgs() << "[TraceFunctionTranslator] Translating MI: ";
               MI.print(luthier::dbgs()););
    llvm::IRBuilder<llvm::InstSimplifyFolder, llvm::IRBuilderCallbackInserter>
        Builder(Ctx, llvm::InstSimplifyFolder{MF.getDataLayout()},
                llvm::IRBuilderCallbackInserter{[&](llvm::Instruction *I) {
                  if (MI.getPCSections())
                    I->setMetadata(llvm::LLVMContext::MD_pcsections,
                                   MI.getPCSections());
                  LLVM_DEBUG(
                      luthier::dbgs()
                      << "[TraceFunctionTranslator] Inserting translated "
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

llvm::Expected<bool> TraceFunctionTranslator::retranslateMBB(
    const llvm::MachineBasicBlock &ConstMBB,
    llvm::SmallVectorImpl<llvm::Instruction *> &PendingDeadInsts) {
  auto &MBB = const_cast<llvm::MachineBasicBlock &>(ConstMBB);
  auto *BodyBB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock());
  if (!BodyBB)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "MBB {0} has no translated IR block to re-translate", MBB.getNumber()));

  LLVM_DEBUG(luthier::dbgs() << "[TraceFunctionTranslator] Re-translating MBB "
                             << MBB.getNumber() << "\n");

  /// Snapshot the old boundary register-value state: every value the old
  /// body exported to other blocks is reachable from here by RegFileKey
  RegValueMap OldState;
  if (auto It = VM.find(BodyBB); It != VM.end()) {
    OldState = std::move(It->second);
    VM.erase(It);
  }

  /// Detach (but do not delete) the old body so external uses stay
  /// resolvable until they are replaced below
  llvm::SmallPtrSet<llvm::Instruction *, 32> OldInsts;
  llvm::SmallVector<llvm::Instruction *> OldInstList;
  while (!BodyBB->empty()) {
    llvm::Instruction &I = BodyBB->back();
    I.removeFromParent();
    OldInsts.insert(&I);
    OldInstList.push_back(&I);
  }

  /// Re-seed the entry state if this is the function's entry block
  if (&MBB == &MF.front()) {
    llvm::IRBuilder EntryBuilder{BodyBB};
    if (MF.getFunction().getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
      initKernelEntryRegs(EntryBuilder);
    else
      initDeviceFunctionEntryRegs(EntryBuilder);
  }

  translateMBBBody(MBB);

  /// Vector MBBs: boundary placeholder PHIs must live in the CheckBB, whose
  /// IR predecessors are the MIR predecessors' IR blocks (the BodyBB's only
  /// IR predecessors are the Check/Skip scaffolding)
  if (auto It = VectorCheckBBs.find(&MBB); It != VectorCheckBBs.end()) {
    llvm::BasicBlock *CheckBB = It->second;
    while (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&BodyBB->front()))
      Phi->moveBefore(*CheckBB, CheckBB->begin());
  }

  fixupPhis();

  /// Repair cross-block dataflow: re-materialize every register value the
  /// old body exported and replace the old value's remaining uses. Values
  /// that were not instructions of the old body (constants, arguments,
  /// values defined in other blocks) are still live and need no repair
  llvm::DenseMap<llvm::Value *, llvm::Value *> Replacements;
  for (auto &[Key, OldVTM] : OldState) {
    for (auto &[Ty, OldV] : OldVTM) {
      auto *OldInst = llvm::dyn_cast<llvm::Instruction>(OldV);
      if (!OldInst || !OldInsts.contains(OldInst) || OldInst->use_empty())
        continue;
      llvm::IRBuilder<llvm::InstSimplifyFolder> Builder(
          BodyBB->getContext(), llvm::InstSimplifyFolder{MF.getDataLayout()});
      Builder.SetInsertPoint(BodyBB->getTerminator());
      llvm::Value &NewV = getOperandAsValue(*BodyBB, Key, Builder, Ty);
      OldInst->replaceAllUsesWith(&NewV);
      Replacements[OldInst] = &NewV;
    }
  }
  /// Boundary PHIs created by the repair itself may need resolving
  fixupPhis();

  /// Other blocks' register caches may still point at old body values;
  /// remap repaired entries and drop unrepaired ones so later reads
  /// re-materialize them
  for (auto &[OtherBB, State] : VM) {
    if (OtherBB == BodyBB)
      continue;
    for (auto &[Key, VTM] : State)
      for (auto It = VTM.begin(); It != VTM.end();)
        if (auto *I = llvm::dyn_cast<llvm::Instruction>(It->second);
            I && OldInsts.contains(I)) {
          auto RIt = Replacements.find(I);
          if (RIt != Replacements.end())
            (It++)->second = RIt->second;
          else
            VTM.erase(It++);
        } else
          ++It;
  }

  /// Drop the detached old body for good. An old instruction that still has
  /// uses is an intermediate value that escaped this block without a
  /// register-file key (e.g. forwarded by single-incoming PHI collapse); its
  /// role is no longer recoverable, so in-place repair is impossible and the
  /// caller must fall back to a full re-translation. The orphans are parked
  /// in \c PendingDeadInsts until the full translate drops their users
  bool NeedFullRetranslate = false;
  for (llvm::Instruction *I : OldInstList)
    I->dropAllReferences();
  for (llvm::Instruction *I : OldInstList) {
    if (I->use_empty()) {
      I->deleteValue();
    } else {
      LLVM_DEBUG(
          luthier::dbgs()
              << "[TraceFunctionTranslator] Unkeyed escaped value; falling "
                 "back to full re-translation: "
              << *I << "\n";);
      PendingDeadInsts.push_back(I);
      NeedFullRetranslate = true;
    }
  }
  return NeedFullRetranslate;
}

bool TraceFunctionTranslator::irSuccessorsMatchMIR(
    const llvm::MachineBasicBlock &MBB) const {
  const llvm::BasicBlock *BodyBB = MBB.getBasicBlock();
  if (!BodyBB || !BodyBB->getTerminator())
    return false;
  /// Each MIR successor is entered through its BodyBB, or through its
  /// CheckBB when it is a vector block. The match must hold in both
  /// directions: an IR edge to a block outside the translated successors
  /// means an edge was removed, and a translated successor missing from the
  /// IR terminator means one was added
  llvm::SmallPtrSet<const llvm::BasicBlock *, 8> IRSuccs{
      llvm::succ_begin(BodyBB), llvm::succ_end(BodyBB)};
  llvm::SmallPtrSet<const llvm::BasicBlock *, 8> Allowed;
  for (const llvm::MachineBasicBlock *Succ : MBB.successors()) {
    const llvm::BasicBlock *Entry = Succ->getBasicBlock();
    if (auto It = VectorCheckBBs.find(Succ); It != VectorCheckBBs.end())
      Entry = It->second;
    if (!Entry || !IRSuccs.contains(Entry))
      return false;
    Allowed.insert(Entry);
  }
  return llvm::all_of(
      IRSuccs, [&](const llvm::BasicBlock *S) { return Allowed.contains(S); });
}

void TraceFunctionTranslator::translate() {
  auto &F = const_cast<llvm::Function &>(MF.getFunction());
  llvm::LLVMContext &Ctx = F.getContext();
  /// Early exit if there are no basic blocks in the machine function
  if (MF.empty())
    return;

  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Translating machine function '"
             << MF.getName() << "' with " << MF.size() << " MBBs\n");

  /// Delete any basic blocks already present in the IR Function. References
  /// are dropped up front so cross-block uses don't trip the use-after-def
  /// check while the blocks are erased one by one
  if (!F.empty()) {
    for (llvm::BasicBlock &BB : F)
      BB.dropAllReferences();
    (void)F.erase(F.begin(), F.end());
  }

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
  for (llvm::MachineBasicBlock &MBB : MF)
    translateMBBBody(MBB);

  /// Insert an EXEC-mask predicate check before every vector MBB's
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

    /// True CFG diamond for VALU MBBs:
    ///
    ///                       ┌── BodyBB (raised body + terminator to
    ///                       │           next MBB)
    ///   pred ─→ CheckBB ────┤        │
    ///                       └── SkipBB (br → same single-successor
    ///                                    target as BodyBB's terminator;
    ///                                    carries pre-body values around
    ///                                    the body via VM[SkipBB])
    ///                                │
    ///                                ▼
    ///                             (next MBB — two IR predecessors per
    ///                              one MIR pred; fixupPhis walks IR
    ///                              predecessors and materializes
    ///                              incomings via VM[SkipBB] /
    ///                              VM[BodyBB])
    ///
    /// The diamond expresses EXEC-inactive lane preservation as CFG
    /// structure: values consumed downstream that were written by the
    /// vector MBB come from BodyBB's exit state, while values that
    /// pre-existed the vector MBB are carried through SkipBB. Only the
    /// single-successor case is diamonded — multi-successor bodies fall
    /// back to SkipBB → BodyBB pass-through so we don't have to
    /// duplicate the branch condition.
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
    /// The condBr in CheckBB and the br in SkipBB below will recreate
    /// the CheckBB→BodyBB and SkipBB→BodyBB (or SkipBB→next) edges.
    BodyBB->replaceUsesWithIf(CheckBB, [&](llvm::Use &U) {
      auto *I = llvm::dyn_cast<llvm::Instruction>(U.getUser());
      return I && I->getParent() != CheckBB && I->getParent() != SkipBB;
    });

    /// Hoist any placeholder PHI nodes from BodyBB to CheckBB. After the
    /// redirect, CheckBB's IR predecessors are exactly the vector MBB's
    /// MIR predecessor IR BBs
    while (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&BodyBB->front())) {
      Phi->moveBefore(*CheckBB, CheckBB->begin());
      for (auto &TBF : ToBeFixedPhis)
        if (TBF.Phi == Phi)
          TBF.BB = CheckBB;
    }

    /// Initialize the CheckBB and SkipBB's VMs
    (void)VM.try_emplace(CheckBB);
    (void)VM.try_emplace(SkipBB);

    emitExecPredicateCheck(CheckBB, BodyBB, SkipBB);

    /// SkipBB's target: the block BodyBB's raised terminator goes to.
    /// For a single-successor body we point SkipBB straight at that
    /// successor, giving the downstream MBB two IR predecessors
    /// (BodyBB and SkipBB) — fixupPhis handles the phi merge naturally
    /// via IR-CFG traversal.
    llvm::IRBuilder<> SkipBuilder(SkipBB);
    llvm::Instruction *BodyTerm = BodyBB->getTerminator();
    if (BodyTerm && BodyTerm->getNumSuccessors() == 1) {
      SkipBuilder.CreateBr(BodyTerm->getSuccessor(0));
    } else {
      SkipBuilder.CreateBr(BodyBB);
    }
  }

  /// Snapshot the per-BB register-file state into a \c WeakTrackingVH
  /// shadow before \c fixupPhis, \c foldHwregIntrinsics, or
  /// \c optimizeNonTraceInsts can rewrite or erase any of the tracked
  /// Values. \c WeakTrackingVH follows RAUW and nulls on erase, so the
  /// shadow stays coherent through the remaining pipeline.
  snapshotBBExitStates();

  /// Fixup all dangeling PHIs
  fixupPhis();

  /// Rewrite @llvm.amdgcn.s.{get,set}reg with a constant hwreg encoding
  /// naming a tracked register (MODE today) into direct read/write of
  /// the tracked SSA value with explicit bitfield ops, so the kernel-
  /// entry MODE constant flows through to the optimizer.
  foldHwregIntrinsics();

  /// First pass over CheckBBs: catches the entry-MBB constant-EXEC case
  /// (where \c ExecVal is a compile-time \c -1) so \c optimizeNonTraceInsts
  /// below can DCE the dead chain in one sweep.
  foldTriviallyActiveExecChecks();

  /// Final cleanup: simplify and remove dead non-trace IR. Trace
  /// instructions (those whose pcsections carry a trace instruction
  /// address) are preserved verbatim. \c LuthierAMDGPUFolder folds
  /// AMDGPU intrinsic calls with foldable constant operands (used
  /// through the folder-backed IRBuilders and by a targeted pass at
  /// this function's worklist init); the pre-worklist pass here also
  /// applies the same fold to existing intrinsic instructions before
  /// \c simplifyInstruction runs.
  optimizeNonTraceInsts();

  /// Second CheckBB fold pass: InstSimplify's PHI-collapse just above
  /// may have exposed \c ExecVal as \c -1 in blocks whose EXEC arrived
  /// as an unresolved PHI at emission. Any constant-true CondBr is
  /// rewritten to an unconditional branch, SkipBB is dropped, and the
  /// mbcnt / lshr / and / trunc chain (dead-with-no-uses now) is DCE'd
  /// locally.
  foldTriviallyActiveExecChecks();

  /// Serialize the per-BB exit register-file state as function-level
  /// \c luthier.bb_exit_reg_map metadata so downstream passes can trace
  /// which SSA value represents each register slice at BB boundaries.
  emitBBExitRegMapMetadata();

  LLVM_DEBUG(luthier::dbgs()
             << "[TraceFunctionTranslator] Translation complete for '"
             << F.getName() << "': " << F.size() << " basic blocks\n");
}

void TraceFunctionTranslator::emitExecPredicateCheck(llvm::BasicBlock *CheckBB,
                                                     llvm::BasicBlock *BodyBB,
                                                     llvm::BasicBlock *SkipBB) {
  llvm::IRBuilder<LuthierAMDGPUFolder, llvm::IRBuilderCallbackInserter>
      Builder(CheckBB->getContext(),
              LuthierAMDGPUFolder{MF.getDataLayout()},
              llvm::IRBuilderCallbackInserter{[](llvm::Instruction *I) {
                LLVM_DEBUG(
                    luthier::dbgs()
                    << "[TraceFunctionTranslator] Inserting exec predicate "
                       "instruction "
                    << *I << "\n");
              }});
  Builder.SetInsertPoint(CheckBB);

  /// EXEC value at the entry of CheckBB. If \c CheckBB has IR
  /// predecessors, \c getOperandAsValue materializes an entry PHI at
  /// \c CheckBB.begin whose incomings \c fixupPhis resolves from each
  /// predecessor's exit state. If \c CheckBB has no IR predecessors it
  /// is the function's entry block (MBB has no MIR preds); the entry
  /// EXEC is set by \c initKernelEntryRegs to all-ones for kernels, or
  /// to a preload argument value for device functions. That seed was
  /// written into \c VM[BodyBB] before body translation ran (BodyBB is
  /// the raised body of MBB.front(), which under this diamond code
  /// path is the MBB whose scaffold we are emitting). We materialize
  /// the seed directly into CheckBB here rather than going through the
  /// placeholder-PHI path, since a PHI with zero incomings would
  /// collapse to \c freeze(poison) and mask the entry EXEC.
  llvm::MCRegister ExecReg = TRI.getExec();
  unsigned ExecWidth = TRI.getRegSizeInBits(ExecReg, MF.getRegInfo());
  llvm::Type *ExecTy = Builder.getIntNTy(ExecWidth);
  llvm::Value *ExecVal;
  if (llvm::pred_empty(CheckBB)) {
    // Kernel entry: EXEC is all-ones. Device-function entry: the entry
    // EXEC comes via the calling convention and \c initKernelEntryRegs /
    // \c initDeviceFunctionEntryRegs already wrote it into \c
    // VM[BodyBB]. Reuse that value from \c BodyBB's tracker if it's a
    // constant (kernel entry always) or a function-argument value
    // (device function entry); those are safe to reference from
    // CheckBB, which dominates BodyBB.
    if (auto It = VM.find(BodyBB); It != VM.end()) {
      auto SubIt = It->second.find(getRegFileKey(ExecReg));
      if (SubIt != It->second.end()) {
        if (auto TIt = SubIt->second.find(ExecTy);
            TIt != SubIt->second.end())
          ExecVal = TIt->second;
        else
          ExecVal = Builder.getInt(llvm::APInt::getAllOnes(ExecWidth));
      } else {
        ExecVal = Builder.getInt(llvm::APInt::getAllOnes(ExecWidth));
      }
    } else {
      ExecVal = Builder.getInt(llvm::APInt::getAllOnes(ExecWidth));
    }
  } else {
    ExecVal =
        &getOperandAsValue(*CheckBB, getRegFileKey(ExecReg), Builder, ExecTy);
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

  /// Populate this CheckBB's slot in \c ExitStateShadow directly from
  /// the placeholder PHIs we anchored on this CheckBB. Every such PHI
  /// lives in \c CheckBB (either hoisted from BodyBB before this call
  /// or just created above for EXEC) and represents both the entry-
  /// and exit-state of CheckBB for its register slice, since CheckBB
  /// does not mutate any tracked physical register. \c SkipBB is
  /// intentionally never populated in the shadow.
  auto &CheckShadow = ExitStateShadow[CheckBB];
  for (const auto &TBF : ToBeFixedPhis) {
    if (!TBF.Phi || TBF.Phi->getParent() != CheckBB)
      continue;
    CheckShadow[TBF.RegKey].emplace_back(TBF.Phi->getType(),
                                         llvm::WeakTrackingVH(TBF.Phi));
  }
}

void TraceFunctionTranslator::foldTriviallyActiveExecChecks() {
  using namespace llvm::PatternMatch;
  /// Snapshot the entries up-front so the per-CheckBB cleanup below can
  /// erase from \c VectorCheckBBs (and delete the CheckBB itself, which
  /// invalidates the map's key hash otherwise) without invalidating our
  /// iteration.
  llvm::SmallVector<std::pair<const llvm::MachineBasicBlock *,
                              llvm::BasicBlock *>,
                    16>
      Snapshot(VectorCheckBBs.begin(), VectorCheckBBs.end());
  for (const auto &KV : Snapshot) {
    llvm::BasicBlock *CheckBB = KV.second;
    if (!CheckBB || CheckBB->empty())
      continue;

    auto *CondBr = llvm::dyn_cast<llvm::CondBrInst>(CheckBB->getTerminator());
    if (!CondBr)
      continue;

    /// Match the exact chain emitted by \c emitExecPredicateCheck:
    ///   trunc(and(lshr(<ExecVal>, <LaneId>), 1), i1)
    /// Also accept a trivially-true constant condition, which is what
    /// upstream \c optimizeNonTraceInsts constant-propagation may have
    /// already reduced the chain to.
    llvm::Value *ExecVal = nullptr;
    if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(CondBr->getCondition())) {
      if (!CI->isOne())
        continue;
    } else if (!match(CondBr->getCondition(),
                      m_Trunc(m_And(m_LShr(m_Value(ExecVal), m_Value()),
                                    m_SpecificInt(1)))) ||
               !isProvablyAllOnesInt(ExecVal)) {
      continue;
    }

    llvm::BasicBlock *BodyBB = CondBr->getSuccessor(0);
    llvm::BasicBlock *SkipBB = CondBr->getSuccessor(1);

    /// Replace the conditional branch with an unconditional branch to
    /// BodyBB and DCE the dead trunc / and / lshr / mbcnt chain rooted
    /// at the former condition. Local DCE is required because this fold
    /// can run after \c optimizeNonTraceInsts (when upstream
    /// constant-propagation has revealed a foldable chain), so no later
    /// sweep is guaranteed to clean up.
    llvm::Value *DeadRoot = CondBr->getCondition();
    llvm::IRBuilder<> B(CondBr);
    B.CreateBr(BodyBB);
    CondBr->eraseFromParent();
    if (auto *DeadI = llvm::dyn_cast<llvm::Instruction>(DeadRoot)) {
      llvm::SmallVector<llvm::WeakTrackingVH, 8> DCE;
      DCE.emplace_back(DeadI);
      while (!DCE.empty()) {
        auto *I =
            llvm::dyn_cast_or_null<llvm::Instruction>(DCE.pop_back_val());
        if (!I || !llvm::isInstructionTriviallyDead(I))
          continue;
        for (llvm::Use &Op : I->operands())
          if (auto *OI = llvm::dyn_cast<llvm::Instruction>(Op.get()))
            DCE.emplace_back(OI);
        I->eraseFromParent();
      }
    }

    /// SkipBB was reachable only from this CheckBB. Drop the predecessor
    /// edge from any successors so their PHIs get patched up, then erase
    /// SkipBB itself if it's now unreferenced. SkipBBs carry no metadata
    /// (per the exit-reg-map design) so they're safe to delete outright.
    for (llvm::BasicBlock *Succ : llvm::successors(SkipBB))
      Succ->removePredecessor(SkipBB);
    if (SkipBB->use_empty()) {
      SkipBB->dropAllReferences();
      ExecScaffoldBBs.erase(SkipBB);
      VM.erase(SkipBB);
      SkipBB->eraseFromParent();
    }

    /// If CheckBB is now a pure pass-through (just the unconditional
    /// branch to BodyBB, no hoisted register-value PHIs left), merge it
    /// with its BodyBB successor
    if (CheckBB->size() == 1 &&
        CheckBB != &CheckBB->getParent()->getEntryBlock()) {
      ExecScaffoldBBs.erase(CheckBB);
      ExitStateShadow.erase(CheckBB);
      VectorCheckBBs.erase(KV.first);
      VM.erase(CheckBB);
      llvm::TryToSimplifyUncondBranchFromEmptyBlock(CheckBB);
    }
  }
}

bool TraceFunctionTranslator::shouldEmitGPRIndexAccess(
    const llvm::MachineInstr &MI, llvm::MCRegister Reg) const {
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

llvm::Value &TraceFunctionTranslator::emitIndexedVGPRSrc(
    const llvm::MachineInstr &MI, llvm::MCRegister Reg, llvm::Type *OutType) {
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
  llvm::Value *Mode = &getOperandAsValue(*BB, llvm::AMDGPU::MODE, I32);
  llvm::Value *EnBit = Builder.CreateAnd(
      Builder.CreateLShr(Mode, llvm::ConstantInt::get(I32, 27)),
      llvm::ConstantInt::get(I32, 1));
  llvm::Value *En = Builder.CreateTrunc(EnBit, Builder.getInt1Ty());

  llvm::Value *M0 = &getOperandAsValue(*BB, llvm::AMDGPU::M0, I32);
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
  llvm::Value *Acc = &getOperandAsValue(*BB, Reg, OutType);
  llvm::Value *AccI32 = Acc;
  if (AccI32->getType() != I32)
    AccI32 = Builder.CreateBitOrPointerCast(AccI32, I32);

  unsigned NumSlots = (TotalHalves - BaseHalves) / 2;
  for (unsigned k = 1; k < NumSlots; ++k) {
    RegFileKey SlotKey = std::make_tuple(BaseReg, BaseHalves + 2 * k, 2u);
    llvm::Value *Slot = &getOperandAsValue(*BB, SlotKey, Builder, I32);
    llvm::Value *KEq =
        Builder.CreateICmpEQ(Idx, llvm::ConstantInt::get(I32, k));
    llvm::Value *Pick = Builder.CreateAnd(En, KEq);
    AccI32 = Builder.CreateSelect(Pick, Slot, AccI32);
  }
  if (AccI32->getType() != OutType)
    AccI32 = Builder.CreateBitOrPointerCast(AccI32, OutType);
  return *AccI32;
}

void TraceFunctionTranslator::emitIndexedVGPRDst(const llvm::MachineInstr &MI,
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
  llvm::Value *Mode = &getOperandAsValue(*BB, llvm::AMDGPU::MODE, I32);
  llvm::Value *EnBit = Builder.CreateAnd(
      Builder.CreateLShr(Mode, llvm::ConstantInt::get(I32, 27)),
      llvm::ConstantInt::get(I32, 1));
  llvm::Value *En = Builder.CreateTrunc(EnBit, Builder.getInt1Ty());

  llvm::Value *M0 = &getOperandAsValue(*BB, llvm::AMDGPU::M0, I32);
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
    llvm::Value *OldVal = &getOperandAsValue(*BB, SlotKey, Builder, I32);
    llvm::Value *Cond =
        Builder.CreateICmpEQ(FinalIdx, llvm::ConstantInt::get(I32, k));
    llvm::Value *NewVal = Builder.CreateSelect(Cond, ValI32, OldVal);
    setRegOperandValue(*BB, SlotKey, Builder, NewVal);
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

void TraceFunctionTranslator::foldHwregIntrinsics() {
  auto &F = const_cast<llvm::Function &>(MF.getFunction());

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
        *BB, Key, Builder, llvm::Type::getInt32Ty(F.getContext()));

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
      setRegOperandValue(*BB, Key, Builder, NewReg);
      Call->eraseFromParent();
    }
  }
}

void TraceFunctionTranslator::optimizeNonTraceInsts() {
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
    for (llvm::Instruction &I : BB) {
      if (IsTrace(&I))
        continue;
      /// Apply AMDGPU-specific intrinsic folds up-front that the optimizations
      /// won't be able to do themselves
      if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (llvm::Function *Callee = Call->getCalledFunction()) {
          llvm::Intrinsic::ID ID = Callee->getIntrinsicID();
          if (ID != llvm::Intrinsic::not_intrinsic) {
            llvm::SmallVector<llvm::Value *, 4> Args(
                Call->arg_begin(), Call->arg_end());
            if (llvm::Value *V = tryFoldAMDGPUIntrinsic(ID, Args)) {
              for (llvm::User *U : I.users())
                if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U))
                  if (!IsTrace(UI))
                    Worklist.emplace_back(UI);
              I.replaceAllUsesWith(V);
            }
          }
        }
      }
      Worklist.emplace_back(&I);
    }
  }

  while (!Worklist.empty()) {
    llvm::WeakTrackingVH WH = Worklist.pop_back_val();
    auto *I = llvm::dyn_cast_or_null<llvm::Instruction>(WH);
    if (!I)
      continue;
    if (IsTrace(I))
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

void TraceFunctionTranslator::snapshotBBExitStates() {
  /// Snapshot every BB's VM (post-diamond BodyBB, CheckBB, SkipBB). The
  /// CheckBB entries under the new per-BB VM are populated on demand by
  /// \c emitExecPredicateCheck and subsequent \c getOperandAsValue
  /// calls; SkipBB entries stay empty (SkipBB never mutates state).
  for (const auto &[BB, State] : VM) {
    if (!BB)
      continue;
    auto &Dst = ExitStateShadow[BB];
    for (const auto &KV : State) {
      auto &Slot = Dst[KV.first];
      Slot.reserve(KV.second.size());
      for (const auto &TV : KV.second) {
        if (!TV.second)
          continue;
        Slot.emplace_back(TV.first, llvm::WeakTrackingVH(TV.second));
      }
    }
  }
}

void TraceFunctionTranslator::emitBBExitRegMapMetadata() {
  auto &F = const_cast<llvm::Function &>(MF.getFunction());
  if (F.empty())
    return;

  llvm::SmallVector<llvm::MDNode *, 16> PerBBEntries;
  PerBBEntries.reserve(F.size());

  for (llvm::BasicBlock &BB : F) {
    auto ShadowIt = ExitStateShadow.find(&BB);
    if (ShadowIt == ExitStateShadow.end())
      continue; // SkipBBs and any BB without a shadow entry get no MD.

    /// \c Names owns the display strings; \c Slices holds \c StringRef
    /// views into \c Names, so \c Names must not reallocate during the
    /// build loop below. Pre-count slices and \c reserve up-front so the
    /// vector's storage address stays stable.
    size_t Total = 0;
    for (const auto &KV : ShadowIt->second)
      Total += KV.second.size();
    llvm::SmallVector<std::string, 8> Names;
    llvm::SmallVector<BBExitRegSlice, 8> Slices;
    Names.reserve(Total);
    Slices.reserve(Total);

    for (const auto &KV : ShadowIt->second) {
      const RegFileKey &Key = KV.first;
      RegValueDesc Desc{std::get<0>(Key), std::get<1>(Key), std::get<2>(Key)};
      for (const auto &TV : KV.second) {
        llvm::Value *V = TV.second;
        if (!V)
          continue;
        Names.emplace_back(
            formatRegValueDescName(Desc, TRI.getName(Desc.BaseReg)));
        Slices.push_back({V, Desc, llvm::StringRef(Names.back())});
      }
    }

    PerBBEntries.emplace_back(buildBBExitRegMapEntry(F, &BB, Slices));
  }

  setBBExitRegMap(F, PerBBEntries);
}

} // namespace luthier
