#include "luthier/Tooling/MIRToIRTranslator.h"
#include "luthier/Common/DenseMapInfo.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Tooling/TargetMachineInstrMDNode.h"
#include <AMDGPUMachineFunction.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Support/Error.h>

namespace {
/// Friend ADL trick to allow access to the private basic block field of
/// machine basic block
/// Unlike what LLVM assumes (IR comes after MIR), we have to construct the
/// IR basic block after we have the machine basic block
struct TagBB {
  using type = const llvm::BasicBlock *llvm::MachineBasicBlock::*;

  friend type get(TagBB);
};

template <typename Tag, typename Tag::type MemPtr> struct Access {
  friend typename Tag::type get(Tag) { return MemPtr; }
};

template struct Access<TagBB, &llvm::MachineBasicBlock::BB>;
} // namespace

namespace luthier {

//===----------------------------------------------------------------------===//
// MBBOperandTracker — lazy register get / set
//===----------------------------------------------------------------------===//

/// Bitcast a scalar integer value to a vector of \p ElemWidth-bit integers,
/// suitable for extractelement / insertelement indexing.
static llvm::Value *scalarToVec(llvm::Value *Scalar, unsigned ElemWidth,
                                llvm::IRBuilder<> &Builder,
                                const llvm::Twine &Name = "") {
  unsigned TotalWidth = Scalar->getType()->getIntegerBitWidth();
  assert(TotalWidth % ElemWidth == 0);
  unsigned NumElems = TotalWidth / ElemWidth;
  auto *VecTy =
      llvm::FixedVectorType::get(Builder.getIntNTy(ElemWidth), NumElems);
  return Builder.CreateBitCast(Scalar, VecTy, Name);
}

/// Bitcast a vector value back to a scalar integer.
static llvm::Value *vecToScalar(llvm::Value *Vec, unsigned TotalWidth,
                                llvm::IRBuilder<> &Builder,
                                const llvm::Twine &Name = "") {
  return Builder.CreateBitCast(Vec, Builder.getIntNTy(TotalWidth), Name);
}

static llvm::SmallVector<llvm::MCRegister>
getOverlappingRegUnits(const llvm::TargetRegisterInfo &TRI,
                       llvm::MCRegister RegA, llvm::MCRegister RegB);

/// Given two registers \p RegA and \p RegB finds the common
/// \c llvm::MCRegister that is common between both registers. Returns the
/// empty \c llvm::MCRegister if no common sub register is found
static llvm::MCRegister
getOverlappingSubReg(const llvm::TargetRegisterInfo &TRI, llvm::MCRegister RegA,
                     llvm::MCRegister RegB) {
  llvm::SmallVector<llvm::MCRegister> OverlappingRegUnits =
      getOverlappingRegUnits(TRI, RegA, RegB);

  switch (OverlappingRegUnits.size()) {
  case 0:
    return {};
  case 1:
    return OverlappingRegUnits[0];
  default: {
    llvm::MCRegister FirstReg = OverlappingRegUnits[0];
    auto OtherOverlappingRegs =
        llvm::ArrayRef<llvm::MCRegister>{OverlappingRegUnits}.drop_front();
    llvm::MCRegister Out{};
    for (auto FirstSupReg = llvm::MCSuperRegIterator(FirstReg, &TRI);
         FirstSupReg.isValid(); ++FirstSupReg) {
      bool DoAllOverlap{true};
      for (llvm::MCRegister OtherRegUnit : OtherOverlappingRegs) {
        DoAllOverlap |= TRI.regsOverlap(*FirstSupReg, OtherRegUnit);
      }

      if (DoAllOverlap &&
          (Out == 0 || TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(Out)) >
                           TRI.getRegSizeInBits(
                               *TRI.getMinimalPhysRegClass(*FirstSupReg)))) {
        Out = *FirstSupReg;
      }
    }
    return Out;
  }
  }
}

llvm::SmallVector<llvm::MCRegister>
MBBOperandTracker::getOverlappingRegUnits(const llvm::TargetRegisterInfo &TRI,
                                          llvm::MCRegister RegA,
                                          llvm::MCRegister RegB) {
  /// This is similar to regsOverlap method in TRI; It's modified to
  /// obtain all common reg units
  auto RangeA = TRI.regunits(RegA);
  llvm::MCRegUnitIterator IA = RangeA.begin(), EA = RangeA.end();
  auto RangeB = TRI.regunits(RegB);
  llvm::MCRegUnitIterator IB = RangeB.begin(), EB = RangeB.end();
  llvm::SmallVector<llvm::MCRegister> OverlappingRegUnits{};
  do {
    if (*IA == *IB) {
      for (llvm::MCRegUnitRootIterator RI(*IA, &TRI); RI.isValid(); ++RI)
        OverlappingRegUnits.push_back(*RI);
    }
  } while (*IA < *IB ? ++IA != EA : ++IB != EB);
  return OverlappingRegUnits;
}

void MBBOperandTracker::invalidateOverlaps(MCRegValueMap &Map,
                                           llvm::MCRegister Reg,
                                           llvm::IRBuilder<> &Builder) {
  const llvm::SIRegisterInfo &TRI = getTRI();

  struct Preserve {
    llvm::MCRegister SubReg;
    llvm::Value *Val;
  };
  llvm::SmallVector<llvm::MCRegister, 8> ToErase;
  llvm::SmallVector<Preserve, 4> ToPreserve;

  for (auto &[StoredReg, Entry] : Map) {
    if (StoredReg == Reg)
      continue;
    if (!TRI.regsOverlap(StoredReg, Reg))
      continue;

    // Case 1: StoredReg ⊂ Reg — fully covered, just erase.
    if (unsigned SubIdx = TRI.getSubRegIndex(Reg, StoredReg); SubIdx != 0) {
      ToErase.push_back(StoredReg);
      continue;
    }

    // Case 2: Reg ⊂ StoredReg — partial overwrite of a super-register.
    // Bitcast super-reg value to a vector of sub-reg-sized elements and
    // extract the non-overlapping parts.
    if (unsigned SubIdx = TRI.getSubRegIndex(StoredReg, Reg); SubIdx != 0) {
      for (llvm::MCSubRegIndexIterator SRII(StoredReg, &TRI); SRII.isValid();
           ++SRII) {
        llvm::MCRegister Sub = SRII.getSubReg();
        unsigned SubSubIdx = SRII.getSubRegIndex();
        if (TRI.regsOverlap(Sub, Reg))
          continue; // Being overwritten.

        const llvm::TargetRegisterClass *SubRC =
            TRI.getMinimalPhysRegClass(Sub);
        unsigned SubSize = TRI.getRegSizeInBits(*SubRC);
        unsigned Offset = TRI.getSubRegIdxOffset(SubSubIdx);
        unsigned ElemIdx = Offset / SubSize;

        llvm::Value *Vec =
            scalarToVec(Entry, SubSize, Builder,
                        llvm::formatv("{0}.vec", TRI.getName(StoredReg)));
        llvm::Value *Extracted =
            Builder.CreateExtractElement(Vec, ElemIdx, TRI.getName(Sub));
        ToPreserve.push_back({Sub, Extracted});
      }
      ToErase.push_back(StoredReg);
      continue;
    }

    // Case 3: Partial overlap, neither sub nor super - recursively erase
    // overlapping sub-regs
    invalidateOverlaps(Map, getOverlappingSubReg(TRI, StoredReg, Reg), Builder);
  }

  for (llvm::MCRegister R : ToErase)
    Map.erase(R);
  for (const Preserve &P : ToPreserve)
    Map[P.SubReg] = P.Val;
}

llvm::Value *MBBOperandTracker::tryExtractFromSuperReg(
    MCRegValueMap &Map, llvm::MCRegister Reg, llvm::IRBuilder<> &Builder) {
  const llvm::SIRegisterInfo &TRI = getTRI();
  unsigned ReqSize = TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(Reg));

  for (auto &[StoredReg, Entry] : Map) {
    unsigned StoredSize =
        TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(StoredReg));
    if (StoredSize <= ReqSize)
      continue;
    unsigned SubIdx = TRI.getSubRegIndex(StoredReg, Reg);
    if (SubIdx == 0)
      continue;
    unsigned Offset = TRI.getSubRegIdxOffset(SubIdx);
    unsigned ElemIdx = Offset / ReqSize;

    llvm::Value *Vec =
        scalarToVec(Entry, ReqSize, Builder,
                    llvm::formatv("{0}.vec", TRI.getName(StoredReg)));
    return Builder.CreateExtractElement(Vec, ElemIdx, TRI.getName(Reg));
  }
  return nullptr;
}

llvm::Value *MBBOperandTracker::tryComposeFromSubRegs(
    MCRegValueMap &Map, llvm::MCRegister Reg, llvm::IRBuilder<> &Builder) {
  const llvm::SIRegisterInfo &TRI = getTRI();
  unsigned ReqSize = TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(Reg));

  struct SubPart {
    unsigned ElemIdx;
    unsigned Width;
    llvm::Value *Val;
  };
  llvm::SmallVector<SubPart, 8> Parts;
  unsigned CoveredBits = 0;

  for (auto &[StoredReg, Entry] : Map) {
    unsigned StoredSize =
        TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(StoredReg));
    if (StoredSize >= ReqSize)
      continue;
    unsigned SubIdx = TRI.getSubRegIndex(Reg, StoredReg);
    if (SubIdx == 0)
      continue;
    unsigned Offset = TRI.getSubRegIdxOffset(SubIdx);
    unsigned ElemIdx = Offset / StoredSize;
    Parts.push_back({ElemIdx, StoredSize, Entry});
    CoveredBits += StoredSize;
  }

  if (CoveredBits < ReqSize)
    return nullptr;

  // All parts must be the same width for uniform vector element type.
  unsigned ElemWidth = Parts[0].Width;
  unsigned NumElems = ReqSize / ElemWidth;
  auto *VecTy =
      llvm::FixedVectorType::get(Builder.getIntNTy(ElemWidth), NumElems);

  llvm::Value *Vec = llvm::PoisonValue::get(VecTy);
  for (const SubPart &P : Parts)
    Vec = Builder.CreateInsertElement(Vec, P.Val, P.ElemIdx);

  return vecToScalar(Vec, ReqSize, Builder, TRI.getName(Reg));
}

llvm::Value *MBBOperandTracker::tryComposeFromOverlappingRegs(
    const llvm::MachineBasicBlock &MBB, MCRegValueMap &Map,
    llvm::MCRegister Reg, llvm::IRBuilder<> &Builder) {
  const llvm::SIRegisterInfo &TRI = getTRI();

  llvm::SmallVector<llvm::Value *> SubRegVals{};

  for (llvm::MCRegUnit IA : TRI.regunits(Reg)) {
    for (llvm::MCRegUnitRootIterator RI(IA, &TRI); RI.isValid(); ++RI) {
      SubRegVals.push_back(&materializeReg(MBB, *RI));
    }
  }
  /// Registers should have been materialized by now; Simply ask for it again
  return &materializeReg(MBB, Reg);
}

void MBBOperandTracker::annotateUniformIfNeeded(llvm::Value *V,
                                                llvm::MCRegister Reg) {
  if (auto *I = llvm::dyn_cast<llvm::Instruction>(V)) {
    if (!getTRI().isVGPRPhysReg(Reg))
      I->setMetadata("amdgpu.uniform", llvm::MDNode::get(I->getContext(), {}));
  }
}

llvm::Value &
MBBOperandTracker::materializeReg(const llvm::MachineBasicBlock &MBB,
                                  llvm::MCRegister Reg) {
  MCRegValueMap &Map = getMap(MBB);
  const llvm::SIRegisterInfo &TRI = getTRI();
  const llvm::TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
  assert(RC && "No register class for Reg");
  unsigned ReqSize = TRI.getRegSizeInBits(*RC);
  auto *BB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock());
  assert(BB && "MBB does not have an IR basic block");
  llvm::IRBuilder Builder{BB};

  // 1. Exact match — the common / fast path.
  auto ExactIt = Map.find(Reg);
  if (ExactIt != Map.end())
    return *ExactIt->second;

  // 2. Try to extract from a stored super-register.
  if (llvm::Value *V = tryExtractFromSuperReg(Map, Reg, Builder)) {
    annotateUniformIfNeeded(V, Reg);
    Map[Reg] = V;
    return *V;
  }

  // 3. Try to compose from stored sub-registers.
  if (llvm::Value *V = tryComposeFromSubRegs(Map, Reg, Builder)) {
    annotateUniformIfNeeded(V, Reg);
    Map[Reg] = V;
    return *V;
  }

  // 4. Try to see if there are multiple overlapping registers that contain
  // the requested register that are not completely sub or super regs

  // 5. Not available locally — search predecessors and emit PHI / undef.
  llvm::IntegerType *RegTy = Builder.getIntNTy(ReqSize);
  llvm::StringRef RegName = TRI.getName(Reg);

  if (MBB.pred_empty()) {
    llvm::Value *Undef = llvm::UndefValue::get(RegTy);
    Map[Reg] = Undef;
    return *Undef;
  }

  llvm::SmallVector<std::pair<llvm::Value *, llvm::BasicBlock *>> PhiVals;
  PhiVals.reserve(MBB.pred_size());
  for (llvm::MachineBasicBlock *Pred : MBB.predecessors()) {
    llvm::Value &PredVal = materializeReg(*Pred, Reg);
    auto *PredBB = const_cast<llvm::BasicBlock *>(Pred->getBasicBlock());
    assert(PredBB && "Predecessor MBB has no IR basic block");
    PhiVals.push_back({&PredVal, PredBB});
  }

  llvm::PHINode *Phi = Builder.CreatePHI(RegTy, PhiVals.size(), RegName);
  for (const auto &[V, B] : PhiVals)
    Phi->addIncoming(V, B);

  annotateUniformIfNeeded(Phi, Reg);
  Map[Reg] = Phi;
  return *Phi;
}

MBBOperandTracker::MBBOperandTracker(const llvm::MachineFunction &MF) : MF(MF) {
  for (const llvm::MachineBasicBlock &MBB : MF)
    VM.insert({std::ref(MBB), MCRegValueMap{}});

  const llvm::Function &F = MF.getFunction();
  /// If this is a kernel entry function, seed the register tracker with the
  /// hardware pre-loaded SGPR/VGPR values.
  if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
    auto *EntryBB = const_cast<llvm::BasicBlock *>(MF.front().getBasicBlock());
    assert(EntryBB && "Entry MBB has no IR basic block");
    llvm::IRBuilder Builder{EntryBB};
    initKernelEntryRegs(MF, Builder, Tracker);
  }
}

llvm::Value &
MBBOperandTracker::getRegisterOperand(const llvm::MachineBasicBlock &MBB,
                                      llvm::MCRegister Reg) {
  return materializeReg(MBB, Reg);
}

llvm::Value &MBBOperandTracker::getRegisterOperand(const llvm::MachineInstr &MI,
                                                   llvm::MCRegister Reg) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI does not have a machine basic block");
  return getRegisterOperand(*MBB, Reg);
}

llvm::Value &
MBBOperandTracker::getOperandAsValue(const llvm::MachineOperand &Op) {
  switch (Op.getType()) {
  case llvm::MachineOperand::MO_Register: {
    const llvm::MachineInstr *MI = Op.getParent();
    assert(MI && "Operand does not have a machine instruction");
    return getRegisterOperand(*MI, Op.getReg());
  }
  case llvm::MachineOperand::MO_Immediate: {
    const llvm::MachineInstr *MI = Op.getParent();
    assert(MI && "Operand does not have a machine instruction");
    llvm::LLVMContext &Ctx = MF.getFunction().getContext();
    return *llvm::ConstantInt::getSigned(llvm::IntegerType::getInt64Ty(Ctx),
                                         Op.getImm());
  }
  case llvm::MachineOperand::MO_MachineBasicBlock: {
    auto *BB = const_cast<llvm::BasicBlock *>(Op.getMBB()->getBasicBlock());
    assert(BB && "MBB operand has no IR BasicBlock");
    return *BB;
  }
  case llvm::MachineOperand::MO_GlobalAddress:
    return *const_cast<llvm::GlobalValue *>(Op.getGlobal());
  default:
    llvm_unreachable("Unhandled operand type");
  }
}

void MBBOperandTracker::setRegOperandValue(const llvm::MachineInstr &MI,
                                           llvm::MCRegister Reg,
                                           llvm::Value *Val) {
  assert(Val && "Val is nullptr");
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI has no parent MBB");
  MCRegValueMap &Map = getMap(*MBB);
  // Emit any extraction IR into the MBB's basic block.
  auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
  assert(BB && "MBB has no IR basic block");
  llvm::IRBuilder Builder{BB};

  // Preserve non-overlapping portions of any partially-overwritten
  // super-registers, then erase fully-covered entries.
  invalidateOverlaps(Map, Reg, Builder);
  annotateUniformIfNeeded(Val, Reg);
  Map[Reg] = Val;
}

void MBBOperandTracker::setRegOperandValue(const llvm::MachineOperand &Op,
                                           llvm::Value *Val) {
  assert(Val && "Val is nullptr");
  assert(Op.isReg() && "Operand is not a register");
  assert(Op.getReg().isPhysical() && "Operand is not a physical register");
  const llvm::MachineInstr *MI = Op.getParent();
  assert(MI && "Machine operand has no parent MI");
  setRegOperandValue(*MI, Op.getReg(), Val);
}

template <uint16_t Opcode>
void raiseMachineInstr(const llvm::MachineInstr &MI,
                       llvm::IRBuilderBase &Builder,
                       MBBOperandTracker &RegisterValueMap);

#define GET_SI_INSTR_SEMANTIC_FUNCTIONS
#define GET_SI_INSTR_SEMANTIC_DISPATCH
#include "SIInstrSemantics.inc"

//===----------------------------------------------------------------------===//
// translateSingleInstr — public API for the fuzzer
//===----------------------------------------------------------------------===//

llvm::Error luthier::translateSingleInstr(const llvm::MachineInstr &MI,
                                          llvm::IRBuilderBase &Builder,
                                          const PhysRegValueMap &InputRegs,
                                          PhysRegValueMap &OutputRegs) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  if (!MBB)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MI has no parent MBB");
  const llvm::MachineFunction *MF = MBB->getParent();
  if (!MF)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "MBB has no parent MF");

  // Create a tracker and seed it with the caller's input register values.
  MBBOperandTracker Tracker(*MF);
  for (const auto &[Reg, Val] : InputRegs)
    Tracker.seedRegValue(*MBB, Reg, Val);

  // Dispatch to the TableGen-generated semantic translation.
  // The static raiseMachineInstr function (from .inc) handles the opcode
  // switch.
  raiseMachineInstr(MI, Builder, Tracker);

  // Extract output register values from the tracker.
  const llvm::MCInstrDesc &Desc = MI.getDesc();
  const auto &TRI = *MF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();

  // Explicit defs.
  for (unsigned I = 0, E = Desc.getNumDefs(); I < E; ++I) {
    const llvm::MachineOperand &DefOp = MI.getOperand(I);
    if (!DefOp.isReg() || !DefOp.getReg().isPhysical())
      continue;
    llvm::MCRegister Reg = DefOp.getReg();
    llvm::Value &Val = Tracker.getRegisterOperand(*MBB, Reg);
    OutputRegs[Reg] = &Val;
  }

  // Implicit defs (SCC, VCC, etc.).
  if (const llvm::MCPhysReg ImpDefs = Desc.implicit_defs()) {
    for (; *ImpDefs; ++ImpDefs) {
      llvm::MCRegister Reg(*ImpDefs);
      llvm::Value &Val = Tracker.getRegisterOperand(*MBB, Reg);
      OutputRegs[Reg] = &Val;
    }
  }

  return llvm::Error::success();
}

/// Initialize the register tracker with the pre-loaded SGPR/VGPR values
/// for an AMDGPU kernel entry function.
///
/// At kernel launch the hardware pre-loads certain values into SGPRs and
/// VGPRs according to the kernel descriptor.  Where possible we emit the
/// corresponding AMDGCN intrinsic so that the resulting IR is idiomatic;
/// for the few pre-loaded values that lack an intrinsic we fall back to a
/// frozen poison placeholder.
static void initKernelEntryRegs(llvm::MachineFunction &MF,
                                llvm::IRBuilderBase &Builder,
                                MBBOperandTracker &Tracker) {
  const auto &Info = *MF.getInfo<llvm::SIMachineFunctionInfo>();
  const auto &TRI = *MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();

  using PV = llvm::AMDGPUFunctionArgInfo::PreloadedValue;

  /// Seed a single preloaded register with \p Val.
  /// Annotates non-VGPR values with \c !amdgpu.uniform.
  auto seed = [&](PV Which, llvm::Value *Val) {
    llvm::MCRegister Reg = Info.getPreloadedReg(Which);
    if (!Reg)
      return;
    if (!TRI.isVGPRPhysReg(Reg)) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(Val))
        I->setMetadata("amdgpu.uniform",
                       llvm::MDNode::get(I->getContext(), {}));
    }
    Tracker.seedRegValue(MF.front(), Reg, Val);
  };

  /// Create a frozen-poison placeholder for values with no intrinsic.
  auto makePlaceholder = [&](PV Which) -> llvm::Value * {
    llvm::MCRegister Reg = Info.getPreloadedReg(Which);
    if (!Reg)
      return nullptr;
    const llvm::TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned BitWidth = TRI.getRegSizeInBits(*RC);
    return Builder.CreateFreeze(
        llvm::PoisonValue::get(Builder.getIntNTy(BitWidth)),
        llvm::formatv("kernel.arg.{0}", TRI.getName(Reg)));
  };

  /// Emit a void-returning intrinsic whose result is a pointer, then
  /// ptrtoint it to match the register's integer type.
  auto ptrIntrinsic = [&](PV Which, llvm::Intrinsic::ID IID) {
    llvm::MCRegister Reg = Info.getPreloadedReg(Which);
    if (!Reg)
      return;
    const llvm::TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned BitWidth = TRI.getRegSizeInBits(*RC);
    llvm::Value *Ptr = Builder.CreateIntrinsic(Builder.getPtrTy(4), IID, {},
                                               nullptr, TRI.getName(Reg));
    seed(Which,
         Builder.CreatePtrToInt(Ptr, Builder.getIntNTy(BitWidth),
                                llvm::formatv("{0}.int", TRI.getName(Reg))));
  };

  /// Emit a scalar-returning intrinsic (i32 or i64).
  auto scalarIntrinsic = [&](PV Which, llvm::Intrinsic::ID IID,
                             llvm::Type *RetTy) {
    llvm::MCRegister Reg = Info.getPreloadedReg(Which);
    if (!Reg)
      return;
    llvm::Value *Val =
        Builder.CreateIntrinsic(RetTy, IID, {}, nullptr, TRI.getName(Reg));
    seed(Which, Val);
  };

  // ---- User SGPRs (allocated in HSA ABI order) ----

  // ImplicitBufferPtr → llvm.amdgcn.implicit.buffer.ptr() : ptr addrspace(4)
  // (Non-HSA only, allocated before PrivateSegmentBuffer.)
  ptrIntrinsic(PV::IMPLICIT_BUFFER_PTR,
               llvm::Intrinsic::amdgcn_implicit_buffer_ptr);

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

  // LDSKernelId → llvm.amdgcn.lds.kernel.id() : i32
  scalarIntrinsic(PV::LDS_KERNEL_ID, llvm::Intrinsic::amdgcn_lds_kernel_id,
                  Builder.getInt32Ty());

  // ---- System SGPRs ----

  // WorkgroupID X/Y/Z → llvm.amdgcn.workgroup.id.{x,y,z}() : i32
  scalarIntrinsic(PV::WORKGROUP_ID_X, llvm::Intrinsic::amdgcn_workgroup_id_x,
                  Builder.getInt32Ty());
  scalarIntrinsic(PV::WORKGROUP_ID_Y, llvm::Intrinsic::amdgcn_workgroup_id_y,
                  Builder.getInt32Ty());
  scalarIntrinsic(PV::WORKGROUP_ID_Z, llvm::Intrinsic::amdgcn_workgroup_id_z,
                  Builder.getInt32Ty());

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
}

llvm::Error translateMachineFunctionToIR(llvm::MachineFunction &MF) {
  llvm::Function &F = MF.getFunction();
  llvm::LLVMContext &Ctx = F.getContext();
  /// Early exit if there are no basic blocks in the machine function
  if (MF.empty())
    return llvm::Error::success();

  /// Delete any basic blocks already present in the IR Function
  if (!F.empty())
    (void)F.erase(F.begin(), F.end());

  /// Create BBs associated with every MBB in the MF
  for (llvm::MachineBasicBlock &MBB : MF) {
    MBB.*get(TagBB()) = llvm::BasicBlock::Create(Ctx, "", &F);
  }

  MBBOperandTracker Tracker(MF);

  /// Iterate over the MBBs in reverse post order (RPO) and raise the machine
  /// instructions in each MBB to LLVM IR.
  /// RPO guarantees that when we visit a block we have already visited its
  /// predecessors.
  for (llvm::MachineBasicBlock &MBB : MF) {
  }

  return llvm::Error::success();
}

} // namespace luthier