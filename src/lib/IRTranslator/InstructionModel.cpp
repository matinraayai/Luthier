#include "luthier/IRTranslator/InstructionModel.h"
#include "luthier/Common/DenseMapInfo.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Tooling/TargetMachineInstrMDNode.h"
#include <SIInstrInfo.h>
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

namespace luthier {

struct TagBB {
  using type = const llvm::BasicBlock *llvm::MachineBasicBlock::*;

  friend type get(TagBB);
};

// This class defines a friend function that can be called via ADL using the tag
// type.
template <typename Tag, typename Tag::type MemPtr> struct Access {
  friend typename Tag::type get(Tag) { return MemPtr; }
};

template struct Access<TagBB, &llvm::MachineBasicBlock::BB>;

/// \returns the greatest common divisor (GCD) of the sizes of all register
/// units of \p Reg
static unsigned getRegUnitsSizeGCD(llvm::MCRegister Reg,
                                   const llvm::TargetRegisterInfo &TRI) {
  unsigned GCD = 1;
  for (llvm::MCRegUnit RegUnit : TRI.regunits(Reg)) {
    for (llvm::MCRegUnitRootIterator Root(RegUnit, &TRI); Root.isValid();
         ++Root) {
      const llvm::TargetRegisterClass *RootRegClass =
          TRI.getMinimalPhysRegClass(*Root);
      assert(RootRegClass &&
             "Failed to get the register class of the root register");
      const unsigned RootRegSize = TRI.getRegSizeInBits(*RootRegClass);
      GCD = std::gcd(GCD, RootRegSize);
    }
  }
  return GCD;
}

/// \brief A utility class used to materialize LLVM Values from
/// <tt>llvm::MachineOperand</tt>s; Inserts any additional cast instructions
/// needed for register values
class MBBOperandTracker {

  using MCRegValueMap =
      llvm::DenseMap<llvm::MCRegister, std::reference_wrapper<llvm::Value>>;

  using BBValueMap =
      llvm::DenseMap<std::reference_wrapper<const llvm::MachineBasicBlock>,
                     MCRegValueMap>;

  const llvm::MachineFunction &MF;

  BBValueMap VM{};

  llvm::Value &getRegisterOperand(const llvm::MachineBasicBlock &MBB,
                                  llvm::MCRegister Reg);

public:
  explicit MBBOperandTracker(const llvm::MachineFunction &MF) : MF(MF) {
    for (const llvm::MachineBasicBlock &MBB : MF) {
      VM.insert({std::ref(MBB), MCRegValueMap{}});
    }
  };

  /// Given \p Op returns its equivalent \c llvm::Value
  /// Register operands are returned as unsigned integers with the same
  /// size as the register
  llvm::Value &getOperandAsValue(const llvm::MachineOperand &Op);

  void setRegOperandValue(const llvm::MachineOperand &Op, llvm::Value &Val);
};

llvm::Value &
MBBOperandTracker::getRegisterOperand(const llvm::MachineBasicBlock &MBB,
                                      llvm::MCRegister Reg) {
  assert(VM.contains(MBB) &&
         "Failed to find the value map associated with MBB");
  auto &MBBValueMap = VM[MBB];
  auto *BB = const_cast<llvm::BasicBlock *>(MBB.getBasicBlock());
  assert(BB && "MBB does not have a IR basic block associated with it");
  const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const llvm::TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
  assert(RC && "The value associated with the register class is nullptr");
  const auto RegSize = TRI->getRegSizeInBits(*RC);
  llvm::StringRef RegName = TRI->getName(Reg);
  llvm::IRBuilder Builder{BB};

  /// Iterate over the regunits of this register and find the smallest one;
  /// This size will be the value granularity of the vector type we will build
  /// out of the concatenated values of the regunits
  llvm::IntegerType *SmallestScalarTy =
      Builder.getIntNTy(getRegUnitsSizeGCD(Reg, *TRI));

  /// Now iterate over the reg units, and get the values currently associated
  /// with them in the value map
  llvm::SmallVector<llvm::Value *, 4> RootRegVals{};
  for (llvm::MCRegUnit RegUnit : TRI->regunits(Reg)) {
    for (llvm::MCRegUnitRootIterator Root(RegUnit, TRI); Root.isValid();
         ++Root) {
      llvm::StringRef RootRegName = TRI->getName(*Root);
      auto RootValIt = MBBValueMap.find(*Root);
      const llvm::TargetRegisterClass *RootRegClass =
          TRI->getMinimalPhysRegClass(*Root);
      assert(RootRegClass &&
             "Failed to get the register class of the root register");
      const auto RootRegSize = TRI->getRegSizeInBits(*RootRegClass);
      llvm::Value *RootRegVal{nullptr};
      /// If there isn't a value currently associated with the regunit
      /// inside the current basic block, we have to recursively search the
      /// predecessors for the value instead
      if (RootValIt == MBBValueMap.end()) {
        llvm::IntegerType *RootRegType = Builder.getIntNTy(RootRegSize);
        /// Find the value from the successor basic blocks and emit a PHI
        /// instruction
        llvm::SmallVector<std::pair<llvm::Value *, llvm::BasicBlock *>>
            PhiVals{};
        PhiVals.reserve(MBB.pred_size());
        for (llvm::MachineBasicBlock *Pred : MBB.predecessors()) {
          llvm::Value &PredRootRegVal = getRegisterOperand(*Pred, *Root);
          auto *PredBB = const_cast<llvm::BasicBlock *>(Pred->getBasicBlock());
          assert(PredBB && "Predecessor MBB does not have an IR basic block "
                           "associated with it");
          PhiVals.push_back({&PredRootRegVal, PredBB});
        }
        /// If we have reached a block with no predecessor and we couldn't
        /// find any values for the register, emit an undef value for it
        if (PhiVals.empty()) {
          RootRegVal = llvm::UndefValue::get(RootRegType);
        } else {
          llvm::PHINode *Phi =
              Builder.CreatePHI(RootRegType, PhiVals.size(), RootRegName);
          for (const auto &[V, B] : PhiVals) {
            Phi->addIncoming(V, B);
          }
          RootRegVal = Phi;
        }
      } else {
        RootRegVal = &RootValIt->getSecond().get();
      }
      /// If the regunit is bigger than the smallest scalar type, emit a
      /// bitcast to break it down to the smaller values, and then add them
      /// to the array of values for the target register
      if (RootRegSize > SmallestScalarTy->getBitWidth()) {
        unsigned NumRootSubRegs = RootRegSize / SmallestScalarTy->getBitWidth();
        llvm::Value *BitCastZExtRootReg = Builder.CreateBitCast(
            RootRegVal,
            llvm::FixedVectorType::get(SmallestScalarTy, NumRootSubRegs));
        for (int i = 0; i < NumRootSubRegs; ++i) {
          RootRegVals.push_back(Builder.CreateExtractElement(
              BitCastZExtRootReg, i,
              llvm::formatv("{0}.sub.{1}", RootRegName, i)));
        }
      } else {
        RootRegVals.push_back(RootRegVal);
      }
    }
  }

  /// If there are only a single value, return it; Otherwise construct a vector
  /// type and concatenate all the registers together, and bitcast the
  /// vector into the final value associated with the register
  if (RootRegVals.size() == 1)
    return *RootRegVals[0];

  llvm::Value *OutVecVal{nullptr};
  llvm::FixedVectorType *FixedVecType =
      llvm::FixedVectorType::get(SmallestScalarTy, RootRegVals.size());
  /// Create insert element instructions to merge the values together into a
  /// single vector value, and then bitcast it to a single scalar value
  for (auto [Idx, SubVal] : llvm::enumerate(RootRegVals)) {
    if (Idx == 0) {
      OutVecVal = Builder.CreateInsertElement(
          FixedVecType, SubVal, Idx,
          llvm::formatv("{0}.sub.{0}", RegName, Idx));
    } else
      OutVecVal = Builder.CreateInsertElement(
          OutVecVal, SubVal, Idx, llvm::formatv("{0}.sub.{0}", RegName, Idx));
  }
  return *Builder.CreateBitCast(OutVecVal, Builder.getIntNTy(RegSize), RegName);
}

llvm::Value &
MBBOperandTracker::getOperandAsValue(const llvm::MachineOperand &Op) {
  switch (Op.getType()) {
  case llvm::MachineOperand::MO_Register: {
    const llvm::MachineInstr *MI = Op.getParent();
    assert(MI && "Operand does not have a machine instruction");
    const llvm::MachineBasicBlock *MBB = MI->getParent();
    assert(MBB && "Operand does not have a machine basic block");
    return getRegisterOperand(*MBB, Op.getReg());
  }
  case llvm::MachineOperand::MO_Immediate: {
    const llvm::MachineInstr *MI = Op.getParent();
    assert(MI && "Operand does not have a machine instruction");
    const llvm::MachineBasicBlock *MBB = MI->getParent();
    assert(MBB && "Operand does not have a machine basic block");
    const llvm::MachineFunction *OpMF = MBB->getParent();
    assert(OpMF && "Operand does not have a machine function");
    llvm::LLVMContext &Ctx = OpMF->getFunction().getContext();
    return *llvm::ConstantInt::getSigned(llvm::IntegerType::getInt64Ty(Ctx),
                                         Op.getImm());
  }
  case llvm::MachineOperand::MO_MachineBasicBlock: {
    auto *BB = const_cast<llvm::BasicBlock *>(Op.getMBB()->getBasicBlock());
    assert(BB &&
           "MBB operand doesn't have an IR Basic Block associated with it");
    return *BB;
  }
  case llvm::MachineOperand::MO_GlobalAddress: {
    return *const_cast<llvm::GlobalValue *>(Op.getGlobal());
  }
  default:
    llvm_unreachable("Unsupported operand type");
  }
}

void MBBOperandTracker::setRegOperandValue(const llvm::MachineOperand &Op,
                                           llvm::Value &Val) {
  assert(Op.isReg() && "Operand is not a register");
  assert(Op.getReg().isPhysical() && "Operand is not a physical register");
  llvm::MCRegister Reg = Op.getReg();
  const llvm::MachineInstr *MI = Op.getParent();
  assert(MI && "Machine operand does not have a machine instruction");
  const llvm::MachineBasicBlock *MBB = MI->getParent();
  assert(MBB && "Machine operand does not have a machine basic block");

  assert(VM.contains(*MBB) &&
         "Failed to find the value map associated with MBB");
  auto &MBBValueMap = VM[*MBB];
  auto *BB = const_cast<llvm::BasicBlock *>(MBB->getBasicBlock());
  assert(BB && "MBB does not have a IR basic block associated with it");
  const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const llvm::TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
  assert(RC && "The value associated with the register class is nullptr");
  const auto RegSize = TRI->getRegSizeInBits(*RC);
  assert(RegSize == Val.getType()->getIntegerBitWidth() &&
         "Physical reg size and value size don't match");
  llvm::StringRef RegName = TRI->getName(Reg);
  llvm::IRBuilder Builder{BB};

  /// Break down the register value into the GCD of all reg unit sizes
  unsigned GCDRegUnitSize = getRegUnitsSizeGCD(Reg, *TRI);

  /// If the size of the register is the same as its only regunit, then assign
  /// it and return; Otherwise, we need to extract each regunit's value from
  /// the big value
  if (GCDRegUnitSize == RegSize) {
    MBBValueMap.emplace_or_assign(Reg, std::ref(Val));
    return;
  }

  llvm::IntegerType *SmallestScalarTy = Builder.getIntNTy(GCDRegUnitSize);

  unsigned NumVecEls = RegSize / GCDRegUnitSize;
  llvm::Type *BitCastType =
      llvm::FixedVectorType::get(SmallestScalarTy, NumVecEls);
  llvm::Value *BitCastedVal = Builder.CreateBitCast(
      &Val, BitCastType, llvm::formatv("{0}.bitcast", RegName));

  unsigned CurrentRegUnitIdx = 0;
  llvm::SmallDenseMap<unsigned, llvm::Value *> VecSizeToVecTypeCast;
  for (llvm::MCRegUnit RegUnit : TRI->regunits(Reg)) {
    for (llvm::MCRegUnitRootIterator Root(RegUnit, TRI); Root.isValid();
         ++Root) {
      llvm::StringRef RootRegName = TRI->getName(*Root);
      const llvm::TargetRegisterClass *RootRegClass =
          TRI->getMinimalPhysRegClass(*Root);
      assert(RootRegClass &&
             "Failed to get the register class of the root register");
      const auto RootRegSize = TRI->getRegSizeInBits(*RootRegClass);
      unsigned NumSubRegs = RootRegSize / GCDRegUnitSize;
      /// Extract the values associated with this regunit and then merge them
      /// together into a single register
      if (NumSubRegs == 1) {
        MBBValueMap.emplace_or_assign(
            *Root, *Builder.CreateExtractElement(
                       BitCastedVal, CurrentRegUnitIdx, RootRegName));
      } else {
        llvm::Value *MergedVal{nullptr};
        for (unsigned int Idx = 0; Idx < NumSubRegs; ++Idx) {
          auto *V = Builder.CreateExtractElement(
              BitCastedVal, Idx + CurrentRegUnitIdx,
              llvm::formatv("{0}.sub.{1}", RootRegName, Idx));
          if (Idx == 0)
            MergedVal = Builder.CreateInsertElement(
                llvm::FixedVectorType::get(Builder.getIntNTy(GCDRegUnitSize),
                                           NumSubRegs),
                V, Idx, RootRegName);
          else
            MergedVal =
                Builder.CreateInsertElement(MergedVal, V, Idx, RootRegName);
        }
        MergedVal = Builder.CreateBitCast(
            MergedVal, Builder.getIntNTy(RootRegSize), RootRegName);
        MBBValueMap.emplace_or_assign(*Root, *MergedVal);
      }
      CurrentRegUnitIdx += NumSubRegs;
    }
  }
}

using MBBToBBMapTy =
    llvm::DenseMap<std::reference_wrapper<const llvm::MachineBasicBlock>,
                   std::reference_wrapper<llvm::BasicBlock>>;

template <uint16_t Opcode, typename... Operands>
void raiseMachineInstr(const llvm::MachineInstr &MI,
                       llvm::IRBuilderBase &Builder,
                       MBBOperandTracker &RegisterValueMap);

template <>
void raiseMachineInstr<llvm::AMDGPU::S_ADD_U32>(
    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,
    MBBOperandTracker &RegisterValueMap) {
  /// D.u = S0.u + S1.u;
  /// SCC = (S0.u + S1.u >= 0x100000000ULL ? 1 : 0). // unsigned
  /// overflow/carry-out
  const llvm::MachineOperand &D = MI.getOperand(0);
  const llvm::MachineOperand &S0 = MI.getOperand(1);
  const llvm::MachineOperand &S1 = MI.getOperand(2);

  llvm::Value &Op1Val = RegisterValueMap.getOperandAsValue(S0);
  llvm::Value &Op2Val = RegisterValueMap.getOperandAsValue(S1);
  llvm::Value *DVal = Builder.CreateAdd(&Op1Val, &Op2Val);
  llvm::Value *SCCVal = Builder.CreateCmp(llvm::CmpInst::ICMP_UGE, DVal,
                                          Builder.getInt64(0x100000000ULL));
  assert(MI.getNumImplicitOperands() == 1 &&
         "Incorrect number of implicit operands");
  const llvm::MachineOperand &SCCOp = *MI.implicit_operands().begin();
  assert(SCCOp.isReg() && SCCOp.getReg() == llvm::AMDGPU::SCC &&
         "SCC is not the implicit operand of the instruction");
  RegisterValueMap.setRegOperandValue(D, *DVal);
  RegisterValueMap.setRegOperandValue(SCCOp, *SCCVal);
}

template <>
void raiseMachineInstr<llvm::AMDGPU::S_SUB_U32>(
    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,
    MBBOperandTracker &RegisterValueMap) {
  /// D.u = S0.u - S1.u;
  /// SCC = (S1.u > S0.u ? 1 : 0). // unsigned overflow or carry-out for
  /// S_SUBB_U32.
  const llvm::MachineOperand &D = MI.getOperand(0);
  const llvm::MachineOperand &S0 = MI.getOperand(1);
  const llvm::MachineOperand &S1 = MI.getOperand(2);
  llvm::Value &S0Val = RegisterValueMap.getOperandAsValue(S0);
  llvm::Value &S1Val = RegisterValueMap.getOperandAsValue(S1);
  llvm::Value *DVal = Builder.CreateSub(&S0Val, &S1Val);
  llvm::Value *SCCVal =
      Builder.CreateCmp(llvm::CmpInst::ICMP_UGT, &S1Val, &S0Val);
  RegisterValueMap.setRegOperandValue(D, *DVal);
  assert(MI.getNumImplicitOperands() == 1 &&
         "Incorrect number of implicit operands");
  const llvm::MachineOperand &SCCOp = *MI.implicit_operands().begin();
  assert(SCCOp.isReg() && SCCOp.getReg() == llvm::AMDGPU::SCC &&
         "SCC is not the implicit operand of the instruction");
  RegisterValueMap.setRegOperandValue(SCCOp, *SCCVal);
}

template <>
void raiseMachineInstr<llvm::AMDGPU::S_ADD_I32>(
    const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,
    MBBOperandTracker &RegisterValueMap) {
  /// tmp = S0.i32 + S1.i32;
  /// SCC = ((S0.u32[31] == S1.u32[31]) && (S0.u32[31] != tmp.u32[31]));
  /// // signed overflow.
  /// D0.i32 = tmp.i32
  const llvm::MachineOperand &D = MI.getOperand(0);
  const llvm::MachineOperand &S0 = MI.getOperand(1);
  const llvm::MachineOperand &S1 = MI.getOperand(2);
  llvm::Value &S0Val = RegisterValueMap.getOperandAsValue(S0);
  llvm::Value &S1Val = RegisterValueMap.getOperandAsValue(S1);
  llvm::Value *DVal = Builder.CreateSub(&S0Val, &S1Val);
  llvm::Value *SCCVal =
      Builder.CreateCmp(llvm::CmpInst::ICMP_UGT, &S1Val, &S0Val);
  RegisterValueMap.setRegOperandValue(D, *DVal);
  assert(MI.getNumImplicitOperands() == 1 &&
         "Incorrect number of implicit operands");
  const llvm::MachineOperand &SCCOp = *MI.implicit_operands().begin();
  assert(SCCOp.isReg() && SCCOp.getReg() == llvm::AMDGPU::SCC &&
         "SCC is not the implicit operand of the instruction");
  RegisterValueMap.setRegOperandValue(SCCOp, *SCCVal);
}

llvm::Expected<llvm::Value &>
raiseMachineInstr(const llvm::MachineInstr &MI, llvm::IRBuilderBase &Builder,
                  MBBToBBMapTy &MBBToBBMap,
                  BasicBlockMCRegisterValueMap &RegisterValueMap) {
  llvm::Value *Out{nullptr};
  llvm::LLVMContext &Ctx = Builder.getContext();
  switch (MI.getOpcode()) {
  case llvm::AMDGPU::S_ADD_U32:
    raiseMachineInstr<llvm::AMDGPU::S_ADD_U32>(
        Builder, MBBToBBMap, RegisterValueMap, MI.getOperand(0),
        MI.getOperand(1), MI.getOperand(2));
    break;
  case llvm::AMDGPU::S_SUB_U32:
    /// D.u = S0.u - S1.u;
    /// SCC = (S1.u > S0.u ? 1 : 0). // unsigned overflow or carry-out for
    /// S_SUBB_U32
    const llvm::MachineOperand &Op1 = MI.getOperand(1);
    const llvm::MachineOperand &Op2 = MI.getOperand(2);
    llvm::Value *Op1Val = Op1.isImm() ? Builder.getInt32(Op2.getImm())
                                      : RegisterValueMap.getValue(Op1.getReg());
    llvm::Value *Op2Val = Op2.isImm() ? Builder.getInt32(Op2.getImm())
                                      : RegisterValueMap.getValue(Op2.getReg());
    if (!Op1Val) {
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to create the first operand");
    }
    if (!Op2Val) {
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to create the second operand");
    }
    Out = Builder.CreateAdd(Op1Val, Op2Val);
    llvm::Value *SCCVal = Builder.CreateSelect(
        Builder.CreateCmp(llvm::CmpInst::ICMP_UGE,
                          Builder.CreateAdd(Op1Val, Op2Val),
                          Builder.getInt64(0x100000000ULL)),
        Builder.getInt1(true), Builder.getInt1(false));
    RegisterValueMap.setValue(MI.getOperand(0).getReg(), *Out);
    RegisterValueMap.setValue(llvm::AMDGPU::SCC, *SCCVal);
    break;
  case llvm::AMDGPU::S_ENDPGM:
    RegisterValueMap.setValue(
        MI.getOperand(0).getReg(),
        *Builder.CreateIntrinsic(Builder.getVoidTy(),
                                 llvm::Intrinsic::amdgcn_endpgm, {}));
    break;
  case llvm::AMDGPU::S_GETPC_B64:
    RegisterValueMap.setValue(
        MI.getOperand(0).getReg(),
        *Builder.CreateIntrinsic(Builder.getInt64Ty(),
                                 llvm::Intrinsic::amdgcn_s_getpc, {}));
    break;
  case llvm::AMDGPU::S_ADD_I32:
    const llvm::MachineOperand &Op1 = MI.getOperand(1);
    const llvm::MachineOperand &Op2 = MI.getOperand(2);
    llvm::Value *Op1Val = Op1.isImm() ? Builder.getInt32(Op1.getImm())
                                      : RegisterValueMap.getValue(Op1.getReg());
    llvm::Value *Op2Val = Op2.isImm() ? Builder.getInt32(Op2.getImm())
                                      : RegisterValueMap.getValue(Op2.getReg());
    if (!Op1Val) {
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to create the first operand");
    }
    if (!Op2Val) {
      return LUTHIER_MAKE_GENERIC_ERROR("Failed to create the second operand");
    }
    Out = Builder.CreateAdd(Op1Val, Op2Val);
    RegisterValueMap.setValue(MI.getOperand(0).getReg(), *Out);
    break;
  default:
    llvm_unreachable("Not modeled");
  }
}

llvm::PreservedAnalyses IRTranslator::run(llvm::Function &F,
                                          llvm::FunctionAnalysisManager &FAM) {
  llvm::LLVMContext &Ctx = F.getContext();
  /// Get the machine function and its analysis manager
  llvm::MachineFunction &MF =
      FAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
  llvm::MachineFunctionAnalysisManager &MFAM =
      FAM.getResult<llvm::MachineFunctionAnalysisManagerFunctionProxy>(F)
          .getManager();
  /// Create BBs associated with every MBB in the MF
  for (llvm::MachineBasicBlock &MBB : MF) {
    MBB.*get(TagBB()) = llvm::BasicBlock::Create(Ctx, "", &F);
  }
  /// Iterate over the MBBs in reverse post order (RPO) and raise the machine
  /// instructions in each MBB to LLVM IR
  /// RPO is used to guarantee when we vist a block we have already visited its
  /// predecessors
  for (llvm::MachineBasicBlock &MBB : MF) {
  }
}

} // namespace luthier