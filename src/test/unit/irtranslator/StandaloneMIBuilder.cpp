#include "StandaloneMIBuilder.h"

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>

#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier::test {

//===----------------------------------------------------------------------===//
// Register helpers (same as MachineKernelBuilder)
//===----------------------------------------------------------------------===//

static llvm::MCRegister getSGPR32(const llvm::SIRegisterInfo &TRI,
                                  unsigned Idx) {
  const llvm::TargetRegisterClass *RC =
      llvm::SIRegisterInfo::getSGPRClassForBitWidth(32);
  assert(Idx < RC->getNumRegs() && "SGPR index out of range");
  return RC->getRegister(Idx);
}

static llvm::MCRegister getVGPR32(const llvm::SIRegisterInfo &TRI,
                                  unsigned Idx) {
  const llvm::TargetRegisterClass *RC = TRI.getVGPRClassForBitWidth(32);
  assert(Idx < RC->getNumRegs() && "VGPR index out of range");
  return RC->getRegister(Idx);
}

static llvm::MCRegister getSGPRTuple(const llvm::SIRegisterInfo &TRI,
                                     unsigned BaseIdx, unsigned BitWidth) {
  if (BitWidth <= 32)
    return getSGPR32(TRI, BaseIdx);
  llvm::MCRegister Base = getSGPR32(TRI, BaseIdx);
  const llvm::TargetRegisterClass *RC =
      llvm::SIRegisterInfo::getSGPRClassForBitWidth(BitWidth);
  assert(RC && "No SGPR class for this bit width");
  llvm::MCRegister Super = TRI.getMatchingSuperReg(Base, AMDGPU::sub0, RC);
  assert(Super && "Failed to form SGPR tuple");
  return Super;
}

static llvm::MCRegister getVGPRTuple(const llvm::SIRegisterInfo &TRI,
                                     unsigned BaseIdx, unsigned BitWidth) {
  if (BitWidth <= 32)
    return getVGPR32(TRI, BaseIdx);
  llvm::MCRegister Base = getVGPR32(TRI, BaseIdx);
  const llvm::TargetRegisterClass *RC = TRI.getVGPRClassForBitWidth(BitWidth);
  assert(RC && "No VGPR class for this bit width");
  llvm::MCRegister Super = TRI.getMatchingSuperReg(Base, AMDGPU::sub0, RC);
  assert(Super && "Failed to form VGPR tuple");
  return Super;
}

//===----------------------------------------------------------------------===//
// Build
//===----------------------------------------------------------------------===//

llvm::Expected<StandaloneMIContext>
StandaloneMIBuilder::build(const InstrProfile &Profile) {
  StandaloneMIContext Ctx;
  Ctx.Ctx = std::make_unique<llvm::LLVMContext>();
  Ctx.Mod = std::make_unique<llvm::Module>("standalone_mi", *Ctx.Ctx);
  Ctx.Mod->setDataLayout(TM.createDataLayout());
  Ctx.Mod->setTargetTriple("amdgcn-amd-amdhsa");

  // Stub IR function (required by MachineFunction).
  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*Ctx.Ctx), false);
  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                   "standalone_" + Profile.Name.str(),
                                   Ctx.Mod.get());
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1");
  llvm::BasicBlock::Create(*Ctx.Ctx, "entry", F);

  Ctx.MMI = std::make_unique<llvm::MachineModuleInfo>(&TM);
  Ctx.MF = &Ctx.MMI->getOrCreateMachineFunction(*F);

  llvm::MachineFunction &MF = *Ctx.MF;
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const auto &TRI = *ST.getRegisterInfo();

  auto *BB = MF.CreateMachineBasicBlock();
  MF.push_back(BB);
  llvm::DebugLoc DL;

  // ====================================================================
  // Allocate physical registers for operands.
  //
  // Index-based allocation:
  //   SGPR 10-19 : scalar inputs
  //   SGPR 20-29 : scalar outputs
  //   VGPR 10-19 : vector inputs
  //   VGPR 20-29 : vector outputs
  // ====================================================================
  unsigned NextInSGPRIdx = 10;
  unsigned NextInVGPRIdx = 10;
  unsigned NextOutSGPRIdx = 20;
  unsigned NextOutVGPRIdx = 20;

  // Pre-allocate output registers (defs come first in MachineInstr).
  for (const auto &Out : Profile.Outputs) {
    unsigned NumDwords = (Out.SizeBits + 31) / 32;
    llvm::MCRegister Reg;
    if (Out.IsSGPR || !Out.IsVGPR) {
      Reg = getSGPRTuple(TRI, NextOutSGPRIdx, Out.SizeBits);
      NextOutSGPRIdx += NumDwords;
    } else {
      Reg = getVGPRTuple(TRI, NextOutVGPRIdx, Out.SizeBits);
      NextOutVGPRIdx += NumDwords;
    }
    Ctx.OutputRegs.push_back(Reg);
  }

  // Pre-allocate input registers.
  for (const auto &In : Profile.Inputs) {
    if (In.IsImm) {
      Ctx.InputRegs.push_back(llvm::MCRegister()); // Placeholder.
      continue;
    }
    unsigned NumDwords = (In.SizeBits + 31) / 32;
    llvm::MCRegister Reg;
    if (In.IsSGPR || !In.IsVGPR) {
      Reg = getSGPRTuple(TRI, NextInSGPRIdx, In.SizeBits);
      NextInSGPRIdx += NumDwords;
    } else {
      Reg = getVGPRTuple(TRI, NextInVGPRIdx, In.SizeBits);
      NextInVGPRIdx += NumDwords;
    }
    Ctx.InputRegs.push_back(Reg);
  }

  // ====================================================================
  // Build the single MachineInstr.
  // ====================================================================
  auto MIB = BuildMI(*BB, BB->end(), DL, TII.get(Profile.Opcode));

  // Defs (outputs).
  for (llvm::MCRegister Reg : Ctx.OutputRegs)
    MIB.addDef(Reg);

  // Uses (inputs).
  for (unsigned I = 0; I < Profile.Inputs.size(); ++I) {
    if (Profile.Inputs[I].IsImm)
      MIB.addImm(0); // Placeholder — the fuzzer can patch this later.
    else
      MIB.addReg(Ctx.InputRegs[I]);
  }

  Ctx.MI = MIB.getInstr();
  return std::move(Ctx);
}

} // namespace luthier::test
