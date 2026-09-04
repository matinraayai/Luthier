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
StandaloneMIBuilder::build(uint16_t Opcode) {
  StandaloneMIContext Ctx;
  Ctx.Ctx = std::make_unique<llvm::LLVMContext>();
  Ctx.Mod = std::make_unique<llvm::Module>("standalone_mi", *Ctx.Ctx);
  Ctx.Mod->setDataLayout(TM.createDataLayout());
  Ctx.Mod->setTargetTriple(llvm::Triple("amdgcn-amd-amdhsa"));

  // Stub IR function (required by MachineFunction).
  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*Ctx.Ctx), false);
  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                   "standalone_mi", Ctx.Mod.get());
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

  const llvm::MCInstrDesc &OpcodeDesc = TII.get(Opcode);

  llvm::MachineInstrBuilder MIB = BuildMI(*BB, BB->end(), {}, OpcodeDesc);

  // --- Analyze explicit operands ---
  unsigned NumDefs = OpcodeDesc.getNumDefs();
  unsigned NumOps = OpcodeDesc.getNumOperands();

  for (unsigned I = 0; I < NumOps; ++I) {
    const llvm::MCOperandInfo &OpInfo = OpcodeDesc.operands()[I];
    OperandInfo OI;
    OI.Idx = I;
    OI.IsDef = I < NumDefs;
    OI.IsReg = OpInfo.OperandType == llvm::MCOI::OPERAND_REGISTER;
    OI.IsImm = OpInfo.OperandType == llvm::MCOI::OPERAND_IMMEDIATE;
    OI.IsSGPR = false;
    OI.IsVGPR = false;
    OI.SizeBits = 32;
    OI.RegClassID = OpInfo.RegClass;

    if (OI.IsReg && OpInfo.RegClass != static_cast<unsigned>(-1)) {
      const llvm::TargetRegisterClass *RC = TRI.getRegClass(OpInfo.RegClass);
      OI.SizeBits = MRI->getRegClass(OpInfo.RegClass).getSizeInBits();

      // Heuristic: AMDGPU SGPR classes have "SReg" or "SGPR" in name,
      // VGPR classes have "VGPR" or "VReg".
      llvm::StringRef ClassName = MRI->getRegClassName(&RC);
      OI.IsSGPR = ClassName.contains("SReg") || ClassName.contains("SGPR");
      OI.IsVGPR = ClassName.contains("VGPR") || ClassName.contains("VReg");
    }

    if (OI.IsDef)
      P.Outputs.push_back(OI);
    else
      P.Inputs.push_back(OI);
  }

  // --- Implicit defs and uses ---
  if (const llvm::MCPhysReg *ImpDefs = OpcodeDesc.implicit_defs()) {
    for (; *ImpDefs; ++ImpDefs)
      P.ImplicitDefs.push_back(llvm::MCRegister(*ImpDefs));
  }
  if (const llvm::MCPhysReg *ImpUses = OpcodeDesc.implicit_uses()) {
    for (; *ImpUses; ++ImpUses)
      P.ImplicitUses.push_back(llvm::MCRegister(*ImpUses));
  }

  // --- Memory access detection ---
  P.Mem.MemKind = MemAccessInfo::None;
  if (OpcodeDesc.mayLoad() || OpcodeDesc.mayStore()) {
    // Determine memory kind from instruction name prefix.
    if (P.Name.starts_with("S_LOAD") || P.Name.starts_with("S_STORE") ||
        P.Name.starts_with("S_BUFFER") || P.Name.starts_with("S_SCRATCH") ||
        P.Name.starts_with("S_ATOMIC"))
      P.Mem.MemKind = MemAccessInfo::Buffer;
    else if (P.Name.starts_with("DS_"))
      P.Mem.MemKind = MemAccessInfo::LDS;
    else if (P.Name.starts_with("FLAT_"))
      P.Mem.MemKind = MemAccessInfo::Flat;
    else if (P.Name.starts_with("GLOBAL_"))
      P.Mem.MemKind = MemAccessInfo::Global;
    else if (P.Name.starts_with("SCRATCH_"))
      P.Mem.MemKind = MemAccessInfo::Scratch;
    else if (P.Name.starts_with("BUFFER_"))
      P.Mem.MemKind = MemAccessInfo::Buffer;
  }

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
