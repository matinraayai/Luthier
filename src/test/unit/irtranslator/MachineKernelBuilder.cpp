#include "MachineKernelBuilder.h"

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>

#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

namespace luthier::test {

//===----------------------------------------------------------------------===//
// Register helpers (shared with StandaloneMIBuilder)
//===----------------------------------------------------------------------===//

static llvm::MCRegister getSGPR32(const llvm::SIRegisterInfo &TRI,
                                  unsigned Idx) {
  const auto *RC = llvm::SIRegisterInfo::getSGPRClassForBitWidth(32);
  return RC->getRegister(Idx);
}

static llvm::MCRegister getVGPR32(const llvm::SIRegisterInfo &TRI,
                                  unsigned Idx) {
  const auto *RC = TRI.getVGPRClassForBitWidth(32);
  return RC->getRegister(Idx);
}

static llvm::MCRegister getSGPRTuple(const llvm::SIRegisterInfo &TRI,
                                     unsigned BaseIdx, unsigned BitWidth) {
  if (BitWidth <= 32)
    return getSGPR32(TRI, BaseIdx);
  auto Base = getSGPR32(TRI, BaseIdx);
  const auto *RC = llvm::SIRegisterInfo::getSGPRClassForBitWidth(BitWidth);
  return TRI.getMatchingSuperReg(Base, AMDGPU::sub0, RC);
}

static llvm::MCRegister getVGPRTuple(const llvm::SIRegisterInfo &TRI,
                                     unsigned BaseIdx, unsigned BitWidth) {
  if (BitWidth <= 32)
    return getVGPR32(TRI, BaseIdx);
  auto Base = getVGPR32(TRI, BaseIdx);
  const auto *RC = TRI.getVGPRClassForBitWidth(BitWidth);
  return TRI.getMatchingSuperReg(Base, AMDGPU::sub0, RC);
}

static unsigned sloadOpcode(unsigned NumDwords) {
  switch (NumDwords) {
  case 1:  return AMDGPU::S_LOAD_DWORD_IMM;
  case 2:  return AMDGPU::S_LOAD_DWORDX2_IMM;
  case 3:  return AMDGPU::S_LOAD_DWORDX3_IMM;
  case 4:  return AMDGPU::S_LOAD_DWORDX4_IMM;
  case 8:  return AMDGPU::S_LOAD_DWORDX8_IMM;
  case 16: return AMDGPU::S_LOAD_DWORDX16_IMM;
  default: return AMDGPU::S_LOAD_DWORD_IMM;
  }
}

//===----------------------------------------------------------------------===//
// Build
//===----------------------------------------------------------------------===//

llvm::Expected<KernelMFContext>
MachineKernelBuilder::build(const StandaloneMIContext &MICtx,
                            const InstrProfile &Profile,
                            KernargLayout &Layout) {
  KernelMFContext KCtx;
  KCtx.Ctx = std::make_unique<llvm::LLVMContext>();
  KCtx.Mod = std::make_unique<llvm::Module>("kernel_ref", *KCtx.Ctx);
  KCtx.Mod->setDataLayout(TM.createDataLayout());
  KCtx.Mod->setTargetTriple("amdgcn-amd-amdhsa");

  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(*KCtx.Ctx), false);
  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                   getKernelName(Profile), KCtx.Mod.get());
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1");
  llvm::BasicBlock::Create(*KCtx.Ctx, "entry", F);

  KCtx.MMI = std::make_unique<llvm::MachineModuleInfo>(&TM);
  KCtx.MF = &KCtx.MMI->getOrCreateMachineFunction(*F);

  llvm::MachineFunction &MF = *KCtx.MF;
  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const auto &TRI = *ST.getRegisterInfo();

  auto *BB = MF.CreateMachineBasicBlock();
  MF.push_back(BB);
  llvm::DebugLoc DL;

  // Fixed ABI registers.
  llvm::MCRegister KernargPtrReg = getSGPRTuple(TRI, 4, 64);
  llvm::MCRegister OutPtrSGPR = getSGPRTuple(TRI, 6, 64);
  llvm::MCRegister OutPtrVGPR = getVGPRTuple(TRI, 0, 64);

  unsigned MovVGPR32 = TII.getMovOpcode(TRI.getVGPRClassForBitWidth(32));

  // ====================================================================
  // 1. Compute kernarg layout from the standalone MI's registers.
  // ====================================================================
  unsigned KernargOffset = 0;

  // Input fields — one per register input in MICtx.InputRegs.
  for (unsigned I = 0; I < MICtx.InputRegs.size(); ++I) {
    llvm::MCRegister Reg = MICtx.InputRegs[I];
    if (!Reg) {
      // Immediate operand.
      Layout.Fields.push_back({KernargOffset, 4, true,
                               llvm::formatv("in{0}", I)});
      KernargOffset += 4;
      continue;
    }
    const auto *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned Bytes = TRI.getRegSizeInBits(*RC) / 8;
    KernargOffset = (KernargOffset + Bytes - 1) & ~(Bytes - 1);
    Layout.Fields.push_back({KernargOffset, Bytes, true,
                             llvm::formatv("in{0}", I)});
    KernargOffset += Bytes;
  }

  // Output pointer.
  KernargOffset = (KernargOffset + 7) & ~7u;
  Layout.OutputPtrOffset = KernargOffset;
  Layout.Fields.push_back({KernargOffset, 8, false, "out_ptr"});
  KernargOffset += 8;
  Layout.TotalSize = KernargOffset;

  // Output fields.
  unsigned OutputOffset = 0;
  for (unsigned I = 0; I < MICtx.OutputRegs.size(); ++I) {
    llvm::MCRegister Reg = MICtx.OutputRegs[I];
    const auto *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned Bytes = TRI.getRegSizeInBits(*RC) / 8;
    OutputOffset = (OutputOffset + Bytes - 1) & ~(Bytes - 1);
    Layout.OutputFields.push_back({OutputOffset, Bytes,
                                   llvm::formatv("out{0}", I)});
    OutputOffset += Bytes;
  }

  // Implicit defs.
  for (llvm::MCRegister Reg : Profile.ImplicitDefs) {
    llvm::StringRef RName = TRI.getName(Reg);
    unsigned Size = (RName.starts_with("VCC") || RName.starts_with("EXEC"))
                        ? 8 : 4;
    Layout.OutputFields.push_back({OutputOffset, Size, RName});
    OutputOffset += Size;
  }
  Layout.OutputBufSize = OutputOffset;

  // ====================================================================
  // 2. Load input registers from kernarg.
  // ====================================================================
  unsigned FieldIdx = 0;
  for (unsigned I = 0; I < MICtx.InputRegs.size(); ++I) {
    llvm::MCRegister Reg = MICtx.InputRegs[I];
    if (!Reg) {
      ++FieldIdx;
      continue; // Immediates baked into the MI.
    }

    const auto &Fld = Layout.Fields[FieldIdx++];
    const auto *RC = TRI.getMinimalPhysRegClass(Reg);
    unsigned Bits = TRI.getRegSizeInBits(*RC);
    unsigned NumDwords = (Bits + 31) / 32;

    if (TRI.isSGPRPhysReg(Reg)) {
      BuildMI(*BB, BB->end(), DL, TII.get(sloadOpcode(NumDwords)), Reg)
          .addReg(KernargPtrReg)
          .addImm(Fld.Offset)
          .addImm(0);
    } else {
      // Load into temp SGPRs, then V_MOV each dword.
      for (unsigned D = 0; D < NumDwords; ++D) {
        llvm::MCRegister TmpSgpr = getSGPR32(TRI, 30 + D);
        BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::S_LOAD_DWORD_IMM),
                TmpSgpr)
            .addReg(KernargPtrReg)
            .addImm(Fld.Offset + D * 4)
            .addImm(0);
      }
      BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::S_WAITCNT)).addImm(0);
      for (unsigned D = 0; D < NumDwords; ++D) {
        llvm::MCRegister DstVGPR;
        if (NumDwords == 1) {
          DstVGPR = Reg;
        } else {
          DstVGPR = TRI.getSubReg(Reg, AMDGPU::sub0 + D);
        }
        BuildMI(*BB, BB->end(), DL, TII.get(MovVGPR32), DstVGPR)
            .addReg(getSGPR32(TRI, 30 + D));
      }
    }
  }

  // Load output pointer.
  BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::S_LOAD_DWORDX2_IMM),
          OutPtrSGPR)
      .addReg(KernargPtrReg)
      .addImm(Layout.OutputPtrOffset)
      .addImm(0);

  BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::S_WAITCNT)).addImm(0);

  // ====================================================================
  // 3. Copy the instruction under test into this kernel's BB.
  // ====================================================================
  MF.CloneMachineInstrBundle(*BB, BB->end(), *MICtx.MI);

  // ====================================================================
  // 4. Store outputs to the output buffer.
  // ====================================================================
  llvm::MCRegister OutPtrLoSGPR = getSGPR32(TRI, 6);
  llvm::MCRegister OutPtrHiSGPR = getSGPR32(TRI, 7);
  BuildMI(*BB, BB->end(), DL, TII.get(MovVGPR32), getVGPR32(TRI, 0))
      .addReg(OutPtrLoSGPR);
  BuildMI(*BB, BB->end(), DL, TII.get(MovVGPR32), getVGPR32(TRI, 1))
      .addReg(OutPtrHiSGPR);

  unsigned StoreVGPRIdx = 2;
  for (unsigned I = 0; I < MICtx.OutputRegs.size(); ++I) {
    llvm::MCRegister OutReg = MICtx.OutputRegs[I];
    const auto *RC = TRI.getMinimalPhysRegClass(OutReg);
    unsigned Bits = TRI.getRegSizeInBits(*RC);
    unsigned NumDwords = (Bits + 31) / 32;
    const auto &OF = Layout.OutputFields[I];

    for (unsigned D = 0; D < NumDwords; ++D) {
      llvm::MCRegister SrcSub =
          NumDwords == 1 ? OutReg
                         : TRI.getSubReg(OutReg, AMDGPU::sub0 + D);
      llvm::MCRegister DstVGPR = getVGPR32(TRI, StoreVGPRIdx);

      if (TRI.isSGPRPhysReg(SrcSub))
        BuildMI(*BB, BB->end(), DL, TII.get(MovVGPR32), DstVGPR)
            .addReg(SrcSub);
      else
        BuildMI(*BB, BB->end(), DL, TII.get(llvm::TargetOpcode::COPY), DstVGPR)
            .addReg(SrcSub);

      BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::FLAT_STORE_DWORD))
          .addReg(OutPtrVGPR)
          .addReg(DstVGPR)
          .addImm(OF.Offset + D * 4)
          .addImm(0);
      ++StoreVGPRIdx;
    }
  }

  // Store implicit defs (SCC, VCC).
  unsigned ImplFieldIdx = MICtx.OutputRegs.size();
  for (llvm::MCRegister Reg : Profile.ImplicitDefs) {
    if (ImplFieldIdx >= Layout.OutputFields.size())
      break;
    const auto &OF = Layout.OutputFields[ImplFieldIdx++];
    llvm::StringRef RName = TRI.getName(Reg);

    if (RName == "SCC") {
      llvm::MCRegister Tmp = getSGPR32(TRI, 40);
      llvm::MCRegister Dst = getVGPR32(TRI, StoreVGPRIdx);
      BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::S_CSELECT_B32), Tmp)
          .addImm(1).addImm(0);
      BuildMI(*BB, BB->end(), DL, TII.get(MovVGPR32), Dst).addReg(Tmp);
      BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::FLAT_STORE_DWORD))
          .addReg(OutPtrVGPR).addReg(Dst).addImm(OF.Offset).addImm(0);
      ++StoreVGPRIdx;
    } else if (RName.starts_with("VCC")) {
      llvm::MCRegister TmpLo = getSGPR32(TRI, 40);
      llvm::MCRegister TmpHi = getSGPR32(TRI, 41);
      BuildMI(*BB, BB->end(), DL, TII.get(llvm::TargetOpcode::COPY), TmpLo)
          .addReg(AMDGPU::VCC_LO);
      BuildMI(*BB, BB->end(), DL, TII.get(llvm::TargetOpcode::COPY), TmpHi)
          .addReg(AMDGPU::VCC_HI);
      llvm::MCRegister D0 = getVGPR32(TRI, StoreVGPRIdx++);
      llvm::MCRegister D1 = getVGPR32(TRI, StoreVGPRIdx++);
      BuildMI(*BB, BB->end(), DL, TII.get(MovVGPR32), D0).addReg(TmpLo);
      BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::FLAT_STORE_DWORD))
          .addReg(OutPtrVGPR).addReg(D0).addImm(OF.Offset).addImm(0);
      BuildMI(*BB, BB->end(), DL, TII.get(MovVGPR32), D1).addReg(TmpHi);
      BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::FLAT_STORE_DWORD))
          .addReg(OutPtrVGPR).addReg(D1).addImm(OF.Offset + 4).addImm(0);
    }
  }

  // ====================================================================
  // 5. Epilogue.
  // ====================================================================
  BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::S_WAITCNT)).addImm(0);
  BuildMI(*BB, BB->end(), DL, TII.get(AMDGPU::S_ENDPGM)).addImm(0);

  return std::move(KCtx);
}

//===----------------------------------------------------------------------===//
// ELF emission
//===----------------------------------------------------------------------===//

llvm::Expected<llvm::SmallVector<char, 0>>
MachineKernelBuilder::emitToELF(KernelMFContext &KCtx) {
  llvm::SmallVector<char, 0> ObjBuffer;
  llvm::raw_svector_ostream ObjStream(ObjBuffer);
  llvm::legacy::PassManager PM;

  if (TM.addPassesToEmitFile(PM, ObjStream, nullptr,
                             llvm::CodeGenFileType::ObjectFile,
                             /*DisableVerify=*/true, KCtx.MMI.get()))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Target does not support object emission");
  PM.run(*KCtx.Mod);
  return ObjBuffer;
}

} // namespace luthier::test
