//===-- SOP.cpp - SOP1 / SOP2 / SOPK / SOPC scalar-ALU ref-kernel builder -===//
//
// Builds the reference kernel for the scalar ALU family. Everything is scalar:
// sdst and src operands are SGPRs, loaded straight from kernarg (P1) and stored
// via an SGPR->VGPR move (E2). SCC is the scalar analog of VCC:
//   * seeded (P4) for ops that read it (S_ADDC_U32, S_CSELECT_B32) by deriving
//     it from a kernarg dword: S_AND_B32 stage, stage, 1 sets SCC = bit 0.
//   * captured (E3) for ops that write it: S_CSELECT_B32 t, 1, 0 -> t = SCC.
// SOPK carries a 16-bit inline literal (simm16), encoded as a fixed constant;
// its tied src0=sdst accumulator (S_ADDK/S_MULK) is pre-seeded (P5).
//
// One work-item (lane 0). Only the SCC bit is meaningful.
//
//===----------------------------------------------------------------------===//
#include "InstructionBuilders.h"
#include "RefKernelSupport.h"

#include <SIDefines.h>
#include <SIInstrInfo.h>

#include <llvm/MC/MCInstrDesc.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier::test {

namespace {

/// Fixed 16-bit signed literal encoded into SOPK simm16 operands. Part of the
/// instruction encoding, so constant per kernel (not fuzzed per dispatch).
constexpr int64_t FixedSimm16 = 100;
/// SGPR holding the captured SCC value (S_CSELECT result), disjoint from the
/// operand ranges (see RefKernelSupport.h).
constexpr unsigned SccCaptureSGPRIdx = StageSGPRBase + 4; // s12

const char *sopClass(const llvm::MCInstrDesc &D) {
  uint64_t F = D.TSFlags;
  if (F & llvm::SIInstrFlags::SOP1) return "SOP1";
  if (F & llvm::SIInstrFlags::SOP2) return "SOP2";
  if (F & llvm::SIInstrFlags::SOPK) return "SOPK";
  if (F & llvm::SIInstrFlags::SOPC) return "SOPC";
  return "SOP";
}

} // namespace

llvm::Expected<KernelMFContext> buildSOP(llvm::TargetMachine &TM,
                                         const InstrProfile &Profile,
                                         KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCRegisterInfo &MRI = TM.getMCRegisterInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);

  // Guards: SCC is the only special register (operand width handled per-dword).
  for (llvm::MCRegister R : Profile.ImplicitDefs)
    if (llvm::StringRef(MRI.getName(R)) != "SCC")
      return makeError(Profile.Name + ": unsupported implicit def " +
                       MRI.getName(R));
  for (llvm::MCRegister R : Profile.ImplicitUses)
    if (llvm::StringRef(MRI.getName(R)) != "SCC")
      return makeError(Profile.Name + ": unsupported implicit use " +
                       MRI.getName(R));

  bool SCCIn = implicitContains(Profile, false, "SCC", MRI);
  bool SCCOut = implicitContains(Profile, true, "SCC", MRI);

  // Classify each MI operand slot (MC-only). SOP immediates are simm16. Each
  // 32-bit component of a wide SGPR operand becomes its own kernarg / output
  // field.
  struct RawOp {
    enum { Def, Use, Tied, Imm } K;
    int16_t RCID = -1;
    int TiedTo = -1;
    unsigned Dwords = 1;
  };
  llvm::SmallVector<RawOp, 4> Raw;
  for (unsigned I = 0; I < Desc.getNumOperands(); ++I) {
    const llvm::MCOperandInfo &OI = Desc.operands()[I];
    if (OI.RegClass < 0) {
      Raw.push_back({RawOp::Imm});
      continue;
    }
    unsigned Bits = MRI.getRegClass(OI.RegClass).getSizeInBits();
    if (Bits % 32 != 0 || Bits > 128)
      return makeError(llvm::formatv(
          "{0}: unsupported {1}-bit operand (op {2})", Profile.Name, Bits, I));
    unsigned Dwords = Bits / 32;
    if (I < Desc.getNumDefs())
      Raw.push_back({RawOp::Def, OI.RegClass, -1, Dwords});
    else {
      int Tied = Desc.getOperandConstraint(I, llvm::MCOI::TIED_TO);
      Raw.push_back(
          {Tied >= 0 ? RawOp::Tied : RawOp::Use, OI.RegClass, Tied, Dwords});
    }
  }
  auto defIsTied = [&](unsigned Slot) {
    for (const RawOp &R : Raw)
      if (R.K == RawOp::Tied && R.TiedTo == (int)Slot)
        return true;
    return false;
  };

  // Kernarg / output layout. One 4-byte field per 32-bit component.
  Layout = KernargLayout{};
  uint32_t KOff = 8, OOff = 0;
  auto addInput = [&](const std::string &Name, unsigned Dwords) {
    uint32_t Base = KOff;
    for (unsigned D = 0; D < Dwords; ++D, KOff += 4)
      Layout.Inputs.push_back(
          {KOff, 4, Dwords > 1 ? llvm::formatv("{0}.{1}", Name, D).str() : Name});
    return Base;
  };
  auto addOutput = [&](const std::string &Name, unsigned Dwords) {
    uint32_t Base = OOff;
    for (unsigned D = 0; D < Dwords; ++D, OOff += 4)
      Layout.Outputs.push_back(
          {OOff, 4, Dwords > 1 ? llvm::formatv("{0}.{1}", Name, D).str() : Name});
    return Base;
  };
  llvm::SmallVector<uint32_t, 4> SlotInOff(Raw.size(), UINT32_MAX);
  llvm::SmallVector<uint32_t, 4> SlotOutOff(Raw.size(), UINT32_MAX);
  for (unsigned I = 0; I < Raw.size(); ++I) {
    if (Raw[I].K == RawOp::Def) {
      SlotOutOff[I] = addOutput(llvm::formatv("def{0}", I).str(), Raw[I].Dwords);
      if (defIsTied(I))
        SlotInOff[I] =
            addInput(llvm::formatv("seed{0}", I).str(), Raw[I].Dwords);
    } else if (Raw[I].K == RawOp::Use) {
      SlotInOff[I] = addInput(llvm::formatv("src{0}", I).str(), Raw[I].Dwords);
    }
  }
  uint32_t SccInOff = SCCIn ? addInput("scc_in", 1) : UINT32_MAX;
  uint32_t SccOutOff = SCCOut ? addOutput("scc_out", 1) : UINT32_MAX;
  Layout.TotalSize = KOff;
  Layout.OutputBufSize = OOff;

  KernelMFContext KCtx;
  KCtx.KernelName = MachineKernelBuilder::getKernelName(Profile);
  auto ScafOrErr = setupScaffold(TM, KCtx, Layout.Inputs.size());
  if (!ScafOrErr)
    return ScafOrErr.takeError();
  Scaffold S = *ScafOrErr;
  const llvm::SIInstrInfo &TII = *S.TII;
  const llvm::SIRegisterInfo &TRI = *S.TRI;

  if (TII.pseudoToMCOpcode(Profile.Opcode) < 0)
    return makeError(Profile.Name +
                     ": no MC encoding for this subtarget (pseudo needs "
                     "expansion)");

  // Assign SGPRs + build binding steps. pick32BitReg falls back to an SGPR for
  // the scalar (SReg_32) classes; the dummy VGPR cursor is never consumed.
  unsigned DummyVGPR = 200;
  unsigned NextInSGPR = ScalarInSGPRBase, NextOutSGPR = ScalarOutSGPRBase;
  llvm::SmallVector<llvm::MCRegister, 4> SlotReg(Raw.size());
  for (unsigned I = 0; I < Raw.size(); ++I) {
    if (Raw[I].K != RawOp::Def)
      continue;
    unsigned Dw = Raw[I].Dwords;
    llvm::MCRegister R = allocOperandReg(TRI, TRI.getRegClass(Raw[I].RCID),
                                         DummyVGPR, NextOutSGPR);
    if (!R || isVGPRTuple(TRI, R, Dw))
      return makeError(Profile.Name + ": no scalar physreg for a def operand");
    SlotReg[I] = R;
    KCtx.Operands.push_back({OperandBinding::DefReg, R, 0, I});
    RegBinding B;
    B.Role = RegBinding::StoreOutput; // E2 (scalar): SGPR -> VGPR -> store.
    B.Reg = R;
    B.Dwords = Dw;
    B.IsVGPR = false;
    B.OutputOffset = SlotOutOff[I];
    B.Note = llvm::formatv("E2 store def op{0} ({1}dw) -> out+{2}", I, Dw,
                           B.OutputOffset);
    KCtx.Epilog.push_back(B);
    if (SlotInOff[I] != UINT32_MAX) {
      RegBinding Sd;
      Sd.Role = RegBinding::SeedTiedDef;
      Sd.Reg = R;
      Sd.Dwords = Dw;
      Sd.IsVGPR = false;
      Sd.KernargOffset = SlotInOff[I];
      Sd.Note = llvm::formatv("P5 seed tied def op{0} ({1}dw) <- ka+{2}", I, Dw,
                              Sd.KernargOffset);
      KCtx.Prolog.push_back(Sd);
    }
  }
  for (unsigned I = 0; I < Raw.size(); ++I) {
    switch (Raw[I].K) {
    case RawOp::Def:
      break;
    case RawOp::Imm:
      KCtx.Operands.push_back({OperandBinding::Imm, {}, FixedSimm16, I});
      break;
    case RawOp::Tied:
      SlotReg[I] = SlotReg[Raw[I].TiedTo];
      KCtx.Operands.push_back({OperandBinding::TiedUse, SlotReg[I], 0, I});
      break;
    case RawOp::Use: {
      unsigned Dw = Raw[I].Dwords;
      llvm::MCRegister R = allocOperandReg(TRI, TRI.getRegClass(Raw[I].RCID),
                                           DummyVGPR, NextInSGPR);
      if (!R || isVGPRTuple(TRI, R, Dw))
        return makeError(Profile.Name + ": no scalar physreg for a use operand");
      SlotReg[I] = R;
      KCtx.Operands.push_back({OperandBinding::UseReg, R, 0, I});
      RegBinding B;
      B.Role = RegBinding::LoadInputSGPR; // P1: kernarg dword -> SGPR.
      B.Reg = R;
      B.Dwords = Dw;
      B.IsVGPR = false;
      B.KernargOffset = SlotInOff[I];
      B.Note = llvm::formatv("P1 src op{0} ({1}dw) <- ka+{2}", I, Dw,
                             B.KernargOffset);
      KCtx.Prolog.push_back(B);
      break;
    }
    }
  }
  if (SCCIn) {
    RegBinding B;
    B.Role = RegBinding::SetSpecial;
    B.Special = SpecialReg::SCC;
    B.FromKernarg = true;
    B.KernargOffset = SccInOff;
    B.Note = llvm::formatv("P4 seed SCC <- ka+{0} (bit 0)", SccInOff);
    KCtx.Prolog.push_back(B);
  }
  if (SCCOut) {
    RegBinding B;
    B.Role = RegBinding::CaptureSpecial;
    B.Special = SpecialReg::SCC;
    B.OutputOffset = SccOutOff;
    B.Note = llvm::formatv("E3 capture SCC -> out+{0}", SccOutOff);
    KCtx.Epilog.push_back(B);
  }

  {
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [" << sopClass(Desc) << (SCCIn ? " scc_in" : "")
       << (SCCOut ? " scc_out" : "") << "]\n  prolog:\n";
    for (const RegBinding &B : KCtx.Prolog)
      OS << "    " << B.Note << "\n";
    OS << "  epilog:\n";
    for (const RegBinding &B : KCtx.Epilog)
      OS << "    " << B.Note << "\n";
  }

  //=== Emit MIR ======================================================//
  EmitState E{S.BB, TII, S.KernargReg, S.OutPtrReg, {}};
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          S.OutPtrReg)
      .addReg(S.KernargReg)
      .addImm(Layout.OutputPtrOffset)
      .addImm(0);

  // Phase A: scalar loads straight into the operand / seed / SCC-stage SGPRs.
  llvm::MCRegister SccStage = sgpr32(StageSGPRBase);
  for (const RegBinding &B : KCtx.Prolog) {
    if (B.Role == RegBinding::LoadInputSGPR ||
        B.Role == RegBinding::SeedTiedDef) {
      for (unsigned D = 0; D < B.Dwords; ++D)
        emitScalarLoad(E, subDword(TRI, B.Reg, D, B.Dwords),
                       B.KernargOffset + D * 4);
    } else if (B.Role == RegBinding::SetSpecial &&
               B.Special == SpecialReg::SCC)
      emitScalarLoad(E, SccStage, B.KernargOffset);
  }
  emitWaitcnt(E);

  // SCC seed must be the last thing before the MI (nothing may clobber SCC).
  if (SCCIn)
    BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_AND_B32), SccStage)
        .addReg(SccStage)
        .addImm(1);

  // The instruction under test.
  llvm::MachineInstrBuilder MIB = BuildMI(*S.BB, S.BB->end(), E.DL, Desc);
  for (const OperandBinding &Op : KCtx.Operands) {
    switch (Op.Kind) {
    case OperandBinding::DefReg:
      MIB.addDef(Op.Reg);
      break;
    case OperandBinding::UseReg:
    case OperandBinding::TiedUse:
      MIB.addReg(Op.Reg);
      break;
    case OperandBinding::Imm:
      MIB.addImm(Op.ImmValue);
      break;
    }
  }
  KCtx.MI = MIB.getInstr();

  // Capture SCC immediately after the MI (before any store touches state).
  llvm::MCRegister SccCap = sgpr32(SccCaptureSGPRIdx);
  if (SCCOut)
    BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_CSELECT_B32),
            SccCap)
        .addImm(1)
        .addImm(0);

  // Epilog: move each captured SGPR component to a VGPR and store. One staging
  // VGPR is reused since each store immediately follows its V_MOV.
  llvm::MCRegister ZeroVGPR = vgpr32(0);
  emitVMovImm(E, ZeroVGPR, 0);
  llvm::MCRegister StoreVGPR = vgpr32(StoreStageVGPRBase);
  for (const RegBinding &B : KCtx.Epilog) {
    if (B.Role == RegBinding::StoreOutput) {
      for (unsigned D = 0; D < B.Dwords; ++D) {
        emitVMovReg(E, StoreVGPR, subDword(TRI, B.Reg, D, B.Dwords));
        emitGlobalStore(E, StoreVGPR, ZeroVGPR, B.OutputOffset + D * 4);
      }
    } else if (B.Role == RegBinding::CaptureSpecial &&
               B.Special == SpecialReg::SCC) {
      emitVMovReg(E, StoreVGPR, SccCap);
      emitGlobalStore(E, StoreVGPR, ZeroVGPR, B.OutputOffset);
    }
  }
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

} // namespace luthier::test
