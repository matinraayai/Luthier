//===-- VOP.cpp - VOP1/2/C/3/3P (VALU) reference-kernel builder ----------===//
//
// Builds the reference kernel for the VALU family, which all share one prolog/
// epilog shape with scalar-or-vector src0/src1/src2 inputs:
//   * VOP1  - single source: vdst, src0.
//   * VOP2  - vdst, src0, src1 (+ optional literal / tied src2 / VCC carry).
//   * VOPC  - compare: src0, src1 -> a scalar VCC mask (implicit VCC def, no
//             explicit VGPR output); captured through the VCC path (E3).
//   * VOP3  - vdst[, sdst], src0..src2, with interleaved modifier immediates
//             (src_modifiers) and trailing clamp/omod.
//   * VOP3P - packed math: vdst, src0..src2 with op_sel/op_sel_hi/neg_lo/neg_hi.
// Modifiers/controls are assumed disabled: every non-literal immediate operand
// is encoded as 0, so it does not influence the computation. A real encoded
// literal (KIMM, e.g. V_MADAK's K) is still fuzzed as a fixed constant.
//
// Per operand: seed register inputs (P2), encode a KIMM literal (P7) or a
// disabled modifier (0), pre-seed a tied src2=dst accumulator (P5), seed an
// implicit VCC carry-in (P4); run the MI; capture the register output (E1) and
// any VCC result (E3).
//
// The builder is operand-generic; classifyVOP2() only affects the auditable log
// (VOP2 subgroup), and valuClass() labels the encoding family.
//
//===----------------------------------------------------------------------===//
#include "InstructionBuilders.h"
#include "RefKernelSupport.h"

#include <SIDefines.h>
#include <SIInstrInfo.h>

#include <cstdlib>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier::test {

const char *toString(VOP2Subgroup S) {
  switch (S) {
  case VOP2Subgroup::NotVOP2:        return "NotVOP2";
  case VOP2Subgroup::Sg1_DstSrc01:   return "Sg1(DST0,SRC0,SRC1)";
  case VOP2Subgroup::Sg2_Literal:    return "Sg2(+simm32 literal)";
  case VOP2Subgroup::Sg2_Cndmask:    return "Sg2(+VCC use / cndmask)";
  case VOP2Subgroup::Sg3_CarryInOut: return "Sg3(VCC in+out / addc,subb)";
  case VOP2Subgroup::CarryOutOnly:   return "CarryOut(VCC def only)";
  case VOP2Subgroup::Tied:           return "Tied(src2=dst / mac,fmac)";
  }
  return "?";
}

/// Name of the VALU encoding family, for the auditable log.
static const char *valuClass(const llvm::MCInstrDesc &Desc) {
  uint64_t F = Desc.TSFlags;
  if (F & llvm::SIInstrFlags::VOP1)
    return "VOP1";
  if (F & llvm::SIInstrFlags::VOP2)
    return "VOP2";
  if (F & llvm::SIInstrFlags::VOPC)
    return "VOPC";
  if (F & llvm::SIInstrFlags::VOP3)
    return "VOP3";
  return "VALU";
}

/// Classify over MC facts: tied constraint -> Tied; implicit VCC use/def ->
/// carry / cndmask; an immediate explicit operand -> literal; else plain.
/// Returns NotVOP2 for non-VOP2 encodings (e.g. VOP1), which the builder still
/// handles generically.
static VOP2Subgroup classifyVOP2(const llvm::MCInstrDesc &Desc,
                                 const InstrProfile &P,
                                 const llvm::MCRegisterInfo &MRI) {
  if (!(Desc.TSFlags & llvm::SIInstrFlags::VOP2))
    return VOP2Subgroup::NotVOP2;
  for (unsigned I = Desc.getNumDefs(); I < Desc.getNumOperands(); ++I)
    if (Desc.getOperandConstraint(I, llvm::MCOI::TIED_TO) >= 0)
      return VOP2Subgroup::Tied;
  bool VCCUse = implicitContains(P, false, "VCC", MRI);
  bool VCCDef = implicitContains(P, true, "VCC", MRI);
  if (VCCUse && VCCDef)
    return VOP2Subgroup::Sg3_CarryInOut;
  if (VCCUse)
    return VOP2Subgroup::Sg2_Cndmask;
  if (VCCDef)
    return VOP2Subgroup::CarryOutOnly;
  for (const auto &In : P.Inputs)
    if (In.IsImm)
      return VOP2Subgroup::Sg2_Literal;
  return VOP2Subgroup::Sg1_DstSrc01;
}

llvm::Expected<KernelMFContext> buildVOP(llvm::TargetMachine &TM,
                                         const InstrProfile &Profile,
                                         KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCRegisterInfo &MRI = TM.getMCRegisterInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);

  for (llvm::MCRegister R : Profile.ImplicitDefs)
    if (llvm::StringRef(MRI.getName(R)) != "VCC")
      return makeError(Profile.Name + ": unsupported implicit def " +
                       MRI.getName(R));
  for (llvm::MCRegister R : Profile.ImplicitUses) {
    llvm::StringRef Nm = MRI.getName(R);
    if (Nm != "VCC" && Nm != "EXEC" && Nm != "MODE")
      return makeError(Profile.Name + ": unsupported implicit use " + Nm.str());
  }

  VOP2Subgroup Sub = classifyVOP2(Desc, Profile, MRI);
  bool VCCIn = implicitContains(Profile, false, "VCC", MRI);
  bool VCCOut = implicitContains(Profile, true, "VCC", MRI);

  // Classify each MI operand slot (MC-only). Immediate operands split into two
  // kinds by their AMDGPU operand type:
  //   * KIMM32/KIMM16  - a real encoded literal (e.g. V_MADAK's K); fuzzed as a
  //                      fixed constant.
  //   * anything else  - a VOP3/VOP3P modifier or control (src_modifiers,
  //                      clamp, omod, op_sel, op_sel_hi, neg_lo, neg_hi, ...).
  //                      We disable modifiers, i.e. encode 0, so they do not
  //                      influence the computation.
  // Register operands can be any width (32-bit and wider register tuples); each
  // 32-bit component becomes its own kernarg / output field, and the emitter
  // seeds / captures each component through its sub-register.
  struct RawOp {
    enum { Def, Use, Tied, Imm } K;
    int16_t RCID = -1;
    int TiedTo = -1;
    int64_t ImmVal = 0;
    unsigned Dwords = 1;
  };
  // VOP3P: the per-source op_sel_hi bit (which lane-1 source half to read) is
  // encoded from bit 3 of each srcN_modifiers operand (SISrcMods::OP_SEL_1), NOT
  // from the standalone op_sel_hi operand (that one is asm/disasm only and the
  // encoder ignores it). For a normal packed op the high lane must read the high
  // half, i.e. OP_SEL_1 set on every source; leaving src_modifiers 0 makes the
  // high lane read the low half instead -- a different, non-standard op.
  // Only the truly-packed V_PK_* ops (2xf16 SIMD) default to op_sel_hi=1 (high
  // lane reads the high f16). The mixed-precision MIX / DOT VOP3P ops use
  // op_sel_hi differently and default to 0, so restrict the fix to V_PK_*.
  const bool IsPacked =
      (Desc.TSFlags & llvm::SIInstrFlags::VOP3P) && Profile.Name.contains("_PK_");
  auto nIdx = [&](llvm::AMDGPU::OpName N) {
    return llvm::AMDGPU::getNamedOperandIdx(Profile.Opcode, N);
  };
  int SrcMod[3] = {nIdx(llvm::AMDGPU::OpName::src0_modifiers),
                   nIdx(llvm::AMDGPU::OpName::src1_modifiers),
                   nIdx(llvm::AMDGPU::OpName::src2_modifiers)};
  auto isSrcMod = [&](int I) {
    return IsPacked && (I == SrcMod[0] || I == SrcMod[1] || I == SrcMod[2]);
  };

  llvm::SmallVector<RawOp, 8> Raw;
  for (unsigned I = 0; I < Desc.getNumOperands(); ++I) {
    const llvm::MCOperandInfo &OI = Desc.operands()[I];
    if (OI.RegClass < 0) {
      bool IsKImm = OI.OperandType == llvm::AMDGPU::OPERAND_KIMM32 ||
                    OI.OperandType == llvm::AMDGPU::OPERAND_KIMM16;
      int64_t ImmV = IsKImm ? FixedLiteral : 0;
      if (isSrcMod((int)I))
        ImmV = 1 << 3; // SISrcMods::OP_SEL_1 = op_sel_hi bit (normal packed)
      Raw.push_back({RawOp::Imm, -1, -1, ImmV, 1});
      continue;
    }
    unsigned Bits = MRI.getRegClass(OI.RegClass).getSizeInBits();
    if (Bits % 32 != 0 || Bits > 128)
      return makeError(llvm::formatv(
          "{0}: unsupported {1}-bit operand (op {2})", Profile.Name, Bits, I));
    unsigned Dwords = Bits / 32;
    if (I < Desc.getNumDefs())
      Raw.push_back({RawOp::Def, OI.RegClass, -1, 0, Dwords});
    else {
      int Tied = Desc.getOperandConstraint(I, llvm::MCOI::TIED_TO);
      Raw.push_back(
          {Tied >= 0 ? RawOp::Tied : RawOp::Use, OI.RegClass, Tied, 0, Dwords});
    }
  }
  auto defIsTied = [&](unsigned Slot) {
    for (const RawOp &R : Raw)
      if (R.K == RawOp::Tied && R.TiedTo == (int)Slot)
        return true;
    return false;
  };

  // Kernarg / output layout. Each 32-bit component of a wide operand is a
  // separate 4-byte field, so the driver keeps filling one dword per field and
  // needs no notion of operand width.
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
  llvm::SmallVector<uint32_t, 6> SlotInOff(Raw.size(), UINT32_MAX);
  llvm::SmallVector<uint32_t, 6> SlotOutOff(Raw.size(), UINT32_MAX);
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
  uint32_t VccInOff = VCCIn ? addInput("vcc_in", 1) : UINT32_MAX;
  uint32_t VccOutOff = VCCOut ? addOutput("vcc_out", 1) : UINT32_MAX;
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
                     "expansion; not emittable via the reference path)");

  // Assign operand registers + build binding steps.
  unsigned NextInVGPR = InputVGPRBase, NextOutVGPR = OutputVGPRBase;
  unsigned NextInSGPR = ScalarInSGPRBase, NextOutSGPR = ScalarOutSGPRBase;
  llvm::SmallVector<llvm::MCRegister, 6> SlotReg(Raw.size());
  for (unsigned I = 0; I < Raw.size(); ++I) {
    if (Raw[I].K != RawOp::Def)
      continue;
    unsigned Dw = Raw[I].Dwords;
    llvm::MCRegister R = allocOperandReg(TRI, TRI.getRegClass(Raw[I].RCID),
                                         NextOutVGPR, NextOutSGPR);
    if (!R)
      return makeError(Profile.Name + ": no physreg for a def operand");
    SlotReg[I] = R;
    KCtx.Operands.push_back({OperandBinding::DefReg, R, 0, I});
    RegBinding B;
    B.Role = RegBinding::StoreOutput;
    B.Reg = R;
    B.Dwords = Dw;
    B.IsVGPR = isVGPRTuple(TRI, R, Dw);
    B.OutputOffset = SlotOutOff[I];
    B.Note = llvm::formatv("E1 store def op{0} ({1}dw) -> out+{2}", I, Dw,
                           B.OutputOffset);
    KCtx.Epilog.push_back(B);
    if (SlotInOff[I] != UINT32_MAX) {
      RegBinding Sd;
      Sd.Role = RegBinding::SeedTiedDef;
      Sd.Reg = R;
      Sd.Dwords = Dw;
      Sd.IsVGPR = isVGPRTuple(TRI, R, Dw);
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
      KCtx.Operands.push_back({OperandBinding::Imm, {}, Raw[I].ImmVal, I});
      break;
    case RawOp::Tied:
      SlotReg[I] = SlotReg[Raw[I].TiedTo];
      KCtx.Operands.push_back({OperandBinding::TiedUse, SlotReg[I], 0, I});
      break;
    case RawOp::Use: {
      unsigned Dw = Raw[I].Dwords;
      llvm::MCRegister R = allocOperandReg(TRI, TRI.getRegClass(Raw[I].RCID),
                                           NextInVGPR, NextInSGPR);
      if (!R)
        return makeError(Profile.Name + ": no physreg for a use operand (op " +
                         llvm::Twine(I) + ")");
      SlotReg[I] = R;
      KCtx.Operands.push_back({OperandBinding::UseReg, R, 0, I});
      RegBinding B;
      bool V = isVGPRTuple(TRI, R, Dw);
      B.Role = V ? RegBinding::LoadInputVGPR : RegBinding::LoadInputSGPR;
      B.Reg = R;
      B.Dwords = Dw;
      B.IsVGPR = V;
      B.KernargOffset = SlotInOff[I];
      B.Note = llvm::formatv("{0} src op{1} ({2}dw) <- ka+{3}", V ? "P2" : "P1",
                             I, Dw, B.KernargOffset);
      KCtx.Prolog.push_back(B);
      break;
    }
    }
  }
  if (VCCIn) {
    RegBinding B;
    B.Role = RegBinding::SetSpecial;
    B.Special = SpecialReg::VCC;
    B.FromKernarg = true;
    B.KernargOffset = VccInOff;
    B.Note = llvm::formatv("P4 seed VCC <- ka+{0}", VccInOff);
    KCtx.Prolog.push_back(B);
  }
  if (VCCOut) {
    RegBinding B;
    B.Role = RegBinding::CaptureSpecial;
    B.Special = SpecialReg::VCC;
    B.OutputOffset = VccOutOff;
    B.Note = llvm::formatv("E3 capture VCC -> out+{0}", VccOutOff);
    KCtx.Epilog.push_back(B);
  }

  {
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [" << valuClass(Desc);
    if (Sub != VOP2Subgroup::NotVOP2)
      OS << ": " << toString(Sub);
    OS << "]\n  prolog:\n";
    for (const RegBinding &B : KCtx.Prolog)
      OS << "    " << B.Note << "\n";
    OS << "  epilog:\n";
    for (const RegBinding &B : KCtx.Epilog)
      OS << "    " << B.Note << "\n";
  }

  // Emit MIR.
  EmitState E{S.BB, TII, S.KernargReg, S.OutPtrReg, {}};
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          S.OutPtrReg)
      .addReg(S.KernargReg)
      .addImm(Layout.OutputPtrOffset)
      .addImm(0);

  auto needsStage = [](const RegBinding &B) {
    return B.Role == RegBinding::LoadInputVGPR ||
           B.Role == RegBinding::SeedTiedDef ||
           (B.Role == RegBinding::SetSpecial && B.FromKernarg);
  };
  // Every staged component occupies a distinct staging SGPR (s8..s15) until its
  // phase-B V_MOV consumes it, so the concurrent staged-dword total must fit.
  unsigned StageBudget = ScalarInSGPRBase - StageSGPRBase;
  unsigned StageNeed = 0;
  for (const RegBinding &B : KCtx.Prolog)
    if (needsStage(B))
      StageNeed += (B.Role == RegBinding::SetSpecial) ? 1 : B.Dwords;
  if (StageNeed > StageBudget)
    return makeError(llvm::formatv(
        "{0}: needs {1} staging SGPRs but only {2} are reserved", Profile.Name,
        StageNeed, StageBudget));

  unsigned Stage = 0;
  for (const RegBinding &B : KCtx.Prolog) {
    if (B.Role == RegBinding::LoadInputSGPR)
      for (unsigned D = 0; D < B.Dwords; ++D)
        emitScalarLoad(E, subDword(TRI, B.Reg, D, B.Dwords),
                       B.KernargOffset + D * 4);
    else if (needsStage(B)) {
      unsigned N = (B.Role == RegBinding::SetSpecial) ? 1 : B.Dwords;
      for (unsigned D = 0; D < N; ++D)
        emitScalarLoad(E, sgpr32(StageSGPRBase + Stage++), B.KernargOffset + D * 4);
    }
  }
  emitWaitcnt(E);

  Stage = 0;
  for (const RegBinding &B : KCtx.Prolog) {
    switch (B.Role) {
    case RegBinding::LoadInputSGPR:
      break;
    case RegBinding::LoadInputVGPR:
    case RegBinding::SeedTiedDef:
      for (unsigned D = 0; D < B.Dwords; ++D)
        emitVMovReg(E, subDword(TRI, B.Reg, D, B.Dwords),
                    sgpr32(StageSGPRBase + Stage++));
      break;
    case RegBinding::SetSpecial: {
      llvm::MCRegister Src = sgpr32(StageSGPRBase + Stage++);
      if (B.Special == SpecialReg::VCC) {
        BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
                llvm::AMDGPU::VCC_LO)
            .addReg(Src);
        // VCC is 64-bit only in wave64; on wave32 it is just VCC_LO.
        if (waveSize(TM) == 64)
          BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
                  llvm::AMDGPU::VCC_HI)
              .addImm(0);
      }
      break;
    }
    default:
      break;
    }
  }

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
  // Tied operands are auto-tied by the MI builder from the TIED_TO constraint.

  llvm::MCRegister ZeroVGPR = vgpr32(0);
  emitVMovImm(E, ZeroVGPR, 0);
  // A single staging VGPR is reused per SGPR-sourced component: each store is
  // emitted right after its V_MOV, so the value is consumed immediately.
  llvm::MCRegister StoreVGPR = vgpr32(StoreStageVGPRBase);
  for (const RegBinding &B : KCtx.Epilog) {
    if (B.Role == RegBinding::StoreOutput) {
      for (unsigned D = 0; D < B.Dwords; ++D) {
        llvm::MCRegister Comp = subDword(TRI, B.Reg, D, B.Dwords);
        llvm::MCRegister Data = Comp;
        if (!B.IsVGPR) {
          Data = StoreVGPR;
          emitVMovReg(E, Data, Comp);
        }
        emitGlobalStore(E, Data, ZeroVGPR, B.OutputOffset + D * 4);
      }
    } else if (B.Role == RegBinding::CaptureSpecial &&
               B.Special == SpecialReg::VCC) {
      emitVMovReg(E, StoreVGPR, llvm::AMDGPU::VCC_LO);
      emitGlobalStore(E, StoreVGPR, ZeroVGPR, B.OutputOffset);
    }
  }
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

} // namespace luthier::test
