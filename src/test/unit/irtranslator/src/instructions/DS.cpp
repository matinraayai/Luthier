//===-- DS.cpp - DS (LDS) reference-kernel builder --------------*- C++ -*-===//
//
// Classifies a DS opcode over MC facts (named operand presence + mayLoad/
// mayStore + mnemonic) and builds its reference kernel. LDS is not host-
// visible, so each kernel: sets M0, points a constant address VGPR at LDS[0],
// initializes LDS from kernarg (loads/atomics), runs the MI, reads LDS back,
// and stores the captured VGPRs. One work-item (lane 0).
//
//===----------------------------------------------------------------------===//
#include "InstructionBuilders.h"
#include "RefKernelSupport.h"

#include <SIInstrInfo.h>

#include <llvm/MC/MCInstrDesc.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier::test {

const char *toString(DSKind K) {
  switch (K) {
  case DSKind::NotDS:       return "NotDS";
  case DSKind::Load:        return "DS-Load";
  case DSKind::Store:       return "DS-Store";
  case DSKind::AtomicNoRet: return "DS-AtomicNoRet";
  case DSKind::AtomicRet:   return "DS-AtomicRet";
  case DSKind::Paired:      return "DS-Paired(TODO)";
  case DSKind::Permute:     return "DS-Permute(TODO)";
  case DSKind::Unsupported: return "DS-Unsupported";
  }
  return "?";
}

static DSKind classifyDS(const llvm::MCInstrDesc &Desc, const InstrProfile &P) {
  if (!(Desc.TSFlags & llvm::SIInstrFlags::DS))
    return DSKind::NotDS;
  llvm::StringRef N = P.Name;
  if (N.contains("PERMUTE") || N.contains("SWIZZLE"))
    return DSKind::Permute;
  if (N.contains("GWS") || N.contains("ORDERED") || N.contains("APPEND") ||
      N.contains("CONSUME") || N.contains("ADDTID") || N.contains("NOP"))
    return DSKind::Unsupported;

  unsigned Op = P.Opcode;
  auto has = [&](llvm::AMDGPU::OpName Name) {
    return llvm::AMDGPU::getNamedOperandIdx(Op, Name) >= 0;
  };
  bool HasVdst = has(llvm::AMDGPU::OpName::vdst);
  bool HasAddr = has(llvm::AMDGPU::OpName::addr);
  bool HasData1 = has(llvm::AMDGPU::OpName::data1);
  bool HasOffset0 = has(llvm::AMDGPU::OpName::offset0);
  if (HasData1 || HasOffset0)
    return DSKind::Paired; // read2/write2/cmpst/mskor -- future work.
  if (!HasAddr)
    return DSKind::Unsupported;
  bool Ld = Desc.mayLoad(), St = Desc.mayStore();
  if (HasVdst && Ld && !St)
    return DSKind::Load;
  if (!HasVdst && St && !Ld)
    return DSKind::Store;
  if (!HasVdst && Ld && St)
    return DSKind::AtomicNoRet;
  if (HasVdst && Ld && St)
    return DSKind::AtomicRet;
  return DSKind::Unsupported;
}

//===----------------------------------------------------------------------===//
// DS cross-lane permutes (ds_permute / ds_bpermute / ds_swizzle).
//
// These have no LDS storage and no M0 -- they use the LDS crossbar to move data
// *between lanes*, so they are meaningless with one work-item. This builder
// dispatches a full wave (64 lanes) and gives every lane its own inputs and its
// own output slot:
//   * v0 holds the lane id (workitem id x, enabled by the per-lane scaffold);
//   * each lane reads its data0 (and, for [b]permute, its addr) from its own
//     slice of the kernarg buffer via a per-lane GLOBAL_LOAD (kernarg is global-
//     addressable), indexed by lane id;
//   * runs the permute (vdst = crossbar(addr, data0));
//   * writes vdst to its own slice of the output buffer via a per-lane store.
// The driver fills / compares all 64 lanes' fields with no changes.
//===----------------------------------------------------------------------===//
static llvm::Expected<KernelMFContext>
buildDSPermute(llvm::TargetMachine &TM, const InstrProfile &Profile,
               KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);
  unsigned Op = Profile.Opcode;

  auto opIdx = [&](llvm::AMDGPU::OpName N) {
    return llvm::AMDGPU::getNamedOperandIdx(Op, N);
  };
  int VdstIdx = opIdx(llvm::AMDGPU::OpName::vdst);
  int AddrIdx = opIdx(llvm::AMDGPU::OpName::addr);
  int Data0Idx = opIdx(llvm::AMDGPU::OpName::data0);
  if (VdstIdx < 0 || Data0Idx < 0)
    return makeError(Profile.Name + ": unexpected cross-lane operand shape");
  const bool HasAddr = AddrIdx >= 0; // permute/bpermute have addr; swizzle none.

  // Cross-lane permutes operate within one wavefront, so a "full wave" is the
  // subtarget's wave size (32 on RDNA wave32, 64 on CDNA / wave64).
  const unsigned NumLanes = waveSize(TM);
  const unsigned NumInRoles = HasAddr ? 2 : 1; // data0 [, addr]
  const unsigned InStride = NumInRoles * 4;    // 4 or 8 (power of two)
  const unsigned OutStride = 4;
  const uint32_t InBase = 8; // after the output pointer (NumPtrArgs = 1)

  //=== Layout: one field per (lane, role) =========================//
  Layout = KernargLayout{};
  Layout.GridSizeX = NumLanes;
  Layout.WorkgroupSizeX = NumLanes;
  Layout.OutputPtrOffset = 0;
  for (unsigned L = 0; L < NumLanes; ++L) {
    Layout.Inputs.push_back(
        {InBase + L * InStride, 4, llvm::formatv("data0.l{0}", L).str()});
    if (HasAddr)
      Layout.Inputs.push_back(
          {InBase + L * InStride + 4, 4, llvm::formatv("addr.l{0}", L).str()});
  }
  for (unsigned L = 0; L < NumLanes; ++L)
    Layout.Outputs.push_back(
        {L * OutStride, 4, llvm::formatv("vdst.l{0}", L).str()});
  Layout.TotalSize = InBase + NumLanes * InStride;
  Layout.OutputBufSize = NumLanes * OutStride;

  KernelMFContext KCtx;
  KCtx.KernelName = MachineKernelBuilder::getKernelName(Profile);
  auto ScafOrErr = setupScaffold(TM, KCtx, Layout.Inputs.size(), /*NumPtrArgs=*/1,
                                 /*EnableFlatScratch=*/false,
                                 /*FlatWorkGroupSize=*/NumLanes);
  if (!ScafOrErr)
    return ScafOrErr.takeError();
  Scaffold S = *ScafOrErr;
  const llvm::SIInstrInfo &TII = *S.TII;
  if (TII.pseudoToMCOpcode(Op) < 0)
    return makeError(Profile.Name + ": no MC encoding for this subtarget");

  llvm::MCRegister Tid = vgpr32(0);
  llvm::MCRegister InVoff = vgpr32(1);
  llvm::MCRegister OutVoff = vgpr32(2);
  llvm::MCRegister Data0V = vgpr32(InputVGPRBase);     // v10
  llvm::MCRegister AddrV = vgpr32(InputVGPRBase + 1);  // v11
  llvm::MCRegister VdstV = vgpr32(OutputVGPRBase);     // v20

  {
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [DS-Permute full-wave " << NumLanes << " lanes]\n"
       << "  prolog:\n"
       << "    P3 per-lane data0" << (HasAddr ? " + addr" : "")
       << " <- kernarg[lane]\n"
       << "  epilog:\n"
       << "    E4 per-lane vdst -> out[lane]\n";
  }

  //=== Emit MIR ===================================================//
  EmitState E{S.BB, TII, S.KernargReg, S.OutPtrReg, {}};
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          S.OutPtrReg)
      .addReg(S.KernargReg).addImm(Layout.OutputPtrOffset).addImm(0);
  emitWaitcnt(E);

  // Per-lane byte offsets: inVoff = tid * InStride, outVoff = tid * 4.
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::V_LSHLREV_B32_e32),
          InVoff).addImm(llvm::Log2_32(InStride)).addReg(Tid);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::V_LSHLREV_B32_e32),
          OutVoff).addImm(llvm::Log2_32(OutStride)).addReg(Tid);

  // Per-lane load of data0 (and addr) from this lane's kernarg slice.
  auto kernargLoad = [&](llvm::MCRegister Dst, uint32_t FieldOff) {
    unsigned LOp = llvm::AMDGPU::GLOBAL_LOAD_DWORD_SADDR;
    int DstI = llvm::AMDGPU::getNamedOperandIdx(LOp, llvm::AMDGPU::OpName::vdst);
    int SaI = llvm::AMDGPU::getNamedOperandIdx(LOp, llvm::AMDGPU::OpName::saddr);
    int VaI = llvm::AMDGPU::getNamedOperandIdx(LOp, llvm::AMDGPU::OpName::vaddr);
    int CpI = llvm::AMDGPU::getNamedOperandIdx(LOp, llvm::AMDGPU::OpName::cpol);
    const llvm::MCInstrDesc &LD = TII.get(LOp);
    llvm::MachineInstrBuilder M = BuildMI(*S.BB, S.BB->end(), E.DL, LD);
    for (unsigned I = 0; I < LD.getNumOperands(); ++I) {
      if ((int)I == DstI)      M.addDef(Dst);
      else if ((int)I == SaI)  M.addReg(S.KernargReg);
      else if ((int)I == VaI)  M.addReg(InVoff);
      else if ((int)I == CpI)  M.addImm(0);
      else                     M.addImm(FieldOff); // the offset immediate
    }
  };
  kernargLoad(Data0V, InBase);
  if (HasAddr)
    kernargLoad(AddrV, InBase + 4);
  emitWaitcnt(E);

  // The permute under test, filled by named role.
  llvm::MachineInstrBuilder MIB = BuildMI(*S.BB, S.BB->end(), E.DL, Desc);
  for (unsigned I = 0; I < Desc.getNumOperands(); ++I) {
    if ((int)I == VdstIdx)        MIB.addDef(VdstV);
    else if ((int)I == AddrIdx)   MIB.addReg(AddrV);
    else if ((int)I == Data0Idx)  MIB.addReg(Data0V);
    else                          MIB.addImm(0); // offset (permute) / swizzle
  }
  KCtx.MI = MIB.getInstr();
  emitWaitcnt(E);

  // Per-lane store of vdst to this lane's output slice.
  emitGlobalStore(E, VdstV, OutVoff, /*OutputOffset=*/0);
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

llvm::Expected<KernelMFContext> buildDS(llvm::TargetMachine &TM,
                                        const InstrProfile &Profile,
                                        KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);
  DSKind Kind = classifyDS(Desc, Profile);

  if (Kind == DSKind::Permute)
    return buildDSPermute(TM, Profile, Layout);
  if (Kind == DSKind::Paired || Kind == DSKind::Unsupported ||
      Kind == DSKind::NotDS)
    return makeError(Profile.Name + ": DS shape " + toString(Kind) +
                     " not handled by the reference path yet");

  bool NeedInit = Kind == DSKind::Load || Kind == DSKind::AtomicNoRet ||
                  Kind == DSKind::AtomicRet;
  bool HasData0 = llvm::AMDGPU::getNamedOperandIdx(
                      Profile.Opcode, llvm::AMDGPU::OpName::data0) >= 0;
  bool CaptureVdst = Kind == DSKind::Load || Kind == DSKind::AtomicRet;
  bool Readback = Kind == DSKind::Store || Kind == DSKind::AtomicNoRet ||
                  Kind == DSKind::AtomicRet;

  // Kernarg inputs: [init?] [data0?]; outputs: [vdst?] [lds_readback?].
  Layout = KernargLayout{};
  Layout.GridSizeX = 1;
  Layout.WorkgroupSizeX = 1;
  Layout.GroupSegmentSize = DSGroupSegmentSize;
  uint32_t KOff = 8, OOff = 0;
  uint32_t InitInOff = UINT32_MAX, Data0InOff = UINT32_MAX;
  uint32_t VdstOutOff = UINT32_MAX, LdsOutOff = UINT32_MAX;
  if (NeedInit) {
    InitInOff = KOff;
    Layout.Inputs.push_back({KOff, 4, "lds_init"});
    KOff += 4;
  }
  if (HasData0) {
    Data0InOff = KOff;
    Layout.Inputs.push_back({KOff, 4, "data0"});
    KOff += 4;
  }
  if (CaptureVdst) {
    VdstOutOff = OOff;
    Layout.Outputs.push_back({OOff, 4, "vdst"});
    OOff += 4;
  }
  if (Readback) {
    LdsOutOff = OOff;
    Layout.Outputs.push_back({OOff, 4, "lds_after"});
    OOff += 4;
  }
  Layout.TotalSize = KOff;
  Layout.OutputBufSize = OOff;

  KernelMFContext KCtx;
  KCtx.KernelName = MachineKernelBuilder::getKernelName(Profile);
  auto ScafOrErr = setupScaffold(TM, KCtx, Layout.Inputs.size());
  if (!ScafOrErr)
    return ScafOrErr.takeError();
  Scaffold S = *ScafOrErr;
  const llvm::SIInstrInfo &TII = *S.TII;

  if (TII.pseudoToMCOpcode(Profile.Opcode) < 0)
    return makeError(Profile.Name +
                     ": no MC encoding for this subtarget (pseudo needs "
                     "expansion)");

  // Register roles.
  llvm::MCRegister AddrV = vgpr32(DSAddrVGPR);
  llvm::MCRegister ReadbackV = vgpr32(DSReadbackVGPR);
  llvm::MCRegister InitV = vgpr32(InputVGPRBase);      // v10
  llvm::MCRegister Data0V = vgpr32(InputVGPRBase + 1); // v11
  llvm::MCRegister VdstV = vgpr32(OutputVGPRBase);     // v20
  llvm::MCRegister ZeroV = vgpr32(0);

  auto note = [&](RegBinding::RoleKind R, const std::string &N) {
    RegBinding B;
    B.Role = R;
    B.Note = N;
    return B;
  };
  {
    RegBinding B = note(RegBinding::SetSpecial, "P4 set M0 = -1");
    B.Special = SpecialReg::M0;
    KCtx.Prolog.push_back(B);
  }
  if (NeedInit)
    KCtx.Prolog.push_back(note(
        RegBinding::LoadInputVGPR,
        llvm::formatv("P2 lds_init <- ka+{0}; DS_WRITE LDS[0]", InitInOff)
            .str()));
  if (HasData0)
    KCtx.Prolog.push_back(note(
        RegBinding::LoadInputVGPR,
        llvm::formatv("P2 data0 <- ka+{0}", Data0InOff).str()));
  if (CaptureVdst) {
    RegBinding B = note(
        RegBinding::StoreOutput,
        llvm::formatv("E1 store vdst -> out+{0}", VdstOutOff).str());
    B.Reg = VdstV;
    B.IsVGPR = true;
    B.OutputOffset = VdstOutOff;
    KCtx.Epilog.push_back(B);
  }
  if (Readback) {
    RegBinding B = note(
        RegBinding::MemReadback,
        llvm::formatv("E5 read LDS[0] -> out+{0}", LdsOutOff).str());
    B.OutputOffset = LdsOutOff;
    KCtx.Epilog.push_back(B);
  }
  {
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [" << toString(Kind) << "]\n  prolog:\n";
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
  unsigned Stage = 0, InitStage = 0, Data0Stage = 0;
  if (NeedInit) {
    InitStage = Stage++;
    emitScalarLoad(E, sgpr32(StageSGPRBase + InitStage), InitInOff);
  }
  if (HasData0) {
    Data0Stage = Stage++;
    emitScalarLoad(E, sgpr32(StageSGPRBase + Data0Stage), Data0InOff);
  }
  emitWaitcnt(E);

  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
          llvm::AMDGPU::M0)
      .addImm(-1);
  emitVMovImm(E, AddrV, 0);
  if (NeedInit)
    emitVMovReg(E, InitV, sgpr32(StageSGPRBase + InitStage));
  if (HasData0)
    emitVMovReg(E, Data0V, sgpr32(StageSGPRBase + Data0Stage));

  if (NeedInit) {
    emitDSWrite(E, AddrV, InitV);
    emitWaitcnt(E);
  }

  // The instruction under test: fill operands by named role.
  auto opIdx = [&](llvm::AMDGPU::OpName N) {
    return llvm::AMDGPU::getNamedOperandIdx(Profile.Opcode, N);
  };
  int VdstIdx = opIdx(llvm::AMDGPU::OpName::vdst);
  int AddrIdx = opIdx(llvm::AMDGPU::OpName::addr);
  int Data0Idx = opIdx(llvm::AMDGPU::OpName::data0);
  llvm::MachineInstrBuilder MIB = BuildMI(*S.BB, S.BB->end(), E.DL, Desc);
  for (unsigned I = 0; I < Desc.getNumOperands(); ++I) {
    if ((int)I == VdstIdx)
      MIB.addDef(VdstV);
    else if ((int)I == AddrIdx)
      MIB.addReg(AddrV);
    else if ((int)I == Data0Idx)
      MIB.addReg(Data0V);
    else
      MIB.addImm(0); // offset / gds
  }
  KCtx.MI = MIB.getInstr();
  emitWaitcnt(E);

  if (Readback) {
    emitDSRead(E, ReadbackV, AddrV);
    emitWaitcnt(E);
  }

  emitVMovImm(E, ZeroV, 0);
  if (CaptureVdst)
    emitGlobalStore(E, VdstV, ZeroV, VdstOutOff);
  if (Readback)
    emitGlobalStore(E, ReadbackV, ZeroV, LdsOutOff);
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

} // namespace luthier::test
