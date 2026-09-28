//===-- MUBUF.cpp - MUBUF (untyped buffer) reference-kernel builder -----===//
//
// Builds the reference kernel for the buffer memory family: MUBUF (untyped) and
// MTBUF (typed). Buffer instructions address memory through a 128-bit resource
// descriptor (V#) in srsrc plus a scalar offset and, in the vaddr forms, a
// per-lane VGPR. All addressing modes are handled at element 0 (single work-
// item): OFFSET (no vaddr), OFFEN/IDXEN (vaddr = a 32-bit VGPR held at 0), and
// BOTHEN (vaddr = a VReg_64 held at 0). MTBUF fills the `format` immediate with
// a 32-bit UINT format (component count from the FORMAT_X/XY/XYZ/XYZW suffix) so
// the value passes through unchanged.
//
// The V# is built in-kernel from a host-visible global buffer's pointer (the
// same technique as S_BUFFER_LOAD). The kernel initializes the buffer, runs the
// buffer op under test, and reads the buffer back into the output buffer -- init
// and readback use untyped BUFFER_*_OFFSET helpers with glc, so the round trip
// is coherent through L2. Data width follows the mnemonic: byte / short
// (sub-dword, zero-extended) or one-or-more dwords.
//
// Rejected cleanly (follow-ups): CMPSWAP (mismatched tied widths) and D16.
//
//===----------------------------------------------------------------------===//
#include "InstructionBuilders.h"
#include "RefKernelSupport.h"

#include <SIInstrInfo.h>

#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier::test {

namespace {

constexpr uint32_t DataBufPtrKernargOffset = 8;
constexpr unsigned BufPtrSGPRIdx = 16;   // s[16:17] = data-buffer pointer
constexpr unsigned SOffsetSGPRIdx = 18;  // s18 = 0 scalar offset
constexpr unsigned VAddrVGPRBase = 2;    // v2 / v[2:3] = 0 vaddr (offen/bothen)
constexpr unsigned VSharpSGPRBase = 24;  // s[24:27] = V# resource descriptor
constexpr unsigned InitVGPRBase = 12;    // v[12:15] = init value tuple
constexpr unsigned VDataVGPRBase = 16;   // v[16:19] = store data / atomic addend
constexpr unsigned VdstVGPRBase = 20;    // v[20:23] = load / atomic-RTN dest
constexpr unsigned ReadbackVGPRBase = 24;// v[24:27] = readback tuple
constexpr uint32_t MUBUFDataBufBytes = 4096;
constexpr int64_t Glc = 1;
constexpr unsigned MaxDwords = 4;

enum class MemKind { Load, Store, AtomicNoRet, AtomicRet };
enum class Elem { Byte, Short, Dword };

const char *toStr(MemKind K) {
  switch (K) {
  case MemKind::Load:        return "Load";
  case MemKind::Store:       return "Store";
  case MemKind::AtomicNoRet: return "AtomicNoRet";
  case MemKind::AtomicRet:   return "AtomicRet";
  }
  return "?";
}

int opIdx(unsigned Opcode, llvm::AMDGPU::OpName N) {
  return llvm::AMDGPU::getNamedOperandIdx(Opcode, N);
}

/// Element-sized OFFSET-form BUFFER helper load/store opcode (for init /
/// readback). \p Dwords applies to Elem::Dword. Loads zero-extend.
unsigned bufHelpOpc(bool Load, Elem El, unsigned Dwords) {
  using namespace llvm::AMDGPU;
  if (El == Elem::Byte)
    return Load ? BUFFER_LOAD_UBYTE_OFFSET : BUFFER_STORE_BYTE_OFFSET;
  if (El == Elem::Short)
    return Load ? BUFFER_LOAD_USHORT_OFFSET : BUFFER_STORE_SHORT_OFFSET;
  static const unsigned LoadD[4] = {BUFFER_LOAD_DWORD_OFFSET,
                                    BUFFER_LOAD_DWORDX2_OFFSET,
                                    BUFFER_LOAD_DWORDX3_OFFSET,
                                    BUFFER_LOAD_DWORDX4_OFFSET};
  static const unsigned StoreD[4] = {BUFFER_STORE_DWORD_OFFSET,
                                     BUFFER_STORE_DWORDX2_OFFSET,
                                     BUFFER_STORE_DWORDX3_OFFSET,
                                     BUFFER_STORE_DWORDX4_OFFSET};
  return (Load ? LoadD : StoreD)[Dwords - 1];
}

/// Single-dword OFFEN-form BUFFER helper (for per-lane init / readback). Loads
/// zero-extend.
unsigned bufHelpOpcOffen(bool Load, Elem El) {
  using namespace llvm::AMDGPU;
  switch (El) {
  case Elem::Byte:  return Load ? BUFFER_LOAD_UBYTE_OFFEN : BUFFER_STORE_BYTE_OFFEN;
  case Elem::Short: return Load ? BUFFER_LOAD_USHORT_OFFEN : BUFFER_STORE_SHORT_OFFEN;
  case Elem::Dword: return Load ? BUFFER_LOAD_DWORD_OFFEN : BUFFER_STORE_DWORD_OFFEN;
  }
  return Load ? BUFFER_LOAD_DWORD_OFFEN : BUFFER_STORE_DWORD_OFFEN;
}

/// Emit a BUFFER op, filling operands by named role. \p DataReg is vdata (def
/// for loads / atomic-RTN, use otherwise); for atomic-RTN it is also the tied
/// vdata_in. \p VAddr (if valid) fills the vaddr slot (offen/idxen/bothen);
/// \p Format fills the MTBUF format immediate.
void emitBuf(EmitState &E, unsigned Opcode, llvm::MCRegister DataReg, bool IsDef,
             llvm::MCRegister Srsrc, llvm::MCRegister Soffset,
             llvm::MCRegister VAddr = {}, int64_t Format = 0) {
  const llvm::MCInstrDesc &D = E.TII.get(Opcode);
  int VdataI = opIdx(Opcode, llvm::AMDGPU::OpName::vdata);
  int VdataInI = opIdx(Opcode, llvm::AMDGPU::OpName::vdata_in);
  int VaddrI = opIdx(Opcode, llvm::AMDGPU::OpName::vaddr);
  int SrsrcI = opIdx(Opcode, llvm::AMDGPU::OpName::srsrc);
  int SoffI = opIdx(Opcode, llvm::AMDGPU::OpName::soffset);
  int FormatI = opIdx(Opcode, llvm::AMDGPU::OpName::format);
  int CpolI = opIdx(Opcode, llvm::AMDGPU::OpName::cpol);
  llvm::MachineInstrBuilder M = BuildMI(*E.BB, E.BB->end(), E.DL, D);
  for (unsigned I = 0; I < D.getNumOperands(); ++I) {
    if ((int)I == VdataI)
      IsDef ? (void)M.addDef(DataReg) : (void)M.addReg(DataReg);
    else if ((int)I == VdataInI)
      M.addReg(DataReg); // tied to vdata (atomic RTN)
    else if ((int)I == VaddrI)
      M.addReg(VAddr);
    else if ((int)I == SrsrcI)
      M.addReg(Srsrc);
    else if ((int)I == SoffI)
      M.addReg(Soffset);
    else if ((int)I == FormatI)
      M.addImm(Format);
    else if ((int)I == CpolI)
      M.addImm(Glc);
    else
      M.addImm(0); // offset / swz / ...
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// True per-lane buffer addressing (OFFEN, full wave): dispatch 64 lanes, and
// give every lane its own buffer element via a per-lane voffset (tid*elemBytes),
// its own kernarg inputs, and its own output slot. Single 32-bit element (dword
// or sub-dword byte/short); load / store / atomic (add + RTN).
//===----------------------------------------------------------------------===//
static llvm::Expected<KernelMFContext>
buildMUBUFPerLane(llvm::TargetMachine &TM, const InstrProfile &Profile,
                  KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);
  unsigned Op = Profile.Opcode;
  const bool IsMTBUF = opIdx(Op, llvm::AMDGPU::OpName::format) >= 0;

  //=== Element kind (single 32-bit element only) ==================//
  Elem El = Elem::Dword;
  int64_t Format = 0;
  {
    llvm::StringRef N = Profile.Name;
    if (N.contains("D16"))
      return makeError(Profile.Name + ": D16 (partial-register) not yet");
    if (N.contains("CMPSWAP"))
      return makeError(Profile.Name + ": buffer CMPSWAP not yet");
    if (IsMTBUF) {
      if (!N.contains("FORMAT_X") || N.contains("FORMAT_XY"))
        return makeError(Profile.Name + ": per-lane MTBUF is FORMAT_X only");
      Format = mtbufFormat(TM.getMCSubtargetInfo(), /*Dfmt=*/4 /*32*/,
                            /*Nfmt=*/4 /*UINT*/);
    } else if (N.contains("BYTE")) {
      El = Elem::Byte;
    } else if (N.contains("SHORT")) {
      El = Elem::Short;
    } else if (N.contains("DWORDX2") || N.contains("DWORDX3") ||
               N.contains("DWORDX4") || N.contains("_X2")) {
      return makeError(Profile.Name + ": wide per-lane buffer not yet");
    }
  }
  const unsigned ElemBytes = El == Elem::Byte ? 1 : El == Elem::Short ? 2 : 4;

  int VaddrIdx = opIdx(Op, llvm::AMDGPU::OpName::vaddr);
  const llvm::MCOperandInfo &VA = Desc.operands()[VaddrIdx];

  bool Ld = Desc.mayLoad(), St = Desc.mayStore();
  bool IsRtn = opIdx(Op, llvm::AMDGPU::OpName::vdata_in) >= 0;
  MemKind Kind;
  if (Ld && !St)        Kind = MemKind::Load;
  else if (St && !Ld)   Kind = MemKind::Store;
  else if (Ld && St && !IsRtn) Kind = MemKind::AtomicNoRet;
  else if (Ld && St && IsRtn)  Kind = MemKind::AtomicRet;
  else return makeError(Profile.Name + ": unhandled MUBUF shape");

  bool NeedInit = Kind != MemKind::Store;
  bool HasData = Kind != MemKind::Load;
  bool CaptureReg = Kind == MemKind::Load || Kind == MemKind::AtomicRet;
  bool Readback = Kind != MemKind::Load;
  bool VdataIsDef = Kind == MemKind::Load || Kind == MemKind::AtomicRet;

  const unsigned NumLanes = waveSize(TM); // 32 (RDNA wave32) or 64 (wave64)
  const unsigned NumInRoles = (NeedInit ? 1 : 0) + (HasData ? 1 : 0);
  const unsigned NumOutRoles = (CaptureReg ? 1 : 0) + (Readback ? 1 : 0);
  const unsigned InStride = NumInRoles * 4;   // 4 or 8 (power of two)
  const unsigned OutStride = NumOutRoles * 4; // 4 or 8
  const uint32_t InBase = 16;                 // after out + databuf pointers
  const uint32_t InitRoleOff = 0;
  const uint32_t DataRoleOff = NeedInit ? 4 : 0;
  const uint32_t CapRoleOff = 0;
  const uint32_t MemRoleOff = CaptureReg ? 4 : 0;

  //=== Layout: one field per (lane, role) ========================//
  Layout = KernargLayout{};
  Layout.GridSizeX = NumLanes;
  Layout.WorkgroupSizeX = NumLanes;
  Layout.OutputPtrOffset = 0;
  Layout.DataBufPtrOffset = DataBufPtrKernargOffset;
  Layout.DataBufSizeBytes = MUBUFDataBufBytes;
  uint32_t InitInOff = UINT32_MAX, DataInOff = UINT32_MAX;
  uint32_t CapOutOff = UINT32_MAX, MemOutOff = UINT32_MAX;
  for (unsigned L = 0; L < NumLanes; ++L) {
    if (NeedInit) {
      uint32_t O = InBase + L * InStride + InitRoleOff;
      if (L == 0) InitInOff = O;
      Layout.Inputs.push_back({O, 4, llvm::formatv("mem_init.l{0}", L).str()});
    }
    if (HasData) {
      uint32_t O = InBase + L * InStride + DataRoleOff;
      if (L == 0) DataInOff = O;
      Layout.Inputs.push_back({O, 4, llvm::formatv("vdata.l{0}", L).str()});
    }
  }
  for (unsigned L = 0; L < NumLanes; ++L) {
    if (CaptureReg) {
      uint32_t O = L * OutStride + CapRoleOff;
      if (L == 0) CapOutOff = O;
      Layout.Outputs.push_back({O, 4, llvm::formatv("vdst.l{0}", L).str()});
    }
    if (Readback) {
      uint32_t O = L * OutStride + MemRoleOff;
      if (L == 0) MemOutOff = O;
      Layout.Outputs.push_back({O, 4, llvm::formatv("mem_after.l{0}", L).str()});
    }
  }
  Layout.TotalSize = InBase + NumLanes * InStride;
  Layout.OutputBufSize = NumLanes * OutStride;

  KernelMFContext KCtx;
  KCtx.KernelName = MachineKernelBuilder::getKernelName(Profile);
  auto ScafOrErr = setupScaffold(TM, KCtx, Layout.Inputs.size(), /*NumPtrArgs=*/2,
                                 /*EnableFlatScratch=*/false,
                                 /*FlatWorkGroupSize=*/NumLanes);
  if (!ScafOrErr)
    return ScafOrErr.takeError();
  Scaffold S = *ScafOrErr;
  const llvm::SIInstrInfo &TII = *S.TII;
  const llvm::SIRegisterInfo &TRI = *S.TRI;
  if (TII.pseudoToMCOpcode(Op) < 0)
    return makeError(Profile.Name + ": no MC encoding for this subtarget");
  if (VA.RegClass < 0 || TRI.getRegSizeInBits(*TRI.getRegClass(VA.RegClass)) != 32)
    return makeError(Profile.Name + ": per-lane needs a 32-bit OFFEN voffset");

  //=== Registers =================================================//
  llvm::MCRegister BufPtr = TRI.getMatchingSuperReg(
      sgpr32(BufPtrSGPRIdx), llvm::AMDGPU::sub0, &llvm::AMDGPU::SGPR_64RegClass);
  llvm::MCRegister VSharp = TRI.getMatchingSuperReg(
      sgpr32(VSharpSGPRBase), llvm::AMDGPU::sub0, &llvm::AMDGPU::SGPR_128RegClass);
  if (!BufPtr || !VSharp)
    return makeError(Profile.Name + ": failed to form sbase / V# registers");
  llvm::MCRegister Soffset = sgpr32(SOffsetSGPRIdx);
  llvm::MCRegister Tid = vgpr32(0);
  llvm::MCRegister InVoff = vgpr32(1);
  llvm::MCRegister OutVoff = vgpr32(2);
  llvm::MCRegister BufVoff = vgpr32(3);
  llvm::MCRegister InitV = vgpr32(10);
  llvm::MCRegister VDataV = vgpr32(11);
  llvm::MCRegister VdstV = vgpr32(12);
  llvm::MCRegister ReadbackV = vgpr32(13);
  llvm::MCRegister OpReg = Kind == MemKind::Load ? VdstV : VDataV;

  {
    const char *ElStr = El == Elem::Byte ? "b8" : El == Elem::Short ? "b16" : "dw";
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [MUBUF-OFFEN per-lane " << NumLanes << " lanes-"
       << ElStr << "]\n  prolog: build V#; per-lane voffset=tid*" << ElemBytes
       << "; " << (NeedInit ? "init buffer[tid]; " : "")
       << (HasData ? "load vdata[tid]" : "") << "\n  epilog: "
       << (CaptureReg ? "capture vdst[tid]; " : "")
       << (Readback ? "read buffer[tid]" : "") << "\n";
  }

  //=== Emit MIR ==================================================//
  EmitState E{S.BB, TII, S.KernargReg, S.OutPtrReg, {}};
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          S.OutPtrReg)
      .addReg(S.KernargReg).addImm(Layout.OutputPtrOffset).addImm(0);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          BufPtr)
      .addReg(S.KernargReg).addImm(Layout.DataBufPtrOffset).addImm(0);
  emitWaitcnt(E);

  auto sMov = [&](unsigned DstIdx, auto Src) {
    BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
            sgpr32(DstIdx)).add(Src);
  };
  sMov(VSharpSGPRBase + 0,
       llvm::MachineOperand::CreateReg(sgpr32(BufPtrSGPRIdx + 0), false));
  sMov(VSharpSGPRBase + 1,
       llvm::MachineOperand::CreateReg(sgpr32(BufPtrSGPRIdx + 1), false));
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_AND_B32),
          sgpr32(VSharpSGPRBase + 1))
      .addReg(sgpr32(VSharpSGPRBase + 1)).addImm(0xFFFF);
  sMov(VSharpSGPRBase + 2, llvm::MachineOperand::CreateImm(MUBUFDataBufBytes));
  sMov(VSharpSGPRBase + 3,
       llvm::MachineOperand::CreateImm(
           vSharpWord3(S.MF->getSubtarget())));
  sMov(SOffsetSGPRIdx, llvm::MachineOperand::CreateImm(0));

  // Per-lane byte offsets.
  auto lshl = [&](llvm::MCRegister Dst, unsigned Shift) {
    BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::V_LSHLREV_B32_e32),
            Dst).addImm(Shift).addReg(Tid);
  };
  lshl(InVoff, llvm::Log2_32(InStride));
  lshl(OutVoff, llvm::Log2_32(OutStride));
  lshl(BufVoff, llvm::Log2_32(ElemBytes)); // ElemBytes is 1/2/4

  // Per-lane kernarg loads (directly into VGPRs).
  auto kernargLoad = [&](llvm::MCRegister Dst, uint32_t FieldOff) {
    unsigned LOp = llvm::AMDGPU::GLOBAL_LOAD_DWORD_SADDR;
    int DstI = opIdx(LOp, llvm::AMDGPU::OpName::vdst);
    int SaI = opIdx(LOp, llvm::AMDGPU::OpName::saddr);
    int VaI = opIdx(LOp, llvm::AMDGPU::OpName::vaddr);
    int CpI = opIdx(LOp, llvm::AMDGPU::OpName::cpol);
    const llvm::MCInstrDesc &LD = TII.get(LOp);
    llvm::MachineInstrBuilder M = BuildMI(*S.BB, S.BB->end(), E.DL, LD);
    for (unsigned I = 0; I < LD.getNumOperands(); ++I) {
      if ((int)I == DstI)      M.addDef(Dst);
      else if ((int)I == SaI)  M.addReg(S.KernargReg);
      else if ((int)I == VaI)  M.addReg(InVoff);
      else if ((int)I == CpI)  M.addImm(0);
      else                     M.addImm(FieldOff);
    }
  };
  if (NeedInit) kernargLoad(InitV, InBase + InitRoleOff);
  if (HasData)  kernargLoad(VDataV, InBase + DataRoleOff);
  emitWaitcnt(E);

  if (NeedInit) {
    emitBuf(E, bufHelpOpcOffen(/*Load=*/false, El), InitV, /*IsDef=*/false,
            VSharp, Soffset, BufVoff);
    emitWaitcnt(E);
  }

  emitBuf(E, Op, OpReg, VdataIsDef, VSharp, Soffset, BufVoff, Format);
  KCtx.MI = &*std::prev(S.BB->end());
  emitWaitcnt(E);

  if (Readback) {
    emitBuf(E, bufHelpOpcOffen(/*Load=*/true, El), ReadbackV, /*IsDef=*/true,
            VSharp, Soffset, BufVoff);
    emitWaitcnt(E);
  }

  if (CaptureReg)
    emitGlobalStore(E, OpReg, OutVoff, CapOutOff);
  if (Readback)
    emitGlobalStore(E, ReadbackV, OutVoff, MemOutOff);
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

llvm::Expected<KernelMFContext> buildMUBUF(llvm::TargetMachine &TM,
                                           const InstrProfile &Profile,
                                           KernargLayout &Layout) {
  // OFFEN forms get true per-lane addressing (each lane -> buffer[tid]).
  if (Profile.Name.contains("OFFEN"))
    return buildMUBUFPerLane(TM, Profile, Layout);

  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);
  unsigned Op = Profile.Opcode;

  //=== Shape guards ================================================//
  int SrsrcIdx = opIdx(Op, llvm::AMDGPU::OpName::srsrc);
  int VdataIdx = opIdx(Op, llvm::AMDGPU::OpName::vdata);
  if (SrsrcIdx < 0 || VdataIdx < 0)
    return makeError(Profile.Name + ": unexpected MUBUF operand shape");
  int VaddrIdx = opIdx(Op, llvm::AMDGPU::OpName::vaddr);
  const bool HasVaddr = VaddrIdx >= 0; // offen/idxen/bothen (else offset form)
  const bool IsMTBUF = opIdx(Op, llvm::AMDGPU::OpName::format) >= 0;

  //=== Element kind / width from the mnemonic =====================//
  Elem El = Elem::Dword;
  unsigned Dwords = 1;
  int64_t Format = 0;
  {
    llvm::StringRef N = Profile.Name;
    if (N.contains("D16"))
      return makeError(Profile.Name + ": D16 (partial-register) not yet");
    if (N.contains("CMPSWAP"))
      return makeError(Profile.Name + ": buffer CMPSWAP not yet");
    if (IsMTBUF) {
      // Typed buffer: component count comes from the FORMAT_X/XY/XYZ/XYZW
      // suffix; use a 32-bit UINT format so the value passes through unchanged
      // (format = dfmt | (nfmt<<4), nfmt UINT = 4).
      if (N.contains("FORMAT_XYZW"))      Dwords = 4;
      else if (N.contains("FORMAT_XYZ"))  Dwords = 3;
      else if (N.contains("FORMAT_XY"))   Dwords = 2;
      else                                Dwords = 1;
      static const int64_t Dfmt[4] = {4 /*32*/, 11 /*32_32*/, 13 /*32_32_32*/,
                                      14 /*32_32_32_32*/};
      Format = mtbufFormat(TM.getMCSubtargetInfo(), Dfmt[Dwords - 1],
                           /*Nfmt=*/4 /*UINT*/);
    } else if (N.contains("BYTE")) {
      El = Elem::Byte;
    } else if (N.contains("SHORT")) {
      El = Elem::Short;
    } else if (N.contains("DWORDX4")) {
      Dwords = 4;
    } else if (N.contains("DWORDX3")) {
      Dwords = 3;
    } else if (N.contains("DWORDX2")) {
      Dwords = 2;
    } else if (N.contains("_X2") || N.contains("_B64") || N.contains("_F64")) {
      Dwords = 2; // 64-bit atomic
    }
  }

  bool Ld = Desc.mayLoad(), St = Desc.mayStore();
  bool IsRtn = opIdx(Op, llvm::AMDGPU::OpName::vdata_in) >= 0;
  MemKind Kind;
  if (Ld && !St)
    Kind = MemKind::Load;
  else if (St && !Ld)
    Kind = MemKind::Store;
  else if (Ld && St && !IsRtn)
    Kind = MemKind::AtomicNoRet;
  else if (Ld && St && IsRtn)
    Kind = MemKind::AtomicRet;
  else
    return makeError(Profile.Name + ": unhandled MUBUF shape");

  bool NeedInit = Kind != MemKind::Store;   // load/atomics read existing mem
  bool HasData = Kind != MemKind::Load;      // store/atomics supply data/addend
  bool CaptureReg = Kind == MemKind::Load || Kind == MemKind::AtomicRet;
  bool Readback = Kind != MemKind::Load;
  // The op-under-test's vdata is a def for loads and atomic-RTN.
  bool VdataIsDef = Kind == MemKind::Load || Kind == MemKind::AtomicRet;

  //=== Layout =====================================================//
  Layout = KernargLayout{};
  Layout.OutputPtrOffset = 0;
  Layout.DataBufPtrOffset = DataBufPtrKernargOffset;
  Layout.DataBufSizeBytes = MUBUFDataBufBytes;
  uint32_t KOff = 16, OOff = 0;
  uint32_t InitInOff = UINT32_MAX, DataInOff = UINT32_MAX;
  uint32_t CapOutOff = UINT32_MAX, MemOutOff = UINT32_MAX;
  auto addFields = [](uint32_t &Cur, auto &V, const char *Name, unsigned N) {
    uint32_t Base = Cur;
    for (unsigned D = 0; D < N; ++D, Cur += 4)
      V.push_back({Cur, 4, N > 1 ? llvm::formatv("{0}.{1}", Name, D).str()
                                 : std::string(Name)});
    return Base;
  };
  if (NeedInit)
    InitInOff = addFields(KOff, Layout.Inputs, "mem_init", Dwords);
  if (HasData)
    DataInOff = addFields(KOff, Layout.Inputs, "vdata", Dwords);
  if (CaptureReg)
    CapOutOff = addFields(OOff, Layout.Outputs, "vdst", Dwords);
  if (Readback)
    MemOutOff = addFields(OOff, Layout.Outputs, "mem_after", Dwords);
  Layout.TotalSize = KOff;
  Layout.OutputBufSize = OOff;

  KernelMFContext KCtx;
  KCtx.KernelName = MachineKernelBuilder::getKernelName(Profile);
  auto ScafOrErr = setupScaffold(TM, KCtx, Layout.Inputs.size(), /*NumPtrArgs=*/2);
  if (!ScafOrErr)
    return ScafOrErr.takeError();
  Scaffold S = *ScafOrErr;
  const llvm::SIInstrInfo &TII = *S.TII;
  const llvm::SIRegisterInfo &TRI = *S.TRI;

  if (TII.pseudoToMCOpcode(Op) < 0)
    return makeError(Profile.Name + ": no MC encoding for this subtarget");
  if (Dwords > MaxDwords)
    return makeError(Profile.Name + ": data too wide");

  //=== Registers ==================================================//
  llvm::MCRegister BufPtr = TRI.getMatchingSuperReg(
      sgpr32(BufPtrSGPRIdx), llvm::AMDGPU::sub0, &llvm::AMDGPU::SGPR_64RegClass);
  llvm::MCRegister VSharp = TRI.getMatchingSuperReg(
      sgpr32(VSharpSGPRBase), llvm::AMDGPU::sub0, &llvm::AMDGPU::SGPR_128RegClass);
  if (!BufPtr || !VSharp)
    return makeError(Profile.Name + ": failed to form sbase / V# registers");
  {
    const llvm::MCOperandInfo &SR = Desc.operands()[SrsrcIdx];
    if (SR.RegClass < 0 || !TRI.getRegClass(SR.RegClass)->contains(VSharp))
      return makeError(Profile.Name + ": srsrc not accepted by operand class");
  }
  llvm::MCRegister Soffset = sgpr32(SOffsetSGPRIdx);
  llvm::MCRegister InitV = vgprTuple(TRI, InitVGPRBase, Dwords);
  llvm::MCRegister VDataV = vgprTuple(TRI, VDataVGPRBase, Dwords);
  llvm::MCRegister VdstV = vgprTuple(TRI, VdstVGPRBase, Dwords);
  llvm::MCRegister ReadbackV = vgprTuple(TRI, ReadbackVGPRBase, Dwords);
  llvm::MCRegister ZeroV = vgpr32(0);
  if (!InitV || !VDataV || !VdstV || !ReadbackV)
    return makeError(Profile.Name + ": failed to form a data tuple");
  // Zero vaddr for the offen/idxen (VGPR_32) or bothen (VReg_64) forms; the
  // access stays at element 0 (single work-item), like the OFFSET form.
  unsigned VAddrDwords = 0;
  llvm::MCRegister VAddrReg;
  if (HasVaddr) {
    const llvm::MCOperandInfo &VA = Desc.operands()[VaddrIdx];
    VAddrDwords =
        VA.RegClass < 0 ? 1 : TRI.getRegSizeInBits(*TRI.getRegClass(VA.RegClass)) / 32;
    // BOTHEN's vaddr is a wide {vindex, voffset} placeholder class (VReg_96 on
    // this build) the codegen verifier rejects -- and it is redundant with OFFEN
    // at element 0. Support the single-VGPR forms (OFFEN / IDXEN) only.
    if (VAddrDwords != 1)
      return makeError(Profile.Name + ": BOTHEN (index+offset vaddr) not supported");
    VAddrReg = vgprTuple(TRI, VAddrVGPRBase, VAddrDwords);
    if (!VAddrReg)
      return makeError(Profile.Name + ": failed to form a vaddr register");
  }
  // The op-under-test's vdata register: the load dest (VdstV) for loads, else
  // the seeded data/addend (VDataV) -- which for atomic-RTN is also the def
  // (tied), so it holds the returned pre-op value afterwards. This is the
  // register captured to the output buffer.
  llvm::MCRegister OpReg = Kind == MemKind::Load ? VdstV : VDataV;

  unsigned StoreHelp = bufHelpOpc(/*Load=*/false, El, Dwords);
  unsigned LoadHelp = bufHelpOpc(/*Load=*/true, El, Dwords);

  {
    const char *ElStr = El == Elem::Byte ? "b8" : El == Elem::Short ? "b16" : "dw";
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [MUBUF-OFFSET-" << toStr(Kind) << "-" << Dwords
       << ElStr << "]\n"
       << "  prolog: build V#; " << (NeedInit ? "init buffer[0]; " : "")
       << (HasData ? "load vdata" : "") << "\n"
       << "  epilog: " << (CaptureReg ? "capture vdst; " : "")
       << (Readback ? "read buffer[0]" : "") << "\n";
  }

  //=== Emit MIR ===================================================//
  EmitState E{S.BB, TII, S.KernargReg, S.OutPtrReg, {}};
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          S.OutPtrReg)
      .addReg(S.KernargReg).addImm(Layout.OutputPtrOffset).addImm(0);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          BufPtr)
      .addReg(S.KernargReg).addImm(Layout.DataBufPtrOffset).addImm(0);
  emitWaitcnt(E);

  // Build the raw V# from the data-buffer pointer.
  auto sMov = [&](unsigned DstIdx, auto Src) {
    BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
            sgpr32(DstIdx)).add(Src);
  };
  sMov(VSharpSGPRBase + 0,
       llvm::MachineOperand::CreateReg(sgpr32(BufPtrSGPRIdx + 0), false));
  sMov(VSharpSGPRBase + 1,
       llvm::MachineOperand::CreateReg(sgpr32(BufPtrSGPRIdx + 1), false));
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_AND_B32),
          sgpr32(VSharpSGPRBase + 1))
      .addReg(sgpr32(VSharpSGPRBase + 1)).addImm(0xFFFF);
  sMov(VSharpSGPRBase + 2, llvm::MachineOperand::CreateImm(MUBUFDataBufBytes));
  sMov(VSharpSGPRBase + 3,
       llvm::MachineOperand::CreateImm(
           vSharpWord3(S.MF->getSubtarget())));
  sMov(SOffsetSGPRIdx, llvm::MachineOperand::CreateImm(0));

  // Load init / data dwords into staging SGPRs, then into their VGPR tuples.
  unsigned Stage = 0, InitStage0 = 0, DataStage0 = 0;
  if (NeedInit) { InitStage0 = Stage; Stage += Dwords; }
  if (HasData)  { DataStage0 = Stage; Stage += Dwords; }
  if (NeedInit)
    for (unsigned D = 0; D < Dwords; ++D)
      emitScalarLoad(E, sgpr32(StageSGPRBase + InitStage0 + D), InitInOff + D * 4);
  if (HasData)
    for (unsigned D = 0; D < Dwords; ++D)
      emitScalarLoad(E, sgpr32(StageSGPRBase + DataStage0 + D), DataInOff + D * 4);
  emitWaitcnt(E);
  emitVMovImm(E, ZeroV, 0);
  for (unsigned D = 0; D < VAddrDwords; ++D)
    emitVMovImm(E, subDword(TRI, VAddrReg, D, VAddrDwords), 0);
  if (NeedInit)
    for (unsigned D = 0; D < Dwords; ++D)
      emitVMovReg(E, subDword(TRI, InitV, D, Dwords),
                  sgpr32(StageSGPRBase + InitStage0 + D));
  if (HasData)
    for (unsigned D = 0; D < Dwords; ++D)
      emitVMovReg(E, subDword(TRI, VDataV, D, Dwords),
                  sgpr32(StageSGPRBase + DataStage0 + D));

  if (NeedInit) {
    emitBuf(E, StoreHelp, InitV, /*IsDef=*/false, VSharp, Soffset);
    emitWaitcnt(E);
  }

  // The instruction under test. For atomic-RTN, vdata (== OpReg == VDataV) is
  // seeded with the addend above and is both the def and the tied vdata_in.
  emitBuf(E, Op, OpReg, VdataIsDef, VSharp, Soffset, VAddrReg, Format);
  KCtx.MI = &*std::prev(S.BB->end());
  emitWaitcnt(E);

  if (Readback) {
    emitBuf(E, LoadHelp, ReadbackV, /*IsDef=*/true, VSharp, Soffset);
    emitWaitcnt(E);
  }

  if (CaptureReg)
    for (unsigned D = 0; D < Dwords; ++D)
      emitGlobalStore(E, subDword(TRI, OpReg, D, Dwords), ZeroV,
                      CapOutOff + D * 4);
  if (Readback)
    for (unsigned D = 0; D < Dwords; ++D)
      emitGlobalStore(E, subDword(TRI, ReadbackV, D, Dwords), ZeroV,
                      MemOutOff + D * 4);
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

} // namespace luthier::test
