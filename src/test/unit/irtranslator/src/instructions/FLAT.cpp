//===-- FLAT.cpp - FLAT / GLOBAL reference-kernel builder ---------------===//
//
// Builds the reference kernel for the FLAT memory family (SCRATCH has its own
// builder). Mirroring the DS builder, the kernel initializes a host-visible
// global buffer, runs the load/store/atomic under test, and reads the buffer
// back into the output buffer, so the existing output comparison covers
// everything. Fixed byte offset 0 (in-bounds address) and `glc` (cpol bit 0)
// so the in-kernel store/op/readback stay coherent through L2.
//
// Three addressing modes:
//   * GLOBAL `_SADDR`: saddr = a 64-bit base pointer (SGPR pair) into the data
//     buffer; vaddr = 0 voffset.
//   * GLOBAL (no saddr): vaddr = the 64-bit global address in a VGPR pair; no
//     FLAT_SCR (a global-aperture instruction does not implicit-use it).
//   * plain FLAT: vaddr = the 64-bit global address (global aperture) in a VGPR
//     pair; FLAT_SCR defined to 0 (unused for a global-aperture access).
//
// Data width follows the mnemonic. The memory element (vdst / mem value) may be
// a byte (UBYTE/SBYTE/STORE_BYTE), a short (USHORT/SSHORT/STORE_SHORT), one
// dword, or an N-dword tuple (DWORDX2..4, and 64-bit atomics _X2/_B64/_F64);
// init/readback use an element-sized helper (byte/short zero-extend on load).
// CMPSWAP packs a 2x-wide vdata operand ({swap, cmp}) while its element/result
// stay element width. Every 32-bit component gets its own kernarg / output
// field. Only D16 (partial-register) forms are still rejected cleanly.
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

constexpr uint32_t DataBufPtrKernargOffset = 8;
// s[16:17] = global base pointer. Kept clear of the staging SGPRs (s8..s15),
// which a wide (DWORDX3/X4) data load/store uses up to s11 -- overlapping the
// saddr pair there would corrupt the address mid-kernel.
constexpr unsigned SaddrSGPRIdx = 16;
constexpr unsigned VOffsetVGPRIdx = 4;   // v4  = 0 byte offset (GLOBAL saddr)
constexpr unsigned FlatAddrVGPRIdx = 6;  // v[6:7] = 64-bit flat address
constexpr unsigned InitVGPRBase = 12;    // v[12:15] = init value tuple
constexpr unsigned VDataVGPRBase = 16;   // v[16:19] = store data tuple
constexpr unsigned VdstVGPRBase = 20;    // v[20:23] = load/atomic-RTN dest
constexpr unsigned ReadbackVGPRBase = 24;// v[24:27] = readback tuple
constexpr uint32_t FlatDataBufBytes = 4096;
constexpr int64_t Glc = 1;
constexpr unsigned MaxDwords = 4;

// GlobalSaddr: GLOBAL_*_SADDR (saddr base ptr + voffset). GlobalFlat: GLOBAL_*
// with a 64-bit vaddr and no saddr (global aperture, no FLAT_SCR). Flat: FLAT_*
// with a 64-bit vaddr (generic aperture, implicit-uses FLAT_SCR).
enum class AddrMode { GlobalSaddr, GlobalFlat, Flat };
enum class MemKind { Load, Store, AtomicNoRet, AtomicRet };

const char *toStr(AddrMode M) {
  switch (M) {
  case AddrMode::GlobalSaddr: return "GLOBAL-saddr";
  case AddrMode::GlobalFlat:  return "GLOBAL-vaddr";
  case AddrMode::Flat:        return "FLAT";
  }
  return "?";
}
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

/// Width-N helper store/load opcode for the given addressing mode. Indexed by
/// dword count (1..4).
unsigned helpOpc(AddrMode Mode, bool Load, unsigned Dwords) {
  using namespace llvm::AMDGPU;
  static const unsigned GLoad[4] = {GLOBAL_LOAD_DWORD_SADDR,
                                    GLOBAL_LOAD_DWORDX2_SADDR,
                                    GLOBAL_LOAD_DWORDX3_SADDR,
                                    GLOBAL_LOAD_DWORDX4_SADDR};
  static const unsigned GStore[4] = {GLOBAL_STORE_DWORD_SADDR,
                                     GLOBAL_STORE_DWORDX2_SADDR,
                                     GLOBAL_STORE_DWORDX3_SADDR,
                                     GLOBAL_STORE_DWORDX4_SADDR};
  static const unsigned FLoad[4] = {FLAT_LOAD_DWORD, FLAT_LOAD_DWORDX2,
                                    FLAT_LOAD_DWORDX3, FLAT_LOAD_DWORDX4};
  static const unsigned FStore[4] = {FLAT_STORE_DWORD, FLAT_STORE_DWORDX2,
                                     FLAT_STORE_DWORDX3, FLAT_STORE_DWORDX4};
  // GLOBAL_* with a 64-bit vaddr and no saddr.
  static const unsigned GFLoad[4] = {GLOBAL_LOAD_DWORD, GLOBAL_LOAD_DWORDX2,
                                     GLOBAL_LOAD_DWORDX3, GLOBAL_LOAD_DWORDX4};
  static const unsigned GFStore[4] = {GLOBAL_STORE_DWORD, GLOBAL_STORE_DWORDX2,
                                      GLOBAL_STORE_DWORDX3, GLOBAL_STORE_DWORDX4};
  const unsigned *T;
  switch (Mode) {
  case AddrMode::GlobalSaddr: T = Load ? GLoad : GStore; break;
  case AddrMode::GlobalFlat:  T = Load ? GFLoad : GFStore; break;
  case AddrMode::Flat:        T = Load ? FLoad : FStore; break;
  }
  return T[Dwords - 1];
}

// Memory element kind: a sub-dword byte/short, or one-or-more whole dwords.
enum class Elem { Byte, Short, Dword };

/// Element-sized helper load/store opcode used to initialize / read back the
/// memory the op-under-test touches. \p Dwords applies to Elem::Dword only.
/// Loads are zero-extending (UBYTE/USHORT) so the captured value is the raw
/// stored low byte/short.
unsigned elemHelpOpc(AddrMode Mode, bool Load, Elem El, unsigned Dwords) {
  using namespace llvm::AMDGPU;
  if (El == Elem::Dword)
    return helpOpc(Mode, Load, Dwords);
  struct Row { unsigned GS, GF, F; };
  // {GLOBAL_SADDR, GLOBAL(no saddr), FLAT}
  const Row LoadByte{GLOBAL_LOAD_UBYTE_SADDR, GLOBAL_LOAD_UBYTE, FLAT_LOAD_UBYTE};
  const Row StoreByte{GLOBAL_STORE_BYTE_SADDR, GLOBAL_STORE_BYTE, FLAT_STORE_BYTE};
  const Row LoadShort{GLOBAL_LOAD_USHORT_SADDR, GLOBAL_LOAD_USHORT,
                      FLAT_LOAD_USHORT};
  const Row StoreShort{GLOBAL_STORE_SHORT_SADDR, GLOBAL_STORE_SHORT,
                       FLAT_STORE_SHORT};
  const Row &R = El == Elem::Byte ? (Load ? LoadByte : StoreByte)
                                  : (Load ? LoadShort : StoreShort);
  switch (Mode) {
  case AddrMode::GlobalSaddr: return R.GS;
  case AddrMode::GlobalFlat:  return R.GF;
  case AddrMode::Flat:        return R.F;
  }
  return R.F;
}

/// Build a FLAT-family memory instruction, filling operands by named role.
/// \p Reg is the vdst (loads) or vdata (stores) -- an N-dword VGPR tuple.
void emitMem(EmitState &E, unsigned Opcode, llvm::MCRegister Reg,
             llvm::MCRegister Saddr, llvm::MCRegister VAddr, int64_t Cpol) {
  int VdstI = opIdx(Opcode, llvm::AMDGPU::OpName::vdst);
  int VdataI = opIdx(Opcode, llvm::AMDGPU::OpName::vdata);
  int SaddrI = opIdx(Opcode, llvm::AMDGPU::OpName::saddr);
  int VaddrI = opIdx(Opcode, llvm::AMDGPU::OpName::vaddr);
  int CpolI = opIdx(Opcode, llvm::AMDGPU::OpName::cpol);
  const llvm::MCInstrDesc &D = E.TII.get(Opcode);
  llvm::MachineInstrBuilder MIB = BuildMI(*E.BB, E.BB->end(), E.DL, D);
  for (unsigned I = 0; I < D.getNumOperands(); ++I) {
    if ((int)I == VdstI)
      MIB.addDef(Reg);
    else if ((int)I == VdataI)
      MIB.addReg(Reg);
    else if ((int)I == SaddrI)
      MIB.addReg(Saddr);
    else if ((int)I == VaddrI)
      MIB.addReg(VAddr);
    else if ((int)I == CpolI)
      MIB.addImm(Cpol);
    else
      MIB.addImm(0);
  }
}

} // namespace

llvm::Expected<KernelMFContext> buildFLAT(llvm::TargetMachine &TM,
                                          const InstrProfile &Profile,
                                          KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);
  unsigned Op = Profile.Opcode;

  //=== Addressing mode ==============================================//
  int SaddrIdx = opIdx(Op, llvm::AMDGPU::OpName::saddr);
  AddrMode Mode;
  if (Desc.TSFlags & llvm::SIInstrFlags::FlatScratch) {
    // Flat-scratch is handled by its own builder (instructions/Scratch.cpp),
    // which is matched before FLAT; this guard is defensive only.
    return makeError(Profile.Name + ": SCRATCH is handled by buildScratch");
  } else if (Desc.TSFlags & llvm::SIInstrFlags::FlatGlobal) {
    // GLOBAL with a saddr base pointer, or (no saddr) a full 64-bit vaddr.
    Mode = SaddrIdx < 0 ? AddrMode::GlobalFlat : AddrMode::GlobalSaddr;
  } else {
    Mode = AddrMode::Flat;
  }

  // Data width / element kind from the mnemonic (the FLAT data operand class is
  // a width-agnostic placeholder, so the width lives in the name).
  //   * RegDwords  = element register width (vdst / mem value / init): 1 for
  //                  byte/short/dword, N for DWORDXN and 2 for 64-bit atomics.
  //   * DataDwords = vdata register width = RegDwords, doubled for CMPSWAP
  //                  (the data operand packs {swap, cmp}).
  //   * El         = memory footprint kind (byte / short / dword).
  Elem El = Elem::Dword;
  unsigned RegDwords = 1;
  bool CmpSwap = false;
  {
    llvm::StringRef N = Profile.Name;
    if (N.contains("D16"))
      return makeError(Profile.Name + ": D16 (partial-register) not supported yet");
    CmpSwap = N.contains("CMPSWAP");
    if (N.contains("BYTE"))
      El = Elem::Byte;
    else if (N.contains("SHORT"))
      El = Elem::Short;
    else if (N.contains("DWORDX4"))
      RegDwords = 4;
    else if (N.contains("DWORDX3"))
      RegDwords = 3;
    else if (N.contains("DWORDX2"))
      RegDwords = 2;
    else if (N.contains("_X2") || N.contains("_B64") || N.contains("_F64"))
      RegDwords = 2; // 64-bit atomic (add/min/max/cmpswap_x2/...)
  }
  const unsigned DataDwords = CmpSwap ? 2 * RegDwords : RegDwords;

  int VdstIdx = opIdx(Op, llvm::AMDGPU::OpName::vdst);
  int VdataIdx = opIdx(Op, llvm::AMDGPU::OpName::vdata);

  bool Ld = Desc.mayLoad(), St = Desc.mayStore();
  MemKind Kind;
  if (VdstIdx >= 0 && Ld && !St)
    Kind = MemKind::Load;
  else if (VdataIdx >= 0 && St && !Ld)
    Kind = MemKind::Store;
  else if (Ld && St && VdstIdx < 0)
    Kind = MemKind::AtomicNoRet;
  else if (Ld && St && VdstIdx >= 0)
    Kind = MemKind::AtomicRet;
  else
    return makeError(Profile.Name + ": unhandled FLAT shape");

  bool NeedInit = Kind == MemKind::Load || Kind == MemKind::AtomicNoRet ||
                  Kind == MemKind::AtomicRet;
  bool HasVData = VdataIdx >= 0;
  bool CaptureVdst = Kind == MemKind::Load || Kind == MemKind::AtomicRet;
  bool Readback = Kind == MemKind::Store || Kind == MemKind::AtomicNoRet ||
                  Kind == MemKind::AtomicRet;

  //=== Layout =======================================================//
  // One 4-byte field per 32-bit component of each data operand.
  Layout = KernargLayout{};
  Layout.OutputPtrOffset = 0;
  Layout.DataBufPtrOffset = DataBufPtrKernargOffset;
  Layout.DataBufSizeBytes = FlatDataBufBytes;
  uint32_t KOff = 16, OOff = 0; // after the output + data-buffer pointers
  uint32_t InitInOff = UINT32_MAX, DataInOff = UINT32_MAX;
  uint32_t VdstOutOff = UINT32_MAX, MemOutOff = UINT32_MAX;
  auto addFields = [](uint32_t &Cur, auto &V, const char *Name, unsigned N) {
    uint32_t Base = Cur;
    for (unsigned D = 0; D < N; ++D, Cur += 4)
      V.push_back({Cur, 4, N > 1 ? llvm::formatv("{0}.{1}", Name, D).str()
                                 : std::string(Name)});
    return Base;
  };
  if (NeedInit)
    InitInOff = addFields(KOff, Layout.Inputs, "mem_init", RegDwords);
  if (HasVData)
    DataInOff = addFields(KOff, Layout.Inputs, "vdata", DataDwords);
  if (CaptureVdst)
    VdstOutOff = addFields(OOff, Layout.Outputs, "vdst", RegDwords);
  if (Readback)
    MemOutOff = addFields(OOff, Layout.Outputs, "mem_after", RegDwords);
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
  if (DataDwords > MaxDwords)
    return makeError(Profile.Name + ": data too wide");

  //=== Registers ====================================================//
  llvm::MCRegister Saddr = TRI.getMatchingSuperReg(
      sgpr32(SaddrSGPRIdx), llvm::AMDGPU::sub0, &llvm::AMDGPU::SGPR_64RegClass);
  llvm::MCRegister FlatAddr = TRI.getMatchingSuperReg(
      vgpr32(FlatAddrVGPRIdx), llvm::AMDGPU::sub0, &llvm::AMDGPU::VReg_64RegClass);
  if (!Saddr || !FlatAddr)
    return makeError(Profile.Name + ": failed to form an address register");
  llvm::MCRegister VOff = vgpr32(VOffsetVGPRIdx);
  // Element-width registers (vdst / mem value / init) and the vdata operand
  // (wider for CMPSWAP).
  llvm::MCRegister InitV = vgprTuple(TRI, InitVGPRBase, RegDwords);
  llvm::MCRegister VDataV = vgprTuple(TRI, VDataVGPRBase, DataDwords);
  llvm::MCRegister ReadbackV = vgprTuple(TRI, ReadbackVGPRBase, RegDwords);
  llvm::MCRegister VdstV = vgprTuple(TRI, VdstVGPRBase, RegDwords);
  llvm::MCRegister ZeroV = vgpr32(0);
  if (!InitV || !VDataV || !ReadbackV || !VdstV)
    return makeError(Profile.Name + ": failed to form a data tuple");

  llvm::MCRegister UseVAddr =
      Mode == AddrMode::GlobalSaddr ? VOff : FlatAddr;
  // Init / readback use the memory element size (byte/short/dword).
  unsigned StoreHelp = elemHelpOpc(Mode, /*Load=*/false, El, RegDwords);
  unsigned LoadHelp = elemHelpOpc(Mode, /*Load=*/true, El, RegDwords);

  //=== Binding log ==================================================//
  auto note = [&](RegBinding::RoleKind R, const std::string &N) {
    RegBinding B;
    B.Role = R;
    B.Note = N;
    return B;
  };
  if (NeedInit)
    KCtx.Prolog.push_back(note(
        RegBinding::LoadInputVGPR,
        llvm::formatv("P2 mem_init[{0}dw] <- ka+{1}; init mem[0]", RegDwords,
                      InitInOff)
            .str()));
  if (HasVData)
    KCtx.Prolog.push_back(note(
        RegBinding::LoadInputVGPR,
        llvm::formatv("P2 vdata[{0}dw] <- ka+{1}", DataDwords, DataInOff).str()));
  if (CaptureVdst) {
    RegBinding B = note(RegBinding::StoreOutput,
                        llvm::formatv("E1 store vdst[{0}dw] -> out+{1}", RegDwords,
                                      VdstOutOff)
                            .str());
    B.Reg = VdstV;
    B.Dwords = RegDwords;
    B.IsVGPR = true;
    B.OutputOffset = VdstOutOff;
    KCtx.Epilog.push_back(B);
  }
  if (Readback) {
    RegBinding B = note(RegBinding::MemReadback,
                        llvm::formatv("E5 read mem[0][{0}dw] -> out+{1}", RegDwords,
                                      MemOutOff)
                            .str());
    B.Dwords = RegDwords;
    B.OutputOffset = MemOutOff;
    KCtx.Epilog.push_back(B);
  }
  {
    const char *ElStr = El == Elem::Byte ? "b8" : El == Elem::Short ? "b16" : "dw";
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [" << toStr(Mode) << "-" << toStr(Kind) << "-"
       << RegDwords << ElStr << (CmpSwap ? "-cmpswap" : "") << "]\n  prolog:\n";
    for (const RegBinding &B : KCtx.Prolog)
      OS << "    " << B.Note << "\n";
    OS << "  epilog:\n";
    for (const RegBinding &B : KCtx.Epilog)
      OS << "    " << B.Note << "\n";
  }

  //=== Emit MIR =====================================================//
  EmitState E{S.BB, TII, S.KernargReg, S.OutPtrReg, {}};
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          S.OutPtrReg)
      .addReg(S.KernargReg).addImm(Layout.OutputPtrOffset).addImm(0);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          Saddr)
      .addReg(S.KernargReg).addImm(Layout.DataBufPtrOffset).addImm(0);

  // Phase A: load each data dword into a staging SGPR (init = element width,
  // vdata = data-operand width, which is wider for CMPSWAP).
  unsigned Stage = 0;
  unsigned InitStage0 = Stage;
  if (NeedInit) Stage += RegDwords;
  unsigned DataStage0 = Stage;
  if (HasVData) Stage += DataDwords;
  if (NeedInit)
    for (unsigned D = 0; D < RegDwords; ++D)
      emitScalarLoad(E, sgpr32(StageSGPRBase + InitStage0 + D), InitInOff + D * 4);
  if (HasVData)
    for (unsigned D = 0; D < DataDwords; ++D)
      emitScalarLoad(E, sgpr32(StageSGPRBase + DataStage0 + D), DataInOff + D * 4);
  emitWaitcnt(E);

  emitVMovImm(E, VOff, 0);
  if (Mode != AddrMode::GlobalSaddr) {
    // Materialize the 64-bit data-buffer address into the VGPR pair.
    emitVMovReg(E, vgpr32(FlatAddrVGPRIdx), sgpr32(SaddrSGPRIdx));
    emitVMovReg(E, vgpr32(FlatAddrVGPRIdx + 1), sgpr32(SaddrSGPRIdx + 1));
    // Plain FLAT implicit-uses FLAT_SCR (define it, unused for a global-aperture
    // access); GLOBAL with a vaddr does not.
    if (Mode == AddrMode::Flat) {
      BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
              llvm::AMDGPU::FLAT_SCR_LO).addImm(0);
      BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
              llvm::AMDGPU::FLAT_SCR_HI).addImm(0);
    }
  }
  // Phase B: move each data dword into its VGPR tuple component.
  if (NeedInit)
    for (unsigned D = 0; D < RegDwords; ++D)
      emitVMovReg(E, subDword(TRI, InitV, D, RegDwords),
                  sgpr32(StageSGPRBase + InitStage0 + D));
  if (HasVData)
    for (unsigned D = 0; D < DataDwords; ++D)
      emitVMovReg(E, subDword(TRI, VDataV, D, DataDwords),
                  sgpr32(StageSGPRBase + DataStage0 + D));

  if (NeedInit) {
    emitMem(E, StoreHelp, InitV, Saddr, UseVAddr, Glc);
    emitWaitcnt(E);
  }

  // The instruction under test, filled by named role.
  int CpolIdx = opIdx(Op, llvm::AMDGPU::OpName::cpol);
  int VaddrIdx = opIdx(Op, llvm::AMDGPU::OpName::vaddr);
  llvm::MachineInstrBuilder MIB = BuildMI(*S.BB, S.BB->end(), E.DL, Desc);
  for (unsigned I = 0; I < Desc.getNumOperands(); ++I) {
    if ((int)I == VdstIdx)
      MIB.addDef(VdstV);
    else if ((int)I == VdataIdx)
      MIB.addReg(VDataV);
    else if ((int)I == VaddrIdx)
      MIB.addReg(UseVAddr);
    else if ((int)I == SaddrIdx)
      MIB.addReg(Saddr);
    else if ((int)I == CpolIdx)
      MIB.addImm(Glc);
    else
      MIB.addImm(0);
  }
  KCtx.MI = MIB.getInstr();
  emitWaitcnt(E);

  if (Readback) {
    emitMem(E, LoadHelp, ReadbackV, Saddr, UseVAddr, Glc);
    emitWaitcnt(E);
  }

  emitVMovImm(E, ZeroV, 0);
  if (CaptureVdst)
    for (unsigned D = 0; D < RegDwords; ++D)
      emitGlobalStore(E, subDword(TRI, VdstV, D, RegDwords), ZeroV,
                      VdstOutOff + D * 4);
  if (Readback)
    for (unsigned D = 0; D < RegDwords; ++D)
      emitGlobalStore(E, subDword(TRI, ReadbackV, D, RegDwords), ZeroV,
                      MemOutOff + D * 4);
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

} // namespace luthier::test
