//===-- Scratch.cpp - SCRATCH (flat-scratch) reference-kernel builder ----===//
//
// Builds the reference kernel for the FLAT-scratch family (SCRATCH_LOAD_*,
// SCRATCH_STORE_*). Unlike GLOBAL/FLAT, the storage medium is the per-wave
// private (scratch) segment, which is not host-visible, so the kernel
// initializes a scratch slot by *storing* to it, runs the instruction under
// test, then loads the slot back and writes the result to the host-visible
// output buffer. Byte offset 0 (in bounds) and `glc` keep the round-trip
// coherent.
//
// The flat-scratch ABI is what makes this work at dispatch: setupScaffold with
// EnableFlatScratch adds "+enable-flat-scratch", so the subtarget drops the
// private_segment_buffer user SGPR, adds flat_scratch_init + the wavefront byte
// offset, and PEI emits `FLAT_SCR = flat_scratch_init + wave_offset`. A stack
// object gives the descriptor a non-zero private_segment_fixed_size so HSA backs
// the scratch and flat_scratch_init points at real memory.
//
// Two addressing modes:
//   * `_SADDR`: saddr = a 32-bit SGPR byte offset into the private segment
//     (vaddr is absent).
//   * plain: vaddr = a 32-bit VGPR byte offset (saddr is absent).
//
// Data width follows the mnemonic (DWORD / DWORDX2 / DWORDX3 / DWORDX4): the
// vdata/vdst operand is an N-dword VGPR tuple, and each 32-bit component becomes
// its own kernarg / output field.
//
//===----------------------------------------------------------------------===//
#include "InstructionBuilders.h"
#include "RefKernelSupport.h"

#include <SIDefines.h>
#include <SIInstrInfo.h>

#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier::test {

namespace {

constexpr unsigned OffsetSGPRIdx = 16;   // s16 = scratch byte offset (saddr)
constexpr unsigned OffsetVGPRIdx = 4;    // v4  = scratch byte offset (vaddr)
constexpr unsigned InitVGPRBase = 12;    // v[12:15] = init value tuple
constexpr unsigned VDataVGPRBase = 16;   // v[16:19] = store data tuple
constexpr unsigned VdstVGPRBase = 20;    // v[20:23] = load-dest tuple
constexpr unsigned ReadbackVGPRBase = 24;// v[24:27] = readback tuple
constexpr int64_t ScratchByteOffset = 0; // in-bounds slot
constexpr int64_t Glc = 1;
constexpr unsigned MaxDwords = 4;
/// Per-work-item scratch requested via a stack object, so the emitted kernel
/// descriptor's private_segment_fixed_size is non-zero (the runtime sizes and
/// backs the wave's scratch from it; a null base otherwise page-faults).
constexpr uint32_t ScratchPrivateBytes = 1024;

enum class MemKind { Load, Store, AtomicNoRet, AtomicRet };

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

/// Width-N SCRATCH helper store/load opcode. Indexed by dword count (1..4).
unsigned helpOpc(bool Saddr, bool Load, unsigned Dwords) {
  using namespace llvm::AMDGPU;
  static const unsigned SLoadS[4] = {SCRATCH_LOAD_DWORD_SADDR,
                                     SCRATCH_LOAD_DWORDX2_SADDR,
                                     SCRATCH_LOAD_DWORDX3_SADDR,
                                     SCRATCH_LOAD_DWORDX4_SADDR};
  static const unsigned SStoreS[4] = {SCRATCH_STORE_DWORD_SADDR,
                                      SCRATCH_STORE_DWORDX2_SADDR,
                                      SCRATCH_STORE_DWORDX3_SADDR,
                                      SCRATCH_STORE_DWORDX4_SADDR};
  static const unsigned SLoad[4] = {SCRATCH_LOAD_DWORD, SCRATCH_LOAD_DWORDX2,
                                    SCRATCH_LOAD_DWORDX3, SCRATCH_LOAD_DWORDX4};
  static const unsigned SStore[4] = {SCRATCH_STORE_DWORD, SCRATCH_STORE_DWORDX2,
                                     SCRATCH_STORE_DWORDX3, SCRATCH_STORE_DWORDX4};
  const unsigned *T = Saddr ? (Load ? SLoadS : SStoreS)
                            : (Load ? SLoad : SStore);
  return T[Dwords - 1];
}

// Memory element kind: sub-dword byte/short, or whole dword(s).
enum class Elem { Byte, Short, Dword };

/// Element-sized SCRATCH helper for init / readback (byte/short zero-extend).
unsigned elemHelpOpc(bool Saddr, bool Load, Elem El, unsigned Dwords) {
  using namespace llvm::AMDGPU;
  if (El == Elem::Dword)
    return helpOpc(Saddr, Load, Dwords);
  struct Row { unsigned S, NoS; };
  const Row LoadByte{SCRATCH_LOAD_UBYTE_SADDR, SCRATCH_LOAD_UBYTE};
  const Row StoreByte{SCRATCH_STORE_BYTE_SADDR, SCRATCH_STORE_BYTE};
  const Row LoadShort{SCRATCH_LOAD_USHORT_SADDR, SCRATCH_LOAD_USHORT};
  const Row StoreShort{SCRATCH_STORE_SHORT_SADDR, SCRATCH_STORE_SHORT};
  const Row &R = El == Elem::Byte ? (Load ? LoadByte : StoreByte)
                                  : (Load ? LoadShort : StoreShort);
  return Saddr ? R.S : R.NoS;
}

/// Emit a SCRATCH-family instruction, filling operands by named role. \p Reg is
/// the vdst (loads) or vdata (stores) N-dword tuple; \p OffsetS / \p OffsetV are
/// the SADDR / vaddr byte-offset registers (only the one the opcode has is used).
void emitScratch(EmitState &E, unsigned Opcode, llvm::MCRegister Reg,
                 llvm::MCRegister OffsetS, llvm::MCRegister OffsetV,
                 int64_t Cpol) {
  int VdstI = opIdx(Opcode, llvm::AMDGPU::OpName::vdst);
  int VdataI = opIdx(Opcode, llvm::AMDGPU::OpName::vdata);
  int SaddrI = opIdx(Opcode, llvm::AMDGPU::OpName::saddr);
  int VaddrI = opIdx(Opcode, llvm::AMDGPU::OpName::vaddr);
  int OffI = opIdx(Opcode, llvm::AMDGPU::OpName::offset);
  int CpolI = opIdx(Opcode, llvm::AMDGPU::OpName::cpol);
  const llvm::MCInstrDesc &D = E.TII.get(Opcode);
  llvm::MachineInstrBuilder MIB = BuildMI(*E.BB, E.BB->end(), E.DL, D);
  for (unsigned I = 0; I < D.getNumOperands(); ++I) {
    if ((int)I == VdstI)
      MIB.addDef(Reg);
    else if ((int)I == VdataI)
      MIB.addReg(Reg);
    else if ((int)I == SaddrI)
      MIB.addReg(OffsetS);
    else if ((int)I == VaddrI)
      MIB.addReg(OffsetV);
    else if ((int)I == OffI)
      MIB.addImm(ScratchByteOffset);
    else if ((int)I == CpolI)
      MIB.addImm(Cpol);
    else
      MIB.addImm(0);
  }
}

} // namespace

llvm::Expected<KernelMFContext> buildScratch(llvm::TargetMachine &TM,
                                             const InstrProfile &Profile,
                                             KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);
  unsigned Op = Profile.Opcode;

  // Element kind / width from the mnemonic. Scratch is load/store only (no
  // atomics), so the register width and the memory element coincide (Dwords).
  Elem El = Elem::Dword;
  unsigned Dwords = 1;
  {
    llvm::StringRef N = Profile.Name;
    if (N.contains("D16"))
      return makeError(Profile.Name + ": D16 (partial-register) not supported yet");
    if (N.contains("BYTE"))
      El = Elem::Byte;
    else if (N.contains("SHORT"))
      El = Elem::Short;
    else if (N.contains("DWORDX4"))
      Dwords = 4;
    else if (N.contains("DWORDX3"))
      Dwords = 3;
    else if (N.contains("DWORDX2"))
      Dwords = 2;
  }

  int VdstIdx = opIdx(Op, llvm::AMDGPU::OpName::vdst);
  int VdataIdx = opIdx(Op, llvm::AMDGPU::OpName::vdata);
  int SaddrIdx = opIdx(Op, llvm::AMDGPU::OpName::saddr);
  bool IsSaddr = SaddrIdx >= 0;

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
    return makeError(Profile.Name + ": unhandled SCRATCH shape");

  unsigned StoreHelp = elemHelpOpc(IsSaddr, /*Load=*/false, El, Dwords);
  unsigned LoadHelp = elemHelpOpc(IsSaddr, /*Load=*/true, El, Dwords);

  bool NeedInit = Kind == MemKind::Load || Kind == MemKind::AtomicNoRet ||
                  Kind == MemKind::AtomicRet;
  bool HasVData = VdataIdx >= 0;
  bool CaptureVdst = Kind == MemKind::Load || Kind == MemKind::AtomicRet;
  bool Readback = Kind == MemKind::Store || Kind == MemKind::AtomicNoRet ||
                  Kind == MemKind::AtomicRet;

  //=== Layout =======================================================//
  Layout = KernargLayout{};
  Layout.OutputPtrOffset = 0;
  Layout.PrivateSegmentSize = ScratchPrivateBytes;
  uint32_t KOff = 8, OOff = 0; // after the output pointer
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
    InitInOff = addFields(KOff, Layout.Inputs, "mem_init", Dwords);
  if (HasVData)
    DataInOff = addFields(KOff, Layout.Inputs, "vdata", Dwords);
  if (CaptureVdst)
    VdstOutOff = addFields(OOff, Layout.Outputs, "vdst", Dwords);
  if (Readback)
    MemOutOff = addFields(OOff, Layout.Outputs, "mem_after", Dwords);
  Layout.TotalSize = KOff;
  Layout.OutputBufSize = OOff;

  KernelMFContext KCtx;
  KCtx.KernelName = MachineKernelBuilder::getKernelName(Profile);
  auto ScafOrErr = setupScaffold(TM, KCtx, Layout.Inputs.size(),
                                 /*NumPtrArgs=*/1, /*EnableFlatScratch=*/true);
  if (!ScafOrErr)
    return ScafOrErr.takeError();
  Scaffold S = *ScafOrErr;
  const llvm::SIInstrInfo &TII = *S.TII;
  const llvm::SIRegisterInfo &TRI = *S.TRI;

  if (TII.pseudoToMCOpcode(Op) < 0)
    return makeError(Profile.Name + ": no MC encoding for this subtarget");
  if (Dwords > MaxDwords)
    return makeError(Profile.Name + ": data too wide");

  // Declare a private-segment allocation so the descriptor's
  // private_segment_fixed_size is non-zero (see the file comment).
  S.MF->getFrameInfo().CreateStackObject(ScratchPrivateBytes, llvm::Align(4),
                                         /*isSpillSlot=*/false);

  //=== Registers ====================================================//
  llvm::MCRegister OffsetS = sgpr32(OffsetSGPRIdx);
  llvm::MCRegister OffsetV = vgpr32(OffsetVGPRIdx);
  llvm::MCRegister InitV = vgprTuple(TRI, InitVGPRBase, Dwords);
  llvm::MCRegister VDataV = vgprTuple(TRI, VDataVGPRBase, Dwords);
  llvm::MCRegister ReadbackV = vgprTuple(TRI, ReadbackVGPRBase, Dwords);
  llvm::MCRegister VdstV = vgprTuple(TRI, VdstVGPRBase, Dwords);
  llvm::MCRegister ZeroV = vgpr32(0);
  if ((Dwords > 1) && (!InitV || !VDataV || !ReadbackV || !VdstV))
    return makeError(Profile.Name + ": failed to form a data tuple");

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
        llvm::formatv("P2 mem_init[{0}dw] <- ka+{1}; store scratch[0]", Dwords,
                      InitInOff)
            .str()));
  if (HasVData)
    KCtx.Prolog.push_back(note(
        RegBinding::LoadInputVGPR,
        llvm::formatv("P2 vdata[{0}dw] <- ka+{1}", Dwords, DataInOff).str()));
  if (CaptureVdst) {
    RegBinding B = note(RegBinding::StoreOutput,
                        llvm::formatv("E1 store vdst[{0}dw] -> out+{1}", Dwords,
                                      VdstOutOff)
                            .str());
    B.Reg = VdstV;
    B.Dwords = Dwords;
    B.IsVGPR = true;
    B.OutputOffset = VdstOutOff;
    KCtx.Epilog.push_back(B);
  }
  if (Readback) {
    RegBinding B = note(RegBinding::MemReadback,
                        llvm::formatv("E5 read scratch[0][{0}dw] -> out+{1}",
                                      Dwords, MemOutOff)
                            .str());
    B.Dwords = Dwords;
    B.OutputOffset = MemOutOff;
    KCtx.Epilog.push_back(B);
  }
  {
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [SCRATCH-" << (IsSaddr ? "SADDR-" : "")
       << toStr(Kind) << "-" << Dwords << "dw]\n  prolog:\n";
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

  // Byte offset registers = 0 (in-bounds slot).
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32), OffsetS)
      .addImm(ScratchByteOffset);
  emitVMovImm(E, OffsetV, ScratchByteOffset);

  // Phase A: load each data dword into a staging SGPR.
  unsigned Stage = 0;
  unsigned InitStage0 = Stage;
  if (NeedInit) Stage += Dwords;
  unsigned DataStage0 = Stage;
  if (HasVData) Stage += Dwords;
  if (NeedInit)
    for (unsigned D = 0; D < Dwords; ++D)
      emitScalarLoad(E, sgpr32(StageSGPRBase + InitStage0 + D), InitInOff + D * 4);
  if (HasVData)
    for (unsigned D = 0; D < Dwords; ++D)
      emitScalarLoad(E, sgpr32(StageSGPRBase + DataStage0 + D), DataInOff + D * 4);
  emitWaitcnt(E);
  if (NeedInit)
    for (unsigned D = 0; D < Dwords; ++D)
      emitVMovReg(E, subDword(TRI, InitV, D, Dwords),
                  sgpr32(StageSGPRBase + InitStage0 + D));
  if (HasVData)
    for (unsigned D = 0; D < Dwords; ++D)
      emitVMovReg(E, subDword(TRI, VDataV, D, Dwords),
                  sgpr32(StageSGPRBase + DataStage0 + D));

  // Initialize the scratch slot so a load/atomic observes a known value.
  if (NeedInit) {
    emitScratch(E, StoreHelp, InitV, OffsetS, OffsetV, Glc);
    emitWaitcnt(E);
  }

  // The instruction under test, filled by named role.
  int VaddrIdx = opIdx(Op, llvm::AMDGPU::OpName::vaddr);
  int OffIdx = opIdx(Op, llvm::AMDGPU::OpName::offset);
  int CpolIdx = opIdx(Op, llvm::AMDGPU::OpName::cpol);
  llvm::MachineInstrBuilder MIB = BuildMI(*S.BB, S.BB->end(), E.DL, Desc);
  for (unsigned I = 0; I < Desc.getNumOperands(); ++I) {
    if ((int)I == VdstIdx)
      MIB.addDef(VdstV);
    else if ((int)I == VdataIdx)
      MIB.addReg(VDataV);
    else if ((int)I == SaddrIdx)
      MIB.addReg(OffsetS);
    else if ((int)I == VaddrIdx)
      MIB.addReg(OffsetV);
    else if ((int)I == OffIdx)
      MIB.addImm(ScratchByteOffset);
    else if ((int)I == CpolIdx)
      MIB.addImm(Glc);
    else
      MIB.addImm(0);
  }
  KCtx.MI = MIB.getInstr();
  emitWaitcnt(E);

  // Read the scratch slot back for the store / atomic cases.
  if (Readback) {
    emitScratch(E, LoadHelp, ReadbackV, OffsetS, OffsetV, Glc);
    emitWaitcnt(E);
  }

  emitVMovImm(E, ZeroV, 0);
  if (CaptureVdst)
    for (unsigned D = 0; D < Dwords; ++D)
      emitGlobalStore(E, subDword(TRI, VdstV, D, Dwords), ZeroV,
                      VdstOutOff + D * 4);
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
