//===-- SMEM.cpp - SMEM (scalar memory) reference-kernel builder --------===//
//
// Builds the reference kernel for the scalar-memory load family. On gfx9 the
// SMEM data ops are loads only (there are no scalar stores/atomics): a value is
// read from the 64-bit byte address `sbase + offset` into the scalar `sdst`
// register (a 1/2/4/8/16-dword SGPR tuple, width taken from the mnemonic).
//
// Like the FLAT builder, a host-visible fine-grained global buffer stands in
// for device memory. The kernel:
//   1. loads the data-buffer pointer into an SGPR pair (sbase),
//   2. initializes buffer[0..N-1] from kernarg via vector GLOBAL_STOREs (glc),
//   3. invalidates the scalar cache (S_DCACHE_INV) so the S_LOAD does not read
//      stale K$ lines written by step 2 (the vector store path and the scalar
//      cache are separate); the S_LOAD under test also carries glc,
//   4. runs the S_LOAD under test,
//   5. captures each sdst dword (SGPR -> VGPR -> output buffer).
// So the existing output comparison covers everything and the functional check
// is simply "the load returns the value the kernel wrote".
//
// S_BUFFER_LOAD_* uses a 128-bit buffer resource descriptor (V#) in sbase, not
// a plain pointer; that needs the MUBUF-style V# path and is rejected cleanly.
//
//===----------------------------------------------------------------------===//
#include "InstructionBuilders.h"
#include "RefKernelSupport.h"

#include <SIInstrInfo.h>

#include <llvm/MC/MCInstrDesc.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

namespace luthier::test {

namespace {

constexpr uint32_t DataBufPtrKernargOffset = 8;
constexpr unsigned BufPtrSGPRIdx = 16;   // s[16:17] = 64-bit data-buffer pointer
constexpr unsigned SOffsetSGPRIdx = 18;  // s18 = 0 scalar offset (SGPR forms)
constexpr unsigned VSharpSGPRBase = 24;  // s[24:27] = V# resource descriptor
constexpr unsigned InitStageSGPRIdx = 8; // s8 = per-dword init staging (reused)
constexpr unsigned SDstSGPRBase = 32;    // s[32:..] = sdst tuple (up to 16 dw)
constexpr unsigned ZeroVGPRIdx = 0;      // v0 = 0 (voffset for GLOBAL_STORE)
constexpr unsigned DataVGPRIdx = 1;      // v1 = per-dword data staging (reused)
constexpr uint32_t SMEMDataBufBytes = 4096;
constexpr int64_t Glc = 1;
constexpr unsigned MaxDwords = 16;

int opIdx(unsigned Opcode, llvm::AMDGPU::OpName N) {
  return llvm::AMDGPU::getNamedOperandIdx(Opcode, N);
}

/// buffer[ByteOff] = DataV, via a vector GLOBAL_STORE_DWORD_SADDR (glc) to the
/// data buffer whose base pointer is in \p Saddr. \p ZeroV is a VGPR holding 0
/// (the per-lane voffset).
void storeDwordToBuf(EmitState &E, llvm::MCRegister Saddr,
                     llvm::MCRegister DataV, llvm::MCRegister ZeroV,
                     uint32_t ByteOff) {
  BuildMI(*E.BB, E.BB->end(), E.DL,
          E.TII.get(llvm::AMDGPU::GLOBAL_STORE_DWORD_SADDR))
      .addReg(ZeroV)  // vaddr
      .addReg(DataV)  // vdata
      .addReg(Saddr)  // saddr
      .addImm(ByteOff)
      .addImm(Glc);   // cpol
}

} // namespace

llvm::Expected<KernelMFContext> buildSMEM(llvm::TargetMachine &TM,
                                          const InstrProfile &Profile,
                                          KernargLayout &Layout) {
  const llvm::MCInstrInfo &MCII = *TM.getMCInstrInfo();
  const llvm::MCRegisterInfo &MRI = TM.getMCRegisterInfo();
  const llvm::MCInstrDesc &Desc = MCII.get(Profile.Opcode);
  unsigned Op = Profile.Opcode;

  //=== Shape guards =================================================//
  // Loads only (no scalar stores/atomics on this subtarget).
  if (!Desc.mayLoad() || Desc.mayStore())
    return makeError(Profile.Name + ": non-load SMEM not supported");
  int SdstIdx = opIdx(Op, llvm::AMDGPU::OpName::sdst);
  int SbaseIdx = opIdx(Op, llvm::AMDGPU::OpName::sbase);
  if (SdstIdx < 0 || SbaseIdx < 0)
    return makeError(Profile.Name + ": SMEM without sdst/sbase (e.g. "
                     "S_MEMTIME / S_DCACHE_*) not supported");
  // S_BUFFER_LOAD passes a 128-bit V# resource descriptor in sbase; a plain
  // scalar load passes a 64-bit pointer. Either is handled (the V# is built
  // in-kernel from the data-buffer pointer); anything else is rejected.
  bool IsBuffer;
  {
    const llvm::MCOperandInfo &SB = Desc.operands()[SbaseIdx];
    unsigned Bits =
        SB.RegClass < 0 ? 0 : MRI.getRegClass(SB.RegClass).getSizeInBits();
    if (Bits == 64)
      IsBuffer = false;
    else if (Bits == 128)
      IsBuffer = true;
    else
      return makeError(Profile.Name + ": unexpected sbase width");
  }

  //=== Data width from the mnemonic ================================//
  unsigned Dwords = 1;
  {
    llvm::StringRef N = Profile.Name;
    if (N.contains("DWORDX16"))
      Dwords = 16;
    else if (N.contains("DWORDX8"))
      Dwords = 8;
    else if (N.contains("DWORDX4"))
      Dwords = 4;
    else if (N.contains("DWORDX2"))
      Dwords = 2;
    else if (N.contains("DWORD"))
      Dwords = 1;
    else
      return makeError(Profile.Name + ": sub-dword scalar load not supported yet");
  }
  if (Dwords > MaxDwords || !sgprTupleClass(Dwords))
    return makeError(Profile.Name + ": unsupported scalar-load width");

  bool HasSOffset = opIdx(Op, llvm::AMDGPU::OpName::soffset) >= 0;

  //=== Layout ======================================================//
  // One 4-byte field per dword: kernarg init input + captured output.
  Layout = KernargLayout{};
  Layout.OutputPtrOffset = 0;
  Layout.DataBufPtrOffset = DataBufPtrKernargOffset;
  Layout.DataBufSizeBytes = SMEMDataBufBytes;
  uint32_t KOff = 16, OOff = 0; // after the output + data-buffer pointers
  uint32_t InitInOff = KOff, SdstOutOff = OOff;
  for (unsigned D = 0; D < Dwords; ++D, KOff += 4)
    Layout.Inputs.push_back(
        {KOff, 4, Dwords > 1 ? llvm::formatv("mem_init.{0}", D).str()
                             : std::string("mem_init")});
  for (unsigned D = 0; D < Dwords; ++D, OOff += 4)
    Layout.Outputs.push_back(
        {OOff, 4, Dwords > 1 ? llvm::formatv("sdst.{0}", D).str()
                             : std::string("sdst")});
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

  //=== Registers ===================================================//
  // The raw 64-bit data-buffer pointer (used for the vector init stores, and
  // as the plain-scalar sbase / the V# base address).
  llvm::MCRegister BufPtr = TRI.getMatchingSuperReg(
      sgpr32(BufPtrSGPRIdx), llvm::AMDGPU::sub0, &llvm::AMDGPU::SGPR_64RegClass);
  // The sbase operand: a 64-bit pointer (scalar load) or a 128-bit V# resource
  // descriptor built in s[VSharpSGPRBase:+3] (buffer load).
  llvm::MCRegister VSharp =
      IsBuffer ? TRI.getMatchingSuperReg(sgpr32(VSharpSGPRBase),
                                         llvm::AMDGPU::sub0,
                                         &llvm::AMDGPU::SGPR_128RegClass)
               : llvm::MCRegister();
  llvm::MCRegister Sbase = IsBuffer ? VSharp : BufPtr;
  llvm::MCRegister SDst = sgprTuple(TRI, SDstSGPRBase, Dwords);
  if (!BufPtr || !Sbase || !SDst)
    return makeError(Profile.Name + ": failed to form sbase / sdst registers");
  // Validate sbase / sdst against their operand classes (self-corrects the
  // MC-vs-codegen regclass numbering artifact, and rejects an unformable tuple).
  {
    const llvm::MCOperandInfo &SB = Desc.operands()[SbaseIdx];
    const llvm::MCOperandInfo &SD = Desc.operands()[SdstIdx];
    if (SB.RegClass < 0 || !TRI.getRegClass(SB.RegClass)->contains(Sbase))
      return makeError(Profile.Name + ": sbase not accepted by operand class");
    if (SD.RegClass < 0 || !TRI.getRegClass(SD.RegClass)->contains(SDst))
      return makeError(Profile.Name + ": sdst tuple not accepted by operand class");
  }
  llvm::MCRegister ZeroV = vgpr32(ZeroVGPRIdx);
  llvm::MCRegister DataV = vgpr32(DataVGPRIdx);

  //=== Binding log =================================================//
  {
    llvm::raw_string_ostream OS(KCtx.BindingLog);
    OS << Profile.Name << "  [SMEM-" << (IsBuffer ? "Buffer" : "Scalar")
       << "-Load-" << Dwords << "dw]\n"
       << "  prolog:\n"
       << "    P2 mem_init[" << Dwords << "dw] <- ka+" << InitInOff
       << "; init buffer[0]\n"
       << (IsBuffer ? "    P6 build V# resource descriptor from buf ptr\n" : "")
       << "    (S_DCACHE_INV before the load)\n"
       << "  epilog:\n"
       << "    E2 store sdst[" << Dwords << "dw] -> out+" << SdstOutOff << "\n";
  }

  //=== Emit MIR ====================================================//
  EmitState E{S.BB, TII, S.KernargReg, S.OutPtrReg, {}};
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          S.OutPtrReg)
      .addReg(S.KernargReg).addImm(Layout.OutputPtrOffset).addImm(0);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM),
          BufPtr)
      .addReg(S.KernargReg).addImm(Layout.DataBufPtrOffset).addImm(0);
  emitWaitcnt(E);

  // For a buffer load, assemble the raw V# resource descriptor in s[24:27] from
  // the data-buffer pointer: word0/1 = base address (high dword masked to the
  // 48-bit address field), word2 = num_records (bytes), word3 = raw-format flags.
  if (IsBuffer) {
    auto sMov = [&](unsigned DstIdx, auto Src) {
      BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
              sgpr32(DstIdx))
          .add(Src);
    };
    sMov(VSharpSGPRBase + 0, llvm::MachineOperand::CreateReg(
                                 sgpr32(BufPtrSGPRIdx + 0), /*isDef=*/false));
    sMov(VSharpSGPRBase + 1, llvm::MachineOperand::CreateReg(
                                 sgpr32(BufPtrSGPRIdx + 1), /*isDef=*/false));
    BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_AND_B32),
            sgpr32(VSharpSGPRBase + 1))
        .addReg(sgpr32(VSharpSGPRBase + 1)).addImm(0xFFFF);
    sMov(VSharpSGPRBase + 2, llvm::MachineOperand::CreateImm(SMEMDataBufBytes));
    sMov(VSharpSGPRBase + 3,
       llvm::MachineOperand::CreateImm(
           vSharpWord3(S.MF->getSubtarget())));
  }

  // Initialize buffer[D] = mem_init[D] via vector stores (glc), one dword at a
  // time (reusing the staging SGPR and data VGPR).
  emitVMovImm(E, ZeroV, 0);
  for (unsigned D = 0; D < Dwords; ++D) {
    emitScalarLoad(E, sgpr32(InitStageSGPRIdx), InitInOff + D * 4);
    emitWaitcnt(E);
    emitVMovReg(E, DataV, sgpr32(InitStageSGPRIdx));
    storeDwordToBuf(E, BufPtr, DataV, ZeroV, D * 4);
  }
  emitWaitcnt(E); // ensure the stores reached L2 before invalidating K$

  // Invalidate the scalar cache so the load sees the freshly written data.
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_DCACHE_INV));
  emitWaitcnt(E);

  // If the offset lives in an SGPR (S_LOAD_*_SGPR forms), zero it.
  if (HasSOffset)
    BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_MOV_B32),
            sgpr32(SOffsetSGPRIdx)).addImm(0);

  // The instruction under test, filled by named role. Address is fixed at
  // buffer[0]; only the data is exercised. glc keeps the read coherent.
  int SoffsetIdx = opIdx(Op, llvm::AMDGPU::OpName::soffset);
  int OffsetIdx = opIdx(Op, llvm::AMDGPU::OpName::offset);
  int CpolIdx = opIdx(Op, llvm::AMDGPU::OpName::cpol);
  llvm::MachineInstrBuilder MIB = BuildMI(*S.BB, S.BB->end(), E.DL, Desc);
  for (unsigned I = 0; I < Desc.getNumOperands(); ++I) {
    if ((int)I == SdstIdx)
      MIB.addDef(SDst);
    else if ((int)I == SbaseIdx)
      MIB.addReg(Sbase);
    else if ((int)I == SoffsetIdx)
      MIB.addReg(sgpr32(SOffsetSGPRIdx));
    else if ((int)I == OffsetIdx)
      MIB.addImm(0);
    else if ((int)I == CpolIdx)
      MIB.addImm(Glc);
    else
      MIB.addImm(0);
  }
  KCtx.MI = MIB.getInstr();
  emitWaitcnt(E);

  // Capture each sdst dword: SGPR -> VGPR -> output buffer.
  for (unsigned D = 0; D < Dwords; ++D) {
    emitVMovReg(E, DataV, subDword(TRI, SDst, D, Dwords));
    emitGlobalStore(E, DataV, ZeroV, SdstOutOff + D * 4);
  }
  emitWaitcnt(E);
  BuildMI(*S.BB, S.BB->end(), E.DL, TII.get(llvm::AMDGPU::S_ENDPGM)).addImm(0);

  finalizeMF(*S.MF);
  return std::move(KCtx);
}

} // namespace luthier::test
