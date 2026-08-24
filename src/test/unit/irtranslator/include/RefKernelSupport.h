//===-- RefKernelSupport.h --------------------------------------*- C++ -*-===//
//
// Shared support for the reference-kernel instruction-class builders
// (instructions/VOP.cpp, instructions/DS.cpp): the register reservation, the
// MachineFunction scaffold (IR shell + ABI user-SGPRs), and the small MIR emit
// primitives. Keeping these here lets each instruction class live in its own
// translation unit while sharing one prolog/epilog vocabulary.
//
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_REF_KERNEL_SUPPORT_H
#define LUTHIER_TEST_REF_KERNEL_SUPPORT_H

#include "MachineKernelBuilder.h"

#include <SIInstrInfo.h>
#include <SIRegisterInfo.h>

#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MathExtras.h>

namespace luthier::test {

//===----------------------------------------------------------------------===//
// Register reservation inside the reference kernel.
//
//   SGPR  s4:5    kernarg segment pointer (ABI preloaded)
//         s6:7    output buffer base pointer (loaded from kernarg)
//         s8..s15 input-staging SGPRs (kernarg dword -> SGPR before V_MOV /
//                 special-set; assigned deterministically across emit phases)
//         s16..19 scalar operand inputs (P1)
//         VCC/M0/EXEC  addressed by name
//   VGPR  v0      zero (voffset for GLOBAL_STORE_DWORD_SADDR)
//         v1..v3  epilog store-staging VGPRs
//         v4      DS LDS address (constant offset)
//         v5      DS LDS readback temp
//         v10..19 vector operand inputs (P2) / DS data VGPRs
//         v20..29 vector operand outputs (E1) / tied defs / DS vdst
//===----------------------------------------------------------------------===//
inline constexpr unsigned OutPtrSGPRIdx = 6;
inline constexpr unsigned StageSGPRBase = 8;
inline constexpr unsigned ScalarInSGPRBase = 16;
inline constexpr unsigned ScalarOutSGPRBase = 20;
inline constexpr unsigned StoreStageVGPRBase = 1;
inline constexpr unsigned DSAddrVGPR = 4;
inline constexpr unsigned DSReadbackVGPR = 5;
inline constexpr unsigned InputVGPRBase = 10;
inline constexpr unsigned OutputVGPRBase = 20;

/// Fixed literal encoded into instructions with a simm32 operand (P7).
inline constexpr int64_t FixedLiteral = 0x3F800000; // 1.0f
/// LDS bytes reserved for DS reference kernels (via the AQL packet).
inline constexpr uint32_t DSGroupSegmentSize = 4096;

inline llvm::Error makeError(const llvm::Twine &Msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), Msg);
}

/// The wavefront size (32 on RDNA wave32, 64 on CDNA / wave64) of \p TM's
/// subtarget. The full-wave / cross-lane builders size a wave from this instead
/// of assuming 64, so the same code runs correctly in either mode.
unsigned waveSize(const llvm::TargetMachine &TM);

/// Word 3 of a raw, 32-bit-element buffer resource descriptor (V#) for \p STI.
///
/// The encoding is generation-dependent, so this must not be a single
/// constant: GFX10 dropped GFX9's {num_format, data_format} pair in favour of
/// one 7-bit unified FORMAT field, and added RESOURCE_LEVEL and OOB_SELECT.
/// Getting OOB_SELECT wrong is silent -- the buffer op still issues, but every
/// access is treated as out of bounds, so loads return 0 and stores are
/// dropped. Mirrors \c llvm::SIInstrInfo::getDefaultRsrcDataFormat.
uint32_t vSharpWord3(const llvm::MCSubtargetInfo &STI);

/// The MTBUF `format` immediate for the typed-buffer element described by the
/// GFX9-style \p Dfmt / \p Nfmt pair, encoded for \p STI's generation: GFX9 and
/// earlier pack it as `dfmt | nfmt << 4`, while GFX10+ takes a single unified
/// format id (converted here via \c llvm::AMDGPU::convertDfmtNfmt2Ufmt so the
/// mapping stays LLVM's).
int64_t mtbufFormat(const llvm::MCSubtargetInfo &STI, unsigned Dfmt,
                    unsigned Nfmt);

//===----------------------------------------------------------------------===//
// Register helpers.
//===----------------------------------------------------------------------===//
inline bool isVGPR32(llvm::MCRegister Reg) {
  return llvm::AMDGPU::VGPR_32RegClass.contains(Reg);
}
inline llvm::MCRegister sgpr32(unsigned Idx) {
  return llvm::AMDGPU::SGPR_32RegClass.getRegister(Idx);
}
inline llvm::MCRegister vgpr32(unsigned Idx) {
  return llvm::AMDGPU::VGPR_32RegClass.getRegister(Idx);
}

/// Pick a concrete 32-bit physical register accepted by \p RC, preferring a
/// VGPR. Returns an invalid register if neither fits.
inline llvm::MCRegister pick32BitReg(const llvm::TargetRegisterClass *RC,
                                     unsigned &NextVGPR, unsigned &NextSGPR) {
  llvm::MCRegister V = vgpr32(NextVGPR);
  if (RC->contains(V)) {
    ++NextVGPR;
    return V;
  }
  llvm::MCRegister S = sgpr32(NextSGPR);
  if (RC->contains(S)) {
    ++NextSGPR;
    return S;
  }
  return {};
}

/// The AMDGPU sub-register index for the \p D-th 32-bit component. The subreg
/// index enum is not guaranteed contiguous, so map explicitly. Covers up to 16
/// dwords (SMEM S_LOAD_DWORDX16).
inline unsigned dwordSubIdx(unsigned D) {
  static const unsigned Idx[] = {
      llvm::AMDGPU::sub0,  llvm::AMDGPU::sub1,  llvm::AMDGPU::sub2,
      llvm::AMDGPU::sub3,  llvm::AMDGPU::sub4,  llvm::AMDGPU::sub5,
      llvm::AMDGPU::sub6,  llvm::AMDGPU::sub7,  llvm::AMDGPU::sub8,
      llvm::AMDGPU::sub9,  llvm::AMDGPU::sub10, llvm::AMDGPU::sub11,
      llvm::AMDGPU::sub12, llvm::AMDGPU::sub13, llvm::AMDGPU::sub14,
      llvm::AMDGPU::sub15};
  return Idx[D];
}

/// The \p D-th 32-bit component of a (possibly multi-dword) register.
inline llvm::MCRegister subDword(const llvm::SIRegisterInfo &TRI,
                                 llvm::MCRegister Reg, unsigned D,
                                 unsigned Dwords) {
  return Dwords <= 1 ? Reg : TRI.getSubReg(Reg, dwordSubIdx(D));
}

/// True if \p Reg's first 32-bit component is a VGPR.
inline bool isVGPRTuple(const llvm::SIRegisterInfo &TRI, llvm::MCRegister Reg,
                        unsigned Dwords) {
  return llvm::AMDGPU::VGPR_32RegClass.contains(subDword(TRI, Reg, 0, Dwords));
}

/// Concrete VGPR / SGPR tuple register class for a given dword count. These are
/// the physically-defined tuple classes (unlike operand classes such as
/// VSrc_b64), so getMatchingSuperReg can form the super-register from them.
inline const llvm::TargetRegisterClass *vgprTupleClass(unsigned Dwords) {
  switch (Dwords) {
  case 1: return &llvm::AMDGPU::VGPR_32RegClass;
  case 2: return &llvm::AMDGPU::VReg_64RegClass;
  case 3: return &llvm::AMDGPU::VReg_96RegClass;
  case 4: return &llvm::AMDGPU::VReg_128RegClass;
  default: return nullptr;
  }
}
inline const llvm::TargetRegisterClass *sgprTupleClass(unsigned Dwords) {
  switch (Dwords) {
  case 1: return &llvm::AMDGPU::SGPR_32RegClass;
  case 2: return &llvm::AMDGPU::SGPR_64RegClass;
  case 3: return &llvm::AMDGPU::SGPR_96RegClass;
  case 4: return &llvm::AMDGPU::SGPR_128RegClass;
  case 8: return &llvm::AMDGPU::SGPR_256RegClass;
  case 16: return &llvm::AMDGPU::SGPR_512RegClass;
  default: return nullptr;
  }
}

/// The N-dword SGPR tuple based at s\p Base (s[Base:Base+N-1]). \p Base must be
/// suitably aligned for the tuple class. Returns s\p Base itself for a single
/// dword, or an invalid register if the tuple cannot be formed.
inline llvm::MCRegister sgprTuple(const llvm::SIRegisterInfo &TRI, unsigned Base,
                                  unsigned Dwords) {
  if (Dwords <= 1)
    return sgpr32(Base);
  const llvm::TargetRegisterClass *C = sgprTupleClass(Dwords);
  return C ? TRI.getMatchingSuperReg(sgpr32(Base), llvm::AMDGPU::sub0, C)
           : llvm::MCRegister();
}

/// The N-dword VGPR tuple based at v\p Base (v[Base:Base+N-1]). \p Base must be
/// suitably aligned (even for 64-bit, /4 for 128-bit). Returns v\p Base itself
/// for a single dword.
inline llvm::MCRegister vgprTuple(const llvm::SIRegisterInfo &TRI, unsigned Base,
                                  unsigned Dwords) {
  if (Dwords <= 1)
    return vgpr32(Base);
  return TRI.getMatchingSuperReg(vgpr32(Base), llvm::AMDGPU::sub0,
                                 vgprTupleClass(Dwords));
}

/// Allocate a register of \p RC's full width (N = size/32 dwords), preferring a
/// VGPR tuple then an SGPR tuple, from the given cursors (aligned to N). The
/// tuple is formed from the concrete VGPR/SGPR tuple class and then checked for
/// membership in the operand class \p RC. Returns invalid if neither fits (e.g.
/// an AGPR-only class or an unsupported width).
inline llvm::MCRegister allocOperandReg(const llvm::SIRegisterInfo &TRI,
                                        const llvm::TargetRegisterClass *RC,
                                        unsigned &NextVGPR, unsigned &NextSGPR) {
  unsigned Dwords = TRI.getRegSizeInBits(*RC) / 32;
  if (const llvm::TargetRegisterClass *VC = vgprTupleClass(Dwords)) {
    unsigned BaseV = llvm::alignTo(NextVGPR, Dwords);
    llvm::MCRegister VTup =
        Dwords == 1 ? vgpr32(BaseV)
                    : TRI.getMatchingSuperReg(vgpr32(BaseV), llvm::AMDGPU::sub0,
                                              VC);
    if (VTup && RC->contains(VTup)) {
      NextVGPR = BaseV + Dwords;
      return VTup;
    }
  }
  if (const llvm::TargetRegisterClass *SC = sgprTupleClass(Dwords)) {
    unsigned BaseS = llvm::alignTo(NextSGPR, Dwords);
    llvm::MCRegister STup =
        Dwords == 1 ? sgpr32(BaseS)
                    : TRI.getMatchingSuperReg(sgpr32(BaseS), llvm::AMDGPU::sub0,
                                              SC);
    if (STup && RC->contains(STup)) {
      NextSGPR = BaseS + Dwords;
      return STup;
    }
  }
  return {};
}

inline bool implicitContains(const InstrProfile &P, bool Defs,
                             llvm::StringRef Name,
                             const llvm::MCRegisterInfo &MRI) {
  for (llvm::MCRegister R : (Defs ? P.ImplicitDefs : P.ImplicitUses))
    if (llvm::StringRef(MRI.getName(R)) == Name)
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Kernel scaffolding (IR shell + ABI user-SGPRs + first basic block).
//===----------------------------------------------------------------------===//
struct Scaffold {
  llvm::MachineFunction *MF;
  const llvm::SIInstrInfo *TII;
  const llvm::SIRegisterInfo *TRI;
  llvm::MachineBasicBlock *BB;
  llvm::MCRegister KernargReg;
  llvm::MCRegister OutPtrReg;
};

/// Build the kernel IR shell (\p NumPtrArgs `ptr addrspace(1)` args followed by
/// \p NumI32Inputs i32 args — the signature that drives the kernarg ABI),
/// create the MachineFunction, allocate the preloaded user SGPRs in ABI order
/// (ISel normally does this), and open a basic block. Fills
/// KCtx.Ctx/Mod/MMIWP/MF; KCtx.KernelName must be set first.
///
/// The first pointer arg is always the output buffer (kernarg offset 0). A
/// second pointer (offset 8) is the FLAT/GLOBAL data buffer.
/// \p FlatWorkGroupSize > 1 requests a full-wave dispatch: the flat work-group
/// size is set to that value and the VGPR workitem-id (x) is enabled, so `v0`
/// holds each lane's id at kernel entry (added as a live-in). Used by the
/// per-lane / cross-lane builders (e.g. DS permute).
llvm::Expected<Scaffold> setupScaffold(llvm::TargetMachine &TM,
                                       KernelMFContext &KCtx,
                                       unsigned NumI32Inputs,
                                       unsigned NumPtrArgs = 1,
                                       bool EnableFlatScratch = false,
                                       unsigned FlatWorkGroupSize = 1);

/// Declare the function already selected / register-allocated and freeze the
/// reserved register set, so the pipeline (started at prologepilog) accepts it.
void finalizeMF(llvm::MachineFunction &MF);

//===----------------------------------------------------------------------===//
// MIR emit primitives.
//===----------------------------------------------------------------------===//
struct EmitState {
  llvm::MachineBasicBlock *BB;
  const llvm::SIInstrInfo &TII;
  llvm::MCRegister KernargReg;
  llvm::MCRegister OutPtrReg;
  llvm::DebugLoc DL;
};

inline void emitScalarLoad(EmitState &E, llvm::MCRegister Dst, uint32_t Offset) {
  BuildMI(*E.BB, E.BB->end(), E.DL, E.TII.get(llvm::AMDGPU::S_LOAD_DWORD_IMM),
          Dst)
      .addReg(E.KernargReg)
      .addImm(Offset)
      .addImm(0);
}
inline void emitWaitcnt(EmitState &E) {
  BuildMI(*E.BB, E.BB->end(), E.DL, E.TII.get(llvm::AMDGPU::S_WAITCNT))
      .addImm(0);
}
inline void emitVMovImm(EmitState &E, llvm::MCRegister Dst, int64_t Imm) {
  BuildMI(*E.BB, E.BB->end(), E.DL, E.TII.get(llvm::AMDGPU::V_MOV_B32_e32), Dst)
      .addImm(Imm);
}
inline void emitVMovReg(EmitState &E, llvm::MCRegister Dst,
                        llvm::MCRegister Src) {
  BuildMI(*E.BB, E.BB->end(), E.DL, E.TII.get(llvm::AMDGPU::V_MOV_B32_e32), Dst)
      .addReg(Src);
}
inline void emitGlobalStore(EmitState &E, llvm::MCRegister DataVGPR,
                            llvm::MCRegister ZeroVGPR, uint32_t OutputOffset) {
  BuildMI(*E.BB, E.BB->end(), E.DL,
          E.TII.get(llvm::AMDGPU::GLOBAL_STORE_DWORD_SADDR))
      .addReg(ZeroVGPR)    // vaddr
      .addReg(DataVGPR)    // vdata
      .addReg(E.OutPtrReg) // saddr
      .addImm(OutputOffset)
      .addImm(0);          // cpol
}
/// Pick whichever of a DS pseudo's two generation variants \p TII can encode.
///
/// GFX9 dropped the M0 read from the DS encodings, so LLVM's `_mc` multiclasses
/// emit two pseudos per DS instruction -- the plain one (reads M0) and a
/// `_gfx9` sibling (does not) -- and only one of them has an encoding on any
/// given subtarget. Emitting the wrong one lowers to opcode -1 and trips an
/// assert deep in the MC layer, so helper opcodes must be chosen the same way
/// the instruction under test is.
inline unsigned dsPseudoFor(const llvm::SIInstrInfo &TII, unsigned Canonical,
                            unsigned Gfx9Sibling) {
  return TII.pseudoToMCOpcode(static_cast<int>(Canonical)) >= 0 ? Canonical
                                                                : Gfx9Sibling;
}
/// LDS[addr] = DataVGPR (helper store used for DS LDS init / observation).
inline void emitDSWrite(EmitState &E, llvm::MCRegister AddrVGPR,
                        llvm::MCRegister DataVGPR) {
  BuildMI(*E.BB, E.BB->end(), E.DL,
          E.TII.get(dsPseudoFor(E.TII, llvm::AMDGPU::DS_WRITE_B32,
                                llvm::AMDGPU::DS_WRITE_B32_gfx9)))
      .addReg(AddrVGPR)
      .addReg(DataVGPR)
      .addImm(0)  // offset
      .addImm(0); // gds
}
/// DstVGPR = LDS[addr] (helper load used for DS readback).
inline void emitDSRead(EmitState &E, llvm::MCRegister DstVGPR,
                       llvm::MCRegister AddrVGPR) {
  BuildMI(*E.BB, E.BB->end(), E.DL,
          E.TII.get(dsPseudoFor(E.TII, llvm::AMDGPU::DS_READ_B32,
                                llvm::AMDGPU::DS_READ_B32_gfx9)),
          DstVGPR)
      .addReg(AddrVGPR)
      .addImm(0)  // offset
      .addImm(0); // gds
}

} // namespace luthier::test

#endif // LUTHIER_TEST_REF_KERNEL_SUPPORT_H
