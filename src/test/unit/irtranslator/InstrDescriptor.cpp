#include "InstrDescriptor.h"

#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/Target/TargetMachine.h>

// AMDGPU-specific headers.
#include <SIDefines.h>

namespace luthier::test {

InstrDescriptor::InstrDescriptor(const llvm::TargetMachine &TM) {
  MCII = TM.getMCInstrInfo();
  MRI = TM.getMCRegisterInfo();
}

unsigned InstrDescriptor::getNumOpcodes() const {
  return MCII->getNumOpcodes();
}

llvm::StringRef InstrDescriptor::getName(unsigned Opcode) const {
  return MCII->getName(Opcode);
}

InstrProfile InstrDescriptor::analyze(unsigned Opcode) const {
  InstrProfile P;
  P.Opcode = Opcode;
  P.Name = MCII->getName(Opcode);

  const llvm::MCInstrDesc &Desc = MCII->get(Opcode);

  // --- Check if this instruction is testable ---
  if (Desc.isBranch() || Desc.isCall() || Desc.isReturn() ||
      Desc.isBarrier()) {
    P.IsTestable = false;
    P.SkipReason = "control flow";
    return P;
  }
  if (P.Name.starts_with("S_NOP") || P.Name.starts_with("S_WAITCNT") ||
      P.Name.starts_with("S_BARRIER") || P.Name.starts_with("S_ENDPGM") ||
      P.Name.starts_with("S_TRAP") || P.Name.starts_with("S_SLEEP") ||
      P.Name.starts_with("S_SETHALT") || P.Name.starts_with("S_SETPRIO") ||
      P.Name.starts_with("S_SENDMSG") || P.Name.starts_with("S_ICACHE") ||
      P.Name.starts_with("S_DCACHE") || P.Name.starts_with("BUFFER_GL") ||
      P.Name.starts_with("BUFFER_INV") || P.Name.starts_with("BUFFER_WB") ||
      P.Name.starts_with("DS_GWS") || P.Name.starts_with("DS_ORDERED")) {
    P.IsTestable = false;
    P.SkipReason = "side-effect only / not data-testable";
    return P;
  }

  // --- Analyze explicit operands ---
  unsigned NumDefs = Desc.getNumDefs();
  unsigned NumOps = Desc.getNumOperands();

  for (unsigned I = 0; I < NumOps; ++I) {
    const llvm::MCOperandInfo &OpInfo = Desc.operands()[I];
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
      const llvm::MCRegisterClass &RC = MRI->getRegClass(OpInfo.RegClass);
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
  if (const llvm::MCPhysReg *ImpDefs = Desc.implicit_defs()) {
    for (; *ImpDefs; ++ImpDefs)
      P.ImplicitDefs.push_back(llvm::MCRegister(*ImpDefs));
  }
  if (const llvm::MCPhysReg *ImpUses = Desc.implicit_uses()) {
    for (; *ImpUses; ++ImpUses)
      P.ImplicitUses.push_back(llvm::MCRegister(*ImpUses));
  }

  // --- Memory access detection ---
  P.Mem.MemKind = MemAccessInfo::None;
  if (Desc.mayLoad() || Desc.mayStore()) {
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

  return P;
}

//===----------------------------------------------------------------------===//
// Memory test classification
//===----------------------------------------------------------------------===//

/// Default data buffer size — large enough for varied offsets.
static constexpr uint32_t DefaultDataBufSize = 4096;
/// Default LDS allocation.
static constexpr uint32_t DefaultLDSSize = 1024;
/// Default scratch allocation per work-item.
static constexpr uint32_t DefaultScratchSize = 1024;

static bool nameContains(llvm::StringRef N, llvm::StringRef Sub) {
  return N.contains(Sub);
}

MemoryTestSetup classifyMemoryTest(const InstrProfile &Profile) {
  MemoryTestSetup S;
  if (Profile.Mem.MemKind == MemAccessInfo::None) {
    S.Kind = MemoryTestSetup::NoMemory;
    return S;
  }

  S.DataElemSizeBytes = std::max(1u, Profile.Mem.DataSizeBits / 8);
  if (S.DataElemSizeBytes == 0)
    S.DataElemSizeBytes = 4; // default dword

  bool IsLoad = nameContains(Profile.Name, "LOAD") ||
                nameContains(Profile.Name, "READ");
  bool IsStore = nameContains(Profile.Name, "STORE") ||
                 nameContains(Profile.Name, "WRITE");
  bool IsAtomic = nameContains(Profile.Name, "ATOMIC") ||
                  nameContains(Profile.Name, "CMPSWAP") ||
                  nameContains(Profile.Name, "CMPST");
  bool IsRTN = nameContains(Profile.Name, "_RTN");
  bool HasDef = !Profile.Outputs.empty();

  switch (Profile.Mem.MemKind) {
  case MemAccessInfo::Global:
  case MemAccessInfo::Flat:
    S.DataBufSizeBytes = DefaultDataBufSize;
    if (IsAtomic) {
      S.Kind = MemoryTestSetup::GlobalAtomic;
      S.NeedsPreFill = true;
      S.DataBufIsOutput = true;
      S.HasRegisterResult = IsRTN || HasDef;
    } else if (IsStore) {
      S.Kind = MemoryTestSetup::GlobalStore;
      S.DataBufIsOutput = true;
      S.HasRegisterResult = false;
    } else {
      S.Kind = MemoryTestSetup::GlobalLoad;
      S.NeedsPreFill = true;
      S.HasRegisterResult = true;
    }
    break;

  case MemAccessInfo::Buffer:
    S.DataBufSizeBytes = DefaultDataBufSize;
    // Distinguish SMEM (scalar) vs MUBUF (vector) buffer.
    if (Profile.Name.starts_with("S_BUFFER_LOAD")) {
      S.Kind = MemoryTestSetup::ScalarBufferLoad;
      S.NeedsVSharp = true;
      S.NeedsPreFill = true;
      S.HasRegisterResult = true;
    } else if (Profile.Name.starts_with("S_LOAD")) {
      S.Kind = MemoryTestSetup::ScalarLoad;
      S.NeedsPreFill = true;
      S.HasRegisterResult = true;
    } else if (IsAtomic) {
      S.Kind = MemoryTestSetup::BufferAtomic;
      S.NeedsVSharp = true;
      S.NeedsPreFill = true;
      S.DataBufIsOutput = true;
      S.HasRegisterResult = IsRTN || HasDef;
    } else if (IsStore) {
      S.Kind = MemoryTestSetup::BufferStore;
      S.NeedsVSharp = true;
      S.DataBufIsOutput = true;
    } else {
      S.Kind = MemoryTestSetup::BufferLoad;
      S.NeedsVSharp = true;
      S.NeedsPreFill = true;
      S.HasRegisterResult = true;
    }
    break;

  case MemAccessInfo::LDS:
    S.GroupSegmentSize = DefaultLDSSize;
    S.DataBufSizeBytes = DefaultLDSSize; // relay buffer
    if (IsAtomic) {
      S.Kind = MemoryTestSetup::LDSAtomic;
      S.NeedsPreFill = true;
      S.DataBufIsOutput = true;
      S.HasRegisterResult = IsRTN || HasDef;
    } else if (IsStore) {
      S.Kind = MemoryTestSetup::LDSStore;
      S.DataBufIsOutput = true;
    } else {
      S.Kind = MemoryTestSetup::LDSLoad;
      S.NeedsPreFill = true;
      S.HasRegisterResult = true;
    }
    break;

  case MemAccessInfo::Scratch:
    S.PrivateSegmentSize = DefaultScratchSize;
    S.DataBufSizeBytes = DefaultScratchSize; // relay buffer
    if (IsStore) {
      S.Kind = MemoryTestSetup::ScratchStore;
      S.DataBufIsOutput = true;
    } else {
      S.Kind = MemoryTestSetup::ScratchLoad;
      S.NeedsPreFill = true;
      S.HasRegisterResult = true;
    }
    break;

  default:
    S.Kind = MemoryTestSetup::NoMemory;
    break;
  }

  return S;
}

//===----------------------------------------------------------------------===//
// Address randomization
//===----------------------------------------------------------------------===//

AddressConfig randomizeAddress(const MemoryTestSetup &Setup,
                               const InstrProfile &Profile, uint64_t Seed) {
  std::mt19937_64 RNG(Seed ^ 0xADD7E55);
  AddressConfig Cfg;

  uint32_t Elem = Setup.DataElemSizeBytes;
  uint32_t Align = (Elem >= 4) ? Elem : std::max(Elem, 1u);

  // Pick a random aligned offset in [0, Max - Elem], clamped to alignment.
  auto randAligned = [&](uint32_t BufSize, uint32_t A) -> uint32_t {
    if (BufSize <= Elem)
      return 0;
    uint32_t MaxOff = BufSize - Elem;
    MaxOff = MaxOff - (MaxOff % A); // align down
    if (MaxOff == 0)
      return 0;
    return static_cast<uint32_t>((RNG() % (MaxOff / A + 1)) * A);
  };

  switch (Setup.Kind) {
  case MemoryTestSetup::GlobalLoad:
  case MemoryTestSetup::GlobalStore:
  case MemoryTestSetup::GlobalAtomic:
    Cfg.BaseOffset = randAligned(Setup.DataBufSizeBytes, Align);
    break;

  case MemoryTestSetup::BufferLoad:
  case MemoryTestSetup::BufferStore:
  case MemoryTestSetup::BufferAtomic: {
    // Split offset budget between voffset (BaseOffset) and soffset.
    uint32_t TotalBudget = Setup.DataBufSizeBytes - Elem;
    Cfg.SOffset = randAligned(TotalBudget / 2, 4);
    uint32_t Remaining = TotalBudget - Cfg.SOffset;
    Cfg.BaseOffset = randAligned(Remaining + 1, Align);
    break;
  }

  case MemoryTestSetup::ScalarBufferLoad:
    Cfg.BaseOffset = randAligned(Setup.DataBufSizeBytes, 4);
    break;

  case MemoryTestSetup::ScalarLoad:
    Cfg.BaseOffset = randAligned(Setup.DataBufSizeBytes, 4);
    break;

  case MemoryTestSetup::LDSLoad:
  case MemoryTestSetup::LDSStore:
  case MemoryTestSetup::LDSAtomic:
    Cfg.LDSOffset0 = randAligned(Setup.GroupSegmentSize, Align);
    Cfg.LDSOffset1 = randAligned(Setup.GroupSegmentSize, Align);
    break;

  case MemoryTestSetup::ScratchLoad:
  case MemoryTestSetup::ScratchStore:
    Cfg.ScratchOffset = randAligned(Setup.PrivateSegmentSize, Align);
    break;

  default:
    break;
  }

  return Cfg;
}

//===----------------------------------------------------------------------===//
// Deterministic data fill
//===----------------------------------------------------------------------===//

void fillDeterministicPattern(void *Buf, size_t Size, uint64_t Seed) {
  auto *Bytes = static_cast<uint8_t *>(Buf);
  // Use a fast hash: each 8-byte block is hash(seed, block_index).
  std::mt19937_64 RNG(Seed ^ 0xDA7ADA7A);
  size_t I = 0;
  for (; I + 8 <= Size; I += 8) {
    uint64_t V = RNG();
    std::memcpy(Bytes + I, &V, 8);
  }
  if (I < Size) {
    uint64_t V = RNG();
    std::memcpy(Bytes + I, &V, Size - I);
  }
}

} // namespace luthier::test
