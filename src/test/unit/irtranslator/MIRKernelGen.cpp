#include "MIRKernelGen.h"

#include <llvm/ADT/Twine.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>

#include <SIDefines.h>

namespace luthier::test {

// ============================================================================
// Register name helpers
// ============================================================================

std::string MIRKernelGen::regName(llvm::MCRegister Reg) const {
  return ("$" + llvm::StringRef(MRI.getName(Reg))).str();
}

// Physical register allocation ranges (disjoint from fixed setup regs):
//   SGPR0-SGPR3   : scratch resource descriptor (reserved)
//   SGPR4-SGPR5   : kernarg segment pointer (reserved)
//   SGPR6-SGPR7   : output buffer pointer loaded here
//   SGPR10-SGPR19 : scalar input operands
//   SGPR20-SGPR29 : scalar output operands (for moving results before store)
//   VGPR0-VGPR3   : utility VGPRs for address setup and stores
//   VGPR10-VGPR19 : vector input operands
//   VGPR20-VGPR29 : vector output operands

llvm::MCRegister MIRKernelGen::pickInputReg(const OperandInfo &OI,
                                            unsigned InputIdx) const {
  if (OI.IsSGPR) {
    // SGPR10 + InputIdx * (SizeBits / 32)
    unsigned BaseIdx = 10 + InputIdx * (OI.SizeBits / 32);
    // Find the SGPR register with this index.
    // For 32-bit: SGPR10, SGPR11, ...
    // For 64-bit: SGPR10_SGPR11, SGPR12_SGPR13, ...
    // We'll just emit the register name directly.
    return llvm::MCRegister(); // We'll use string-based emission instead.
  }
  return llvm::MCRegister();
}

llvm::MCRegister MIRKernelGen::pickOutputPtrReg() const {
  return llvm::MCRegister(); // We use SGPR6_SGPR7 by convention.
}

// ============================================================================
// Helper to get SGPR pair/tuple names for a given base index and width
// ============================================================================

static std::string sgprName(unsigned BaseIdx, unsigned SizeBits) {
  if (SizeBits <= 32)
    return llvm::formatv("$sgpr{0}", BaseIdx);
  if (SizeBits == 64)
    return llvm::formatv("$sgpr{0}_sgpr{1}", BaseIdx, BaseIdx + 1);
  if (SizeBits == 96)
    return llvm::formatv("$sgpr{0}_sgpr{1}_sgpr{2}", BaseIdx, BaseIdx + 1,
                         BaseIdx + 2);
  if (SizeBits == 128)
    return llvm::formatv("$sgpr{0}_sgpr{1}_sgpr{2}_sgpr{3}", BaseIdx,
                         BaseIdx + 1, BaseIdx + 2, BaseIdx + 3);
  // Fallback for wider.
  return llvm::formatv("$sgpr{0}", BaseIdx);
}

static std::string vgprName(unsigned BaseIdx, unsigned SizeBits) {
  if (SizeBits <= 32)
    return llvm::formatv("$vgpr{0}", BaseIdx);
  if (SizeBits == 64)
    return llvm::formatv("$vgpr{0}_vgpr{1}", BaseIdx, BaseIdx + 1);
  if (SizeBits == 96)
    return llvm::formatv("$vgpr{0}_vgpr{1}_vgpr{2}", BaseIdx, BaseIdx + 1,
                         BaseIdx + 2);
  if (SizeBits == 128)
    return llvm::formatv("$vgpr{0}_vgpr{1}_vgpr{2}_vgpr{3}", BaseIdx,
                         BaseIdx + 1, BaseIdx + 2, BaseIdx + 3);
  return llvm::formatv("$vgpr{0}", BaseIdx);
}

/// Return the S_LOAD opcode suffix for a given register width in bits.
static llvm::StringRef sloadSuffix(unsigned SizeBits) {
  switch (SizeBits) {
  case 32:
    return "DWORD";
  case 64:
    return "DWORDX2";
  case 96:
    return "DWORDX3";
  case 128:
    return "DWORDX4";
  case 256:
    return "DWORDX8";
  case 512:
    return "DWORDX16";
  default:
    return "DWORD";
  }
}

// ============================================================================
// MIR Generation
// ============================================================================

std::string MIRKernelGen::generate(const InstrProfile &Profile,
                                   KernargLayout &Layout) const {
  std::string MIR;
  llvm::raw_string_ostream OS(MIR);

  std::string KName = getKernelName(Profile);

  // Track register allocation for inputs.
  unsigned NextSGPR = 10;  // First available SGPR for inputs.
  unsigned NextVGPR = 10;  // First available VGPR for inputs.
  unsigned KernargOffset = 0;

  // For each input, record the assigned register name and kernarg offset.
  struct InputBinding {
    std::string Reg;        // Physical register string (e.g. "$sgpr10").
    unsigned KernargOff;    // Offset in kernarg buffer.
    unsigned SizeBits;      // Width.
    bool IsSGPR;
    bool IsImm;
    std::string Name;
  };
  std::vector<InputBinding> InputBindings;

  for (const auto &In : Profile.Inputs) {
    InputBinding B;
    B.SizeBits = In.SizeBits;
    B.IsSGPR = In.IsSGPR;
    B.IsImm = In.IsImm;
    B.Name = llvm::formatv("in{0}", InputBindings.size());

    if (In.IsImm) {
      // Immediates are loaded as 32-bit values from kernarg, then embedded.
      B.KernargOff = KernargOffset;
      B.Reg = ""; // Will be an immediate, not a register.
      Layout.Fields.push_back(
          {KernargOffset, 4, /*IsInput=*/true, B.Name});
      KernargOffset += 4;
    } else if (In.IsSGPR || !In.IsVGPR) {
      // Scalar register — load via S_LOAD_DWORD*.
      unsigned NumDwords = (In.SizeBits + 31) / 32;
      // Align offset to size.
      unsigned AlignBytes = NumDwords * 4;
      KernargOffset = (KernargOffset + AlignBytes - 1) & ~(AlignBytes - 1);
      B.KernargOff = KernargOffset;
      B.Reg = sgprName(NextSGPR, In.SizeBits);
      NextSGPR += NumDwords;
      Layout.Fields.push_back(
          {KernargOffset, NumDwords * 4u, /*IsInput=*/true, B.Name});
      KernargOffset += NumDwords * 4;
    } else {
      // Vector register — load via S_LOAD then V_MOV.
      unsigned NumDwords = (In.SizeBits + 31) / 32;
      unsigned AlignBytes = NumDwords * 4;
      KernargOffset = (KernargOffset + AlignBytes - 1) & ~(AlignBytes - 1);
      B.KernargOff = KernargOffset;
      B.Reg = vgprName(NextVGPR, In.SizeBits);
      NextVGPR += NumDwords;
      Layout.Fields.push_back(
          {KernargOffset, NumDwords * 4u, /*IsInput=*/true, B.Name});
      KernargOffset += NumDwords * 4;
    }
    InputBindings.push_back(B);
  }

  // Reserve kernarg space for the output buffer pointer (64-bit, 8-byte aligned).
  KernargOffset = (KernargOffset + 7) & ~7u;
  Layout.OutputPtrOffset = KernargOffset;
  Layout.Fields.push_back(
      {KernargOffset, 8, /*IsInput=*/false, "out_ptr"});
  KernargOffset += 8;
  Layout.TotalSize = KernargOffset;

  // Compute output buffer layout.
  unsigned OutputOffset = 0;
  struct OutputBinding {
    std::string Reg;
    unsigned OutBufOff;
    unsigned SizeBits;
    bool IsSGPR;
    std::string Name;
  };
  std::vector<OutputBinding> OutputBindings;

  for (const auto &Out : Profile.Outputs) {
    OutputBinding B;
    B.SizeBits = Out.SizeBits;
    B.IsSGPR = Out.IsSGPR;
    B.Name = llvm::formatv("out{0}", OutputBindings.size());
    // Outputs use the instruction's own destination registers.
    // We assign them from SGPR20+ / VGPR20+.
    unsigned NumDwords = (Out.SizeBits + 31) / 32;
    if (Out.IsSGPR || !Out.IsVGPR)
      B.Reg = sgprName(20 + OutputBindings.size() * NumDwords, Out.SizeBits);
    else
      B.Reg = vgprName(20 + OutputBindings.size() * NumDwords, Out.SizeBits);

    unsigned AlignBytes = NumDwords * 4;
    OutputOffset = (OutputOffset + AlignBytes - 1) & ~(AlignBytes - 1);
    B.OutBufOff = OutputOffset;
    Layout.OutputFields.push_back({OutputOffset, NumDwords * 4u, B.Name});
    OutputOffset += NumDwords * 4;
    OutputBindings.push_back(B);
  }

  // Implicit defs (SCC, VCC) also go to the output buffer.
  for (llvm::MCRegister Reg : Profile.ImplicitDefs) {
    llvm::StringRef RName = MRI.getName(Reg);
    // SCC → 4 bytes, VCC → 8 bytes, EXEC → 8 bytes.
    unsigned Size = 4;
    if (RName == "VCC" || RName == "VCC_LO" || RName == "EXEC" ||
        RName == "EXEC_LO")
      Size = 8;

    Layout.OutputFields.push_back({OutputOffset, Size, RName});
    OutputOffset += Size;
  }
  Layout.OutputBufSize = OutputOffset;

  // =========================================================================
  // Emit MIR YAML
  // =========================================================================

  // --- Module IR stub ---
  OS << "--- |\n";
  OS << "  define amdgpu_kernel void @" << KName
     << "(ptr addrspace(4) %kernarg) #0 {\n";
  OS << "    ret void\n";
  OS << "  }\n";
  OS << "  attributes #0 = { \"amdgpu-flat-work-group-size\"=\"1,1\" }\n";
  OS << "...\n";

  // --- Machine function ---
  OS << "---\n";
  OS << "name: " << KName << "\n";
  OS << "tracksRegLiveness: true\n";
  OS << "machineFunctionInfo:\n";
  OS << "  isEntryFunction: true\n";
  OS << "  scratchRSrcReg: '$sgpr0_sgpr1_sgpr2_sgpr3'\n";
  OS << "  kernargSegmentPtr: '$sgpr4_sgpr5'\n";
  OS << "body: |\n";
  OS << "  bb.0:\n";
  OS << "    liveins: $sgpr4_sgpr5\n";

  // --- Load inputs from kernarg ---
  for (const auto &B : InputBindings) {
    if (B.IsImm)
      continue; // Immediates don't need register loads.

    unsigned NumDwords = (B.SizeBits + 31) / 32;
    std::string Suffix = sloadSuffix(B.SizeBits).str();

    if (B.IsSGPR) {
      // Direct scalar load from kernarg.
      OS << "    " << B.Reg << " = S_LOAD_" << Suffix
         << "_IMM $sgpr4_sgpr5, " << B.KernargOff << ", 0"
         << " :: (load (s" << B.SizeBits << "), addrspace 4)\n";
    } else {
      // Load into a temporary SGPR then V_MOV to VGPR.
      // Use SGPR30+ as temp for VGPR loads.
      std::string TmpSgpr = sgprName(30, B.SizeBits);
      OS << "    " << TmpSgpr << " = S_LOAD_" << Suffix
         << "_IMM $sgpr4_sgpr5, " << B.KernargOff << ", 0"
         << " :: (load (s" << B.SizeBits << "), addrspace 4)\n";
      // V_MOV per dword component.
      for (unsigned D = 0; D < NumDwords; ++D) {
        OS << "    " << vgprName(10 + D, 32) << " = V_MOV_B32_e32 "
           << sgprName(30 + D, 32) << ", implicit $exec\n";
      }
    }
  }

  // Load output buffer pointer into SGPR6_SGPR7.
  OS << "    $sgpr6_sgpr7 = S_LOAD_DWORDX2_IMM $sgpr4_sgpr5, "
     << Layout.OutputPtrOffset << ", 0"
     << " :: (load (s64), addrspace 4)\n";

  // Wait for all scalar loads to complete.
  OS << "    S_WAITCNT 0\n";

  // --- The instruction under test ---
  OS << "    ; === INSTRUCTION UNDER TEST ===\n";
  // Build the instruction string with assigned registers.
  OS << "    ";

  // Output operands first (defs).
  for (unsigned I = 0; I < OutputBindings.size(); ++I) {
    if (I > 0)
      OS << ", ";
    OS << OutputBindings[I].Reg;
  }

  // Implicit defs.
  bool HasImplicitSCC = false;
  bool HasImplicitVCC = false;
  for (llvm::MCRegister Reg : Profile.ImplicitDefs) {
    llvm::StringRef RName = MRI.getName(Reg);
    if (RName == "SCC")
      HasImplicitSCC = true;
    if (RName.starts_with("VCC"))
      HasImplicitVCC = true;
  }

  if (!OutputBindings.empty())
    OS << " = ";

  OS << Profile.Name;

  // Input operands.
  for (unsigned I = 0; I < InputBindings.size(); ++I) {
    OS << (I == 0 ? " " : ", ");
    if (InputBindings[I].IsImm) {
      OS << "0"; // Placeholder immediate — the actual value is loaded
                 // from kernarg and patched by a more sophisticated generator.
    } else {
      OS << InputBindings[I].Reg;
    }
  }

  // Implicit defs/uses in MIR syntax.
  if (HasImplicitSCC)
    OS << ", implicit-def $scc";
  if (HasImplicitVCC)
    OS << ", implicit-def $vcc";
  for (llvm::MCRegister Reg : Profile.ImplicitUses) {
    llvm::StringRef RName = MRI.getName(Reg);
    OS << ", implicit $" << RName.lower();
  }
  OS << "\n";

  // --- Store output register values to the output buffer ---
  // Move output buffer pointer to VGPRs for FLAT_STORE.
  OS << "    $vgpr0 = V_MOV_B32_e32 $sgpr6, implicit $exec\n";
  OS << "    $vgpr1 = V_MOV_B32_e32 $sgpr7, implicit $exec\n";

  unsigned StoreVGPR = 2; // Next available utility VGPR for store data.
  for (const auto &OB : OutputBindings) {
    unsigned NumDwords = (OB.SizeBits + 31) / 32;
    for (unsigned D = 0; D < NumDwords; ++D) {
      std::string SrcReg;
      if (OB.IsSGPR) {
        // Move SGPR to VGPR for flat store.
        unsigned SgprBase = 20 + (&OB - OutputBindings.data()) *
                                      ((OB.SizeBits + 31) / 32);
        SrcReg = sgprName(SgprBase + D, 32);
        OS << "    $vgpr" << StoreVGPR << " = V_MOV_B32_e32 " << SrcReg
           << ", implicit $exec\n";
      } else {
        // VGPR output can be stored directly.
        unsigned VgprBase = 20 + (&OB - OutputBindings.data()) *
                                      ((OB.SizeBits + 31) / 32);
        SrcReg = vgprName(VgprBase + D, 32);
        OS << "    $vgpr" << StoreVGPR << " = COPY " << SrcReg << "\n";
      }
      OS << "    FLAT_STORE_DWORD $vgpr0_vgpr1, $vgpr" << StoreVGPR << ", "
         << (OB.OutBufOff + D * 4) << ", 0"
         << " :: (store (s32), addrspace 1)\n";
      ++StoreVGPR;
    }
  }

  // Store implicit def values.
  unsigned ImplicitOutOff = 0;
  for (const auto &OF : Layout.OutputFields) {
    // Skip explicit output fields (already stored above).
    if (ImplicitOutOff < OutputBindings.size()) {
      ++ImplicitOutOff;
      continue;
    }
    llvm::StringRef RName = OF.Name;
    if (RName == "SCC") {
      // Convert SCC to i32 via S_CSELECT_B32.
      OS << "    $sgpr40 = S_CSELECT_B32 1, 0, implicit $scc\n";
      OS << "    $vgpr" << StoreVGPR
         << " = V_MOV_B32_e32 $sgpr40, implicit $exec\n";
      OS << "    FLAT_STORE_DWORD $vgpr0_vgpr1, $vgpr" << StoreVGPR << ", "
         << OF.Offset << ", 0 :: (store (s32), addrspace 1)\n";
      ++StoreVGPR;
    } else if (RName.starts_with("VCC")) {
      // Store VCC as 64-bit.
      OS << "    $sgpr40 = COPY $vcc_lo\n";
      OS << "    $sgpr41 = COPY $vcc_hi\n";
      OS << "    $vgpr" << StoreVGPR
         << " = V_MOV_B32_e32 $sgpr40, implicit $exec\n";
      OS << "    FLAT_STORE_DWORD $vgpr0_vgpr1, $vgpr" << StoreVGPR << ", "
         << OF.Offset << ", 0 :: (store (s32), addrspace 1)\n";
      ++StoreVGPR;
      OS << "    $vgpr" << StoreVGPR
         << " = V_MOV_B32_e32 $sgpr41, implicit $exec\n";
      OS << "    FLAT_STORE_DWORD $vgpr0_vgpr1, $vgpr" << StoreVGPR << ", "
         << (OF.Offset + 4) << ", 0 :: (store (s32), addrspace 1)\n";
      ++StoreVGPR;
    }
  }

  OS << "    S_WAITCNT 0\n";
  OS << "    S_ENDPGM 0\n";
  OS << "...\n";

  return MIR;
}

} // namespace luthier::test
