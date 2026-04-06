#ifndef LUTHIER_TEST_INSTR_DESCRIPTOR_H
#define LUTHIER_TEST_INSTR_DESCRIPTOR_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/MC/MCRegister.h>

#include <cstdint>
#include <random>

namespace llvm {
class TargetMachine;
class MCInstrInfo;
class MCRegisterInfo;
} // namespace llvm

namespace luthier::test {

/// Describes a single explicit operand of an instruction.
struct OperandInfo {
  unsigned Idx;         ///< Index in the MachineInstr operand list.
  bool IsDef;           ///< true = output, false = input.
  bool IsReg;           ///< true = register operand.
  bool IsImm;           ///< true = immediate operand.
  bool IsSGPR;          ///< true if register is scalar.
  bool IsVGPR;          ///< true if register is vector.
  unsigned SizeBits;    ///< Register size in bits (32, 64, 128, ...).
  unsigned RegClassID;  ///< AMDGPU register class ID.
};

/// Describes what kind of memory an instruction accesses.
struct MemAccessInfo {
  enum Kind { None, Global, LDS, Scratch, Buffer, Flat };
  Kind MemKind = None;
  unsigned DataSizeBits = 0;
};

/// Complete operand profile for a single instruction opcode.
struct InstrProfile {
  unsigned Opcode;
  llvm::StringRef Name; ///< Instruction mnemonic name.

  llvm::SmallVector<OperandInfo, 4> Inputs;
  llvm::SmallVector<OperandInfo, 2> Outputs;

  llvm::SmallVector<llvm::MCRegister, 4> ImplicitDefs;
  llvm::SmallVector<llvm::MCRegister, 4> ImplicitUses;

  MemAccessInfo Mem;

  /// Whether this instruction can be meaningfully fuzzed.
  bool IsTestable = true;
  /// Reason for skipping (if !IsTestable).
  llvm::StringRef SkipReason;
};

/// Describes what memory resources an instruction under test needs and how
/// the fuzzer should set them up.
struct MemoryTestSetup {
  enum TestKind {
    NoMemory,          ///< ALU-only, no memory access.
    GlobalLoad,        ///< Pre-filled global buffer → register.
    GlobalStore,       ///< Register → global buffer.
    GlobalAtomic,      ///< Pre-filled buffer → register + buffer.
    BufferLoad,        ///< V# + offset → register (MUBUF).
    BufferStore,       ///< V# + data → buffer (MUBUF).
    BufferAtomic,      ///< V# + data → register + buffer (MUBUF).
    ScalarBufferLoad,  ///< S_BUFFER_LOAD: V# + offset → SGPR.
    ScalarLoad,        ///< S_LOAD: base64 + offset → SGPR.
    LDSLoad,           ///< Kernel-mediated LDS load.
    LDSStore,          ///< Kernel-mediated LDS store.
    LDSAtomic,         ///< Kernel-mediated LDS atomic.
    ScratchLoad,       ///< Kernel-mediated scratch load.
    ScratchStore,      ///< Kernel-mediated scratch store.
  };

  TestKind Kind = NoMemory;
  uint32_t DataBufSizeBytes = 0;    ///< Host-allocated data buffer size.
  uint32_t DataElemSizeBytes = 4;   ///< Element size accessed by the instr.
  bool NeedsPreFill = false;        ///< Host pre-fills the data buffer.
  bool DataBufIsOutput = false;     ///< Host compares data buffer post-dispatch.
  bool HasRegisterResult = false;   ///< Instruction writes a register output.
  bool NeedsVSharp = false;         ///< Needs a 128-bit V# descriptor in kernarg.
  uint32_t GroupSegmentSize = 0;    ///< LDS size for AQL packet.
  uint32_t PrivateSegmentSize = 0;  ///< Scratch size for AQL packet.
};

/// Randomized addresses within the target address space.
struct AddressConfig {
  uint32_t BaseOffset = 0;    ///< Byte offset into global/buffer data buf.
  uint32_t VIndex = 0;        ///< Buffer struct-mode element index.
  uint32_t SOffset = 0;       ///< Scalar offset (MUBUF/SMEM).
  uint32_t LDSOffset0 = 0;   ///< Primary LDS offset (DS instructions).
  uint32_t LDSOffset1 = 0;   ///< Secondary LDS offset (READ2/WRITE2).
  uint32_t ScratchOffset = 0; ///< Scratch byte offset.
};

/// 128-bit buffer resource descriptor (V#) for MUBUF/SMEM.
struct VSharpDescriptor {
  uint32_t Words[4] = {};

  /// Construct a raw buffer descriptor pointing at \p BaseAddr with
  /// \p NumBytes accessible.
  static VSharpDescriptor createRaw(uint64_t BaseAddr, uint32_t NumBytes) {
    VSharpDescriptor V;
    V.Words[0] = static_cast<uint32_t>(BaseAddr);
    V.Words[1] = static_cast<uint32_t>(BaseAddr >> 32) & 0xFFFF;
    V.Words[2] = NumBytes;
    V.Words[3] = 0x00027FAC; // GFX9 raw buffer, 32-bit float format.
    return V;
  }
};

/// Classify the memory test requirements for an instruction.
MemoryTestSetup classifyMemoryTest(const InstrProfile &Profile);

/// Pick randomized, in-bounds addresses for the instruction under test.
AddressConfig randomizeAddress(const MemoryTestSetup &Setup,
                               const InstrProfile &Profile, uint64_t Seed);

/// Fill \p Buf with a deterministic pattern derived from \p Seed so that
/// any address miscalculation produces a detectably wrong value.
void fillDeterministicPattern(void *Buf, size_t Size, uint64_t Seed);

/// Analyzes AMDGPU instructions using MCInstrDesc and SIInstrInfo.
class InstrDescriptor {
public:
  /// Initialize from a TargetMachine (must be an AMDGPU target).
  explicit InstrDescriptor(const llvm::TargetMachine &TM);

  /// Analyze the instruction with the given \p Opcode.
  InstrProfile analyze(unsigned Opcode) const;

  /// \returns The total number of opcodes known to the target.
  unsigned getNumOpcodes() const;

  /// \returns The instruction name for the given opcode.
  llvm::StringRef getName(unsigned Opcode) const;

private:
  const llvm::MCInstrInfo *MCII;
  const llvm::MCRegisterInfo *MRI;
};

} // namespace luthier::test

#endif // LUTHIER_TEST_INSTR_DESCRIPTOR_H
