#ifndef LUTHIER_TEST_MIR_KERNEL_GEN_H
#define LUTHIER_TEST_MIR_KERNEL_GEN_H

#include "InstrDescriptor.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/MC/MCRegisterInfo.h>

#include <cstdint>
#include <string>
#include <vector>

namespace luthier::test {

/// Describes the kernarg layout so the host can fill inputs and read outputs
/// at the correct offsets.
struct KernargLayout {
  /// Total size of the kernarg buffer in bytes.
  uint32_t TotalSize = 0;

  struct Field {
    uint32_t Offset;    ///< Byte offset within the kernarg buffer.
    uint32_t SizeBytes; ///< Size in bytes.
    bool IsInput;       ///< true = input value, false = output buffer pointer.
    llvm::StringRef Name; ///< Human-readable label (e.g. "src0", "out_ptr").
  };
  std::vector<Field> Fields;

  /// Byte offset of the output buffer pointer inside the kernarg.
  uint32_t OutputPtrOffset = 0;
  /// Total size of the output buffer in bytes.
  uint32_t OutputBufSize = 0;

  /// Per-output entry describing where in the output buffer each result lives.
  struct OutputField {
    uint32_t Offset;    ///< Byte offset within the output buffer.
    uint32_t SizeBytes; ///< Size in bytes.
    llvm::StringRef Name; ///< e.g. "sdst", "SCC".
  };
  std::vector<OutputField> OutputFields;

  //===-- Memory instruction support ------------------------------------===//

  /// Byte offset of the data buffer pointer in kernarg (global/flat/relay).
  /// UINT32_MAX if not applicable.
  uint32_t DataBufPtrOffset = UINT32_MAX;

  /// Byte offset of the 128-bit V# descriptor in kernarg (16-byte aligned).
  /// UINT32_MAX if not applicable.
  uint32_t VSharpOffset = UINT32_MAX;

  /// Size of the data buffer the host must allocate.
  uint32_t DataBufSizeBytes = 0;

  /// Whether the host pre-fills the data buffer with a deterministic pattern.
  bool DataBufNeedsPreFill = false;

  /// Whether the data buffer contents are compared after dispatch
  /// (for stores and atomics).
  bool DataBufIsOutput = false;

  /// Required group_segment_size for the AQL dispatch packet.
  uint32_t GroupSegmentSize = 0;

  /// Required private_segment_size for the AQL dispatch packet.
  uint32_t PrivateSegmentSize = 0;
};

/// Generates a minimal MIR kernel that wraps a single instruction under test.
///
/// The generated kernel:
///   1. Loads input operand values from the kernarg segment into physical regs.
///   2. Executes the single target instruction.
///   3. Stores output register values to a global output buffer.
///   4. Ends with S_ENDPGM.
class MIRKernelGen {
public:
  /// \param MRI  The MCRegisterInfo for the target (used for register names).
  explicit MIRKernelGen(const llvm::MCRegisterInfo &MRI) : MRI(MRI) {}

  /// Generate the MIR text and kernarg layout for the given instruction.
  ///
  /// \param Profile  The analyzed instruction profile.
  /// \param[out] Layout  Filled with the kernarg and output buffer layout.
  /// \returns The complete MIR YAML text that can be compiled to an ELF.
  std::string generate(const InstrProfile &Profile,
                       KernargLayout &Layout) const;

  /// \returns The kernel function name used in the generated MIR.
  static std::string getKernelName(const InstrProfile &Profile) {
    return ("test_ref_" + Profile.Name).str();
  }

private:
  const llvm::MCRegisterInfo &MRI;

  /// Emit a register name suitable for MIR (e.g. "$sgpr10").
  std::string regName(llvm::MCRegister Reg) const;

  /// Pick a physical register for an input operand that doesn't conflict
  /// with the instruction's own operands or the kernarg/scratch setup regs.
  llvm::MCRegister pickInputReg(const OperandInfo &OI,
                                unsigned InputIdx) const;

  /// Pick a physical register for loading the output pointer.
  llvm::MCRegister pickOutputPtrReg() const;
};

} // namespace luthier::test

#endif // LUTHIER_TEST_MIR_KERNEL_GEN_H
