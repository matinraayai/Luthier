#ifndef LUTHIER_TEST_MACHINE_KERNEL_BUILDER_H
#define LUTHIER_TEST_MACHINE_KERNEL_BUILDER_H

#include "InstrDescriptor.h"
#include "StandaloneMIBuilder.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luthier::test {

/// Describes the kernarg layout so the host can fill inputs and read outputs
/// at the correct offsets.
struct KernargLayout {
  uint32_t TotalSize = 0;

  struct Field {
    uint32_t Offset;
    uint32_t SizeBytes;
    bool IsInput;
    llvm::StringRef Name;
  };
  std::vector<Field> Fields;

  uint32_t OutputPtrOffset = 0;
  uint32_t OutputBufSize = 0;

  struct OutputField {
    uint32_t Offset;
    uint32_t SizeBytes;
    llvm::StringRef Name;
  };
  std::vector<OutputField> OutputFields;

  // Memory instruction support.
  uint32_t DataBufPtrOffset = UINT32_MAX;
  uint32_t VSharpOffset = UINT32_MAX;
  uint32_t DataBufSizeBytes = 0;
  bool DataBufNeedsPreFill = false;
  bool DataBufIsOutput = false;
  uint32_t GroupSegmentSize = 0;
  uint32_t PrivateSegmentSize = 0;
};

/// Holds the kernel MachineFunction and associated objects.
struct KernelMFContext {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Mod;
  std::unique_ptr<llvm::MachineModuleInfo> MMI;
  llvm::MachineFunction *MF = nullptr;
};

/// Builds an AMDGPU kernel MachineFunction that wraps a standalone
/// MachineInstr: loads its input registers from kernarg, copies the MI,
/// stores output registers to a global output buffer, and ends with
/// S_ENDPGM.
class MachineKernelBuilder {
public:
  explicit MachineKernelBuilder(llvm::TargetMachine &TM) : TM(TM) {}

  /// Build a kernel MachineFunction wrapping the instruction in \p MICtx.
  ///
  /// \param MICtx      The standalone MI context (instruction + register map).
  /// \param Profile    The instruction profile (for implicit defs info).
  /// \param[out] Layout  Kernarg / output buffer layout for the host.
  /// \returns Context owning the kernel MachineFunction, or an error.
  llvm::Expected<KernelMFContext>
  build(const StandaloneMIContext &MICtx, const InstrProfile &Profile,
        KernargLayout &Layout);

  /// Emit a kernel MachineFunction to an in-memory ELF.
  llvm::Expected<llvm::SmallVector<char, 0>>
  emitToELF(KernelMFContext &KCtx);

  /// \returns The kernel function name.
  static std::string getKernelName(const InstrProfile &Profile) {
    return ("test_ref_" + Profile.Name).str();
  }

private:
  llvm::TargetMachine &TM;
};

} // namespace luthier::test

#endif // LUTHIER_TEST_MACHINE_KERNEL_BUILDER_H
