#ifndef LUTHIER_TEST_IR_KERNEL_GEN_H
#define LUTHIER_TEST_IR_KERNEL_GEN_H

#include "InstrDescriptor.h"
#include "MachineKernelBuilder.h"
#include "StandaloneMIBuilder.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>

namespace luthier::test {

/// Generates an LLVM IR kernel that reproduces the behavior of a single
/// AMDGPU instruction using the TableGen-generated semantic translation.
///
/// Takes a standalone MachineInstr (from StandaloneMIBuilder) and the same
/// KernargLayout used by MachineKernelBuilder, so both kernels share the
/// identical memory layout.
class IRKernelGen {
public:
  explicit IRKernelGen(const llvm::TargetMachine &TM) : TM(TM) {}

  /// Generate an LLVM IR Module containing the translated kernel.
  ///
  /// \param MICtx    The standalone MI context (MI + register assignments).
  /// \param Profile  The instruction profile (for implicit defs).
  /// \param Layout   The kernarg/output layout (from MachineKernelBuilder).
  /// \param Ctx      The LLVMContext for the new Module.
  llvm::Expected<std::unique_ptr<llvm::Module>>
  generate(const StandaloneMIContext &MICtx, const InstrProfile &Profile,
           const KernargLayout &Layout, llvm::LLVMContext &Ctx) const;

  static std::string getKernelName(const InstrProfile &Profile) {
    return ("test_trans_" + Profile.Name).str();
  }

private:
  const llvm::TargetMachine &TM;
};

} // namespace luthier::test

#endif // LUTHIER_TEST_IR_KERNEL_GEN_H
