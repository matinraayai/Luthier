#ifndef LUTHIER_TEST_STANDALONE_MI_BUILDER_H
#define LUTHIER_TEST_STANDALONE_MI_BUILDER_H

#include "InstrDescriptor.h"

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>

namespace luthier::test {

/// Owns the LLVM objects that keep a standalone MachineFunction alive.
struct StandaloneMIContext {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Mod;
  std::unique_ptr<llvm::MachineModuleInfo> MMI;
  llvm::MachineFunction *MF = nullptr;

  /// The single MachineInstr under test.  Owned by the MachineBasicBlock
  /// inside \c MF.
  llvm::MachineInstr *MI = nullptr;

  /// Physical registers assigned to each explicit input operand (in order).
  /// For immediate operands, the entry is MCRegister() (invalid).
  llvm::SmallVector<llvm::MCRegister, 8> InputRegs;

  /// Physical registers assigned to each explicit output (def) operand.
  llvm::SmallVector<llvm::MCRegister, 4> OutputRegs;
};

/// Builds a standalone MachineFunction containing a single basic block with
/// a single target instruction.  The instruction's operands are bound to
/// concrete physical registers using proper MCRegisterInfo lookups.
///
/// The resulting MachineInstr can then be passed to both:
///   - MachineKernelBuilder (copies it into a kernel wrapper)
///   - IRKernelGen / translateSingleInstr (translates it to IR)
class StandaloneMIBuilder {
public:
  explicit StandaloneMIBuilder(llvm::TargetMachine &TM) : TM(TM) {}

  /// Build a standalone MachineFunction for the given instruction profile.
  ///
  /// The MachineInstr is created with physical registers selected from
  /// non-conflicting ranges (SGPR10+ for scalar inputs, VGPR10+ for
  /// vector inputs, SGPR20+ for scalar outputs, VGPR20+ for vector outputs).
  ///
  /// \param Profile  The analyzed instruction profile.
  /// \returns Context owning the MachineFunction and the target MI.
  llvm::Expected<StandaloneMIContext> build(const InstrProfile &Profile);

private:
  llvm::TargetMachine &TM;
};

} // namespace luthier::test

#endif // LUTHIER_TEST_STANDALONE_MI_BUILDER_H
