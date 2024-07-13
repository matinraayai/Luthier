//===-- code_generator.hpp - Luthier Code Generator  ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's code generator, which instruments lifted
/// representations given an instrumentation task.
//===----------------------------------------------------------------------===//
#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "common/object_utils.hpp"
#include "common/singleton.hpp"
#include "hsa/hsa_executable.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include <luthier/instrumentation_task.h>
#include <luthier/lifted_representation.h>
#include <luthier/types.h>

namespace llvm {
class LivePhysRegs;
}

namespace luthier {

namespace hsa {

class ISA;

} // namespace hsa

class CodeGenerator : public Singleton<CodeGenerator> {
public:
  llvm::Error
  instrument(const LiftedRepresentation &LR,
             llvm::function_ref<llvm::Error(InstrumentationTask &,
                                            LiftedRepresentation &)>
                 Mutator,
             llvm::DenseMap<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>
                 &CompiledCodeObjects,
             llvm::DenseMap<hsa::LoadedCodeObject, llvm::SmallVector<char>>
                 *AssemblyFiles = nullptr);

private:
  static llvm::Expected<llvm::Function &> generateHookIR(
      const llvm::MachineInstr &MI,
      llvm::ArrayRef<InstrumentationTask::hook_invocation_descriptor> HookSpecs,
      const hsa::ISA &ISA, llvm::Module &IModule);

  static llvm::Error insertHooks(LiftedRepresentation &LR,
                                 const InstrumentationTask &Tasks);

  /// Compiles the relocatable object file in \p Code
  /// \param Code
  /// \param ISA
  /// \param Out
  /// \return
  static llvm::Error
  compileRelocatableToExecutable(const llvm::ArrayRef<uint8_t> &Code,
                                 const hsa::ISA &ISA,
                                 llvm::SmallVectorImpl<uint8_t> &Out);
};

/// \brief Iterate over the LiveIns of the MI and set them as reserved
/// so that the register allocator does not overwrite them in the generated
/// instrumentation kernel
class ReserveLiveRegs : public llvm::MachineFunctionPass {
public:
  static char ID;
  typedef llvm::DenseMap<llvm::Function *, std::unique_ptr<llvm::LivePhysRegs>>
      hook_live_regs_map_t;

private:
  hook_live_regs_map_t HookLiveRegs;
  llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
      HookToInsertionPointMap;

public:
  llvm::StringRef getPassName() const override { return "Reserve Live Regs"; }

  ReserveLiveRegs(const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
                      &MIToHookFuncMap);

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

// This custom pass iterates through the Instrumentation Modules frame objects
// and adds the amount of stack allocated by the IPointMI's parent Machine
// Function to the frame object offset
class StackFrameOffset : public llvm::MachineFunctionPass {
public:
  static char ID;

private:
  llvm::DenseMap<llvm::Function *, unsigned int> FrameOffset;

public:
  explicit StackFrameOffset(
      const LiftedRepresentation &LR,
      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
          &BeforeMIHooks,
      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
          &AfterMIHooks);

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;
};

class InstBundler : public llvm::MachineFunctionPass {
public:
  static char ID;

public:
  InstBundler() : llvm::MachineFunctionPass(ID){};

  llvm::StringRef getPassName() const override {
    return "luthier-inst-bundler";
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override;
};

} // namespace luthier

#endif