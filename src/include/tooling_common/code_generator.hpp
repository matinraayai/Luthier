//===-- code_generator.hpp - Luthier Code Generator  ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's code generator, which instruments lifted
/// representations given a mutator function.
//===----------------------------------------------------------------------===//
#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "common/object_utils.hpp"
#include "common/singleton.hpp"
#include "hsa/hsa_executable.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include <llvm/ADT/Any.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <luthier/instrumentation_task.h>
#include <luthier/lifted_representation.h>
#include <luthier/types.h>

#include "tooling_common/intrinsic/IntrinsicLoweringInfo.hpp"
#include <utility>

namespace llvm {
class LivePhysRegs;
}

namespace luthier {

namespace hsa {

class ISA;

} // namespace hsa

class CodeGenerator : public Singleton<CodeGenerator> {
private:
  /// Holds information regarding how to lower Luthier intrinsics
  llvm::StringMap<IntrinsicProcessor> IntrinsicsProcessors;

public:
  /// Register a Luthier intrinsic with the <tt>CodeGenerator</tt> and provide a
  /// way to lower it to Machine IR
  /// \param Name the demangled function name of the intrinsic, without the
  /// template arguments
  /// \param Processor the \c IntrinsicProcessor describing how to lower the
  /// Luthier intrinsic
  void registerIntrinsic(llvm::StringRef Name, IntrinsicProcessor Processor) {
    IntrinsicsProcessors[Name] = std::move(Processor);
  }

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
  /// Returns the full demangled name of \p MangledFuncName without its template
  /// arguments; e.g. if the demangled function name is
  /// <tt>a::b::c<int>(int i)</tt>, then <tt>a::b::c</tt> is returned \n
  /// This name is used as the unique identifier of the intrinsic inside the
  /// \c CodeGenerator
  /// \param MangledIntrinsicName name of the intrinsic used in an
  /// instrumentation module
  /// \return on success, the full demangled name of the function, and an
  /// \c llvm::Error if an issue was encountered during the process
  static llvm::Expected<std::string>
  getDemangledIntrinsicName(llvm::StringRef MangledIntrinsicName);

  /// Finds <tt>llvm::Function</tt>s marked as intrinsics inside the
  /// \p InstModule and Applies the IR processor function to their
  /// <tt>llvm::User</tt>s
  /// \param [in, out] Module the instrumentation \c llvm::Module containing
  /// hook logic IR
  /// \param [in] TM the \c llvm::TargetMachine of the code generation process
  /// \param [out] InlineAsmMIRMap a map which keeps track of the unique dummy
  /// inline assembly instruction strings inserted inside the IR as well as
  /// their \c IntrinsicValueLoweringInfo used later in the MIR processing stage
  /// \return \c llvm::Error describing the success or failure of the
  /// operation
  llvm::Error processInstModuleIntrinsicsAtIRLevel(
      llvm::Module &Module, const llvm::GCNTargetMachine &TM,
      llvm::StringMap<IntrinsicIRLoweringInfo> &InlineAsmMIRMap);

  static llvm::Expected<llvm::Function &> generateHookIR(
      const llvm::MachineInstr &MI,
      llvm::ArrayRef<InstrumentationTask::hook_invocation_descriptor> HookSpecs,
      const llvm::GCNTargetMachine &TM, llvm::Module &IModule);

  llvm::Error insertHooks(LiftedRepresentation &LR,
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