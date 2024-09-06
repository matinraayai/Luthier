//===-- CodeGenerator.hpp - Luthier Code Generator ------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's code generator, which instruments lifted
/// representations given a mutator function.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_CODE_GENERATOR_HPP
#define LUTHIER_TOOLING_COMMON_CODE_GENERATOR_HPP

#include "common/ObjectUtils.hpp"
#include "common/Singleton.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include <llvm/ADT/Any.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <luthier/InstrumentationTask.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>
#include <luthier/LiftedRepresentation.h>
#include <luthier/types.h>

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
    IntrinsicsProcessors.insert({Name, std::move(Processor)});
  }

  /// Instruments the passed \p LR by first cloning it and then
  /// applying the \p Mutator onto its contents
  /// \param LR the \c LiftedRepresentation about to be instrumented
  /// \param Mutator
  /// \return a new \c LiftedRepresentation containing the instrumented code,
  /// or an \c llvm::Error in case an issue was encountered during the process
  llvm::Expected<std::unique_ptr<LiftedRepresentation>>
  instrument(const LiftedRepresentation &LR,
             llvm::function_ref<llvm::Error(InstrumentationTask &,
                                            LiftedRepresentation &)>
                 Mutator);

  /// Runs the \c llvm::AsmPrinter pass on the \p Module and the
  /// \c llvm::MachineModuleInfo of the \p MMIWP to generate a relocatable file
  /// \note This function does not access the Module's \c llvm::LLVMContext in a
  /// thread-safe manner
  /// \note After printing, \p MMIWP will be deleted by the legacy pass manager
  /// used to print the assembly file
  /// \param [in] Module the \c llvm::Module to be printed
  /// \param [in] TM the \c llvm::GCNTargetMachine the \p MMIWP was created with
  /// \param [in] MMIWP the \c llvm::MachineModuleInfoWrapperPass to be printed;
  /// \param [out] CompiledObjectFile the compiled relocatable file
  /// \param FileType type of the <tt>CompiledObjectFile</tt>;
  /// Either \c llvm::CodeGenFileType::AssemblyFile or
  /// \c llvm::CodeGenFileType::ObjectFile
  /// \return an \c llvm::Error in case of any issues encountered during the
  /// process
  static llvm::Error
  printAssembly(llvm::Module &Module, llvm::GCNTargetMachine &TM,
                llvm::MachineModuleInfoWrapperPass *MMIWP,
                llvm::SmallVectorImpl<char> &CompiledObjectFile,
                llvm::CodeGenFileType FileType);

  /// Links the relocatable object file passed in \p Code to an executable,
  /// which can then be loaded into the HSA runtime
  /// \param [in] Code the relocatable file
  /// \param [in] ISA the ISA of the relocatable file
  /// \param [out] Out the linked executable
  /// \return an \c llvm::Error in case any issues were encountered during the
  /// process
  static llvm::Error
  linkRelocatableToExecutable(const llvm::ArrayRef<char> &Code,
                              const hsa::ISA &ISA,
                              llvm::SmallVectorImpl<uint8_t> &Out);

private:
  /// Finds <tt>llvm::Function</tt>s marked as intrinsics inside the
  /// \p InstModule and applies the IR processor function to their
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
      llvm::SmallVectorImpl<
          std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
          &InlineAsmMIRMap);

  std::pair<llvm::MachineModuleInfoWrapperPass *,
            std::unique_ptr<llvm::legacy::PassManager>>
  runCodeGenPipeline(
      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
          &MIToHookFuncMap,
      const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
          &HookFuncToMIMap,
      const llvm::SmallVectorImpl<
          std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
          &ToBeLoweredIntrinsics,
      bool DisableVerify, const LiftedRepresentation &LR,
      const luthier::hsa::LoadedCodeObject &LCO, llvm::GCNTargetMachine *TM,
      llvm::Module &M);

  static llvm::Expected<llvm::Function &> generateHookIR(
      const llvm::MachineInstr &MI,
      llvm::ArrayRef<InstrumentationTask::hook_invocation_descriptor> HookSpecs,
      const llvm::GCNTargetMachine &TM, llvm::Module &IModule);

  llvm::Error insertHooks(LiftedRepresentation &LR,
                          const InstrumentationTask &Tasks);
};

///// \brief Aggregates the app's live registers at each hook insertion point,
///// as well as physical registers read by the hook itself, and creates def/uses
///// for them in appropriate places. If the target app uses the stack, creates
///// a stack operand as well as appropriate defs/uses.
///// TODO: Make this pass work with dynamic stack usage
//class DefineLiveRegsAndAppStackUsagePass : public llvm::MachineFunctionPass {
//public:
//  static char ID;
//  typedef llvm::DenseMap<llvm::Function *, llvm::LivePhysRegs *>
//      hook_live_regs_map_t;
//
//private:
//  hook_live_regs_map_t HookLiveRegs;
//
//  llvm::DenseMap<llvm::Function *, size_t> StaticSizedHooksToStackSize;
//
//public:
//  llvm::StringRef getPassName() const override {
//    return "Define Live Regs and Stack Usage Pass";
//  }
//
//  explicit DefineLiveRegsAndAppStackUsagePass(
//      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
//          &MIToHookFuncMap,
//      const LiftedRepresentation &LR);
//
//  bool runOnMachineFunction(llvm::MachineFunction &MF) override;
//
//  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
//};

} // namespace luthier

#endif