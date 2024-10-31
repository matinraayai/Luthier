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

struct PreKernelEmissionDescriptor;

class LRRegisterLiveness;

class LRCallGraph;

namespace hsa {

class ISA;

} // namespace hsa

/// \brief Singleton in charge of generating instrumented machine code
/// \details <tt>CodeGenerator</tt> performs the following tasks:
/// 1. Create calls to hooks inside an instrumentation
/// <tt>llvm::Module</tt> (<tt>IModule</tt>), creating a collection of
/// instrumentation <tt>llvm::Function</tts> inside the <tt>IModule</tt>. \n
/// 2. Run the IR optimization pipeline on the instrumentation module to
/// optimize the instrumentation functions. \n
/// 3. Run the IR lowering functions of Luthier intrinsics. \n
/// 4. Run a modified version of the LLVM CodeGen pipeline on the
/// instrumentation module, which involves: a) running normal ISEL,
/// b) calling MIR lowering functions on intrinsics, c) virtualizing access to
/// physical registers, and expressing register constraints in MIR, d)
/// a custom frame lowering after register allocation and lowering of stack
/// operands inside instrumentation Module functions.
/// 5. Keep track of how each intrinsic is lowered; There are a set of
/// intrinsics built-in for Luthier (e.g. <tt>readReg</tt>) and there are a set
/// of intrinsics which a tool writer can register by describing how they
/// are lowered.
class CodeGenerator : public Singleton<CodeGenerator> {
private:
  /// Holds information regarding how to lower Luthier intrinsics
  llvm::StringMap<IntrinsicProcessor> IntrinsicsProcessors;

public:
  /// Register a Luthier intrinsic with the <tt>CodeGenerator</tt> and provide a
  /// way to lower it to Machine IR
  /// \param Name the demangled function name of the intrinsic, without the
  /// template arguments but with the namespace(s) its binding is defined
  /// (e.g. <tt>"luthier::readReg"</tt>)
  /// \param Processor the \c IntrinsicProcessor describing how to lower the
  /// Luthier intrinsic
  void registerIntrinsic(llvm::StringRef Name, IntrinsicProcessor Processor) {
    IntrinsicsProcessors.insert({Name, std::move(Processor)});
  }

  /// Instruments the passed \p LR by first cloning it and then
  /// applying the \p Mutator onto its contents
  /// \param LR the \c LiftedRepresentation about to be instrumented
  /// \param Mutator a function that can modify the lifted representation
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
                std::unique_ptr<llvm::MachineModuleInfoWrapperPass> &MMIWP,
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
  /// Applies the instrumentation task \p Task to the lifted representation
  /// of \p LR \n
  /// The \p Task is created and populated by the mutator function
  /// in <tt>CodeGenerator::instrument</tt>
  /// \param [in] Task the \c InstrumentationTask applied to the \p LR which
  /// contains a set of hook calls that will be injected before a set of
  /// <tt>llvm::MachineInstr</tt>s of the target application
  /// \param [in, out] LR the \c LiftedRepresentation being instrumented
  /// \return an \c llvm::Error indicating if any issues where encountered
  /// during the process
  llvm::Error applyInstrumentationTask(const InstrumentationTask &Task,
                                       LiftedRepresentation &LR);

  /// Generates an instrumentation \c llvm::Function containing the
  /// code that will be injected before the \p MI in the target
  /// application; The injected payload function is a series of calls to
  /// the hooks with the arguments, in the order they are specified by
  /// \p HookInvocationSpecs
  /// \param IModule the instrumentation \c llvm::Module that will encapsulate
  /// the generated injected payload function
  /// \param HookInvocationSpecs A list of hooks to be called inside the
  /// injected payload, as well as their arguments
  /// \param ApplicationMI the \c llvm::MachineInstr inside the application
  /// that will get instrumented with the injected payload's contents
  /// \return a reference to a newly-created \c llvm::Function (with no
  /// arguments and return value) which contains the LLVM IR of the payload
  /// to be injected before the target application's \p MI
  static llvm::Expected<llvm::Function &>
  generateInjectedPayloadForApplicationMI(
      llvm::Module &IModule,
      llvm::ArrayRef<InstrumentationTask::hook_invocation_descriptor>
          HookInvocationSpecs,
      const llvm::MachineInstr &ApplicationMI);

  /// Finds <tt>llvm::Function</tt>s marked as intrinsics inside the
  /// \p IModule, applies the IR processor function to their
  /// <tt>llvm::User</tt>s, and returns the saved information regarding
  /// the IR lowering stage in \p IntrinsicLoweringInfoVec
  /// \param [in, out] IModule the instrumentation \c llvm::Module containing
  /// the instrumentation logic
  /// \param [in] TM the \c llvm::TargetMachine of the code generation process;
  /// Mainly passed to the IR Lowering processors of all intrinsics
  /// \param [out] IntrinsicLoweringInfoVec a vector which keeps track of the
  /// \c IntrinsicIRLoweringInfo of each intrinsic use lowered. The index of
  /// the lowering info inside the vector is the number assigned in its dummy
  /// inline assembly instruction string
  /// \return \c llvm::Error describing the success or failure of the
  /// operation
  llvm::Error processIModuleIntrinsicUsersAtIRLevel(
      llvm::Module &IModule, const llvm::GCNTargetMachine &TM,
      llvm::SmallVectorImpl<IntrinsicIRLoweringInfo> &IntrinsicLoweringInfoVec);

  /// Applies the custom CodeGen pipeline to the \p IModule
  /// \param [in] IModule the instrumentation module being worked on
  /// \param [in] LR the \c LiftedRepresentation being worked on
  /// \param [in] LCO the \c hsa::LoadedCodeObject of \p LR currently being
  /// worked on
  /// \param [in] LRLiveRegs the live register analysis of \p LR providing the
  /// set of live registers at each machine instruction inside \p LR
  /// \param [in] CG the call graph analysis for the \p LR
  /// \param [in] InstPointToInjectedPayloadMap Mapping between instrumentation
  /// points inside the \p LR to their injected payload function inside
  /// the instrumentation module
  /// \param [in] InjectedPayloadToInstPointMap Inverse mapping of
  /// \p InstPointToInjectedPayloadMap
  /// \param [in] IntrinsicIRLoweringInfoVec Set of intrinsics that need to be
  /// lowered across the \p IModule
  /// \param [out] PM a legacy pass manager to use for running the CodeGen
  /// pipeline, should be empty
  /// \param [in] TM the \c llvm::GCNTargetMachine being targeted
  /// \param [out] IMMIWP a \c llvm::MachineModuleInfoWrapperPass where the
  /// final machine code of \p IModule will be stored into
  /// \param [out] PKInfo the pre-kernel emission info which will be used to
  ///
  /// \param DisableVerify whether to disable running machine verification
  /// passe right after each pass or not
  /// \return an \c llvm::Error indicating the success of failure of the
  /// operation
  llvm::Error runModifiedCodeGenPipelineOnIModule(
      llvm::Module &IModule, const LiftedRepresentation &LR,
      const hsa::LoadedCodeObject &LCO, const LRRegisterLiveness &LRLiveRegs,
      const LRCallGraph &CG,
      const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
          &InstPointToInjectedPayloadMap,
      const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
          &InjectedPayloadToInstPointMap,
      const llvm::SmallVectorImpl<IntrinsicIRLoweringInfo>
          &IntrinsicIRLoweringInfoVec,
      llvm::legacy::PassManager &PM, llvm::GCNTargetMachine &TM,
      llvm::MachineModuleInfoWrapperPass &IMMIWP,
      PreKernelEmissionDescriptor PKInfo, bool DisableVerify);
};

} // namespace luthier

#endif