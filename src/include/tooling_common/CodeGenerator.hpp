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
/// This file describes Luthier's code generator, a singleton in charge of
/// instrumenting lifted representation given an instrumentation task and
/// a mutator function.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_CODE_GENERATOR_HPP
#define LUTHIER_TOOLING_COMMON_CODE_GENERATOR_HPP
#include "common/Singleton.hpp"
#include "luthier/Intrinsic/IntrinsicProcessor.h"

namespace llvm {

class MachineModuleInfoWrapperPass;

} // namespace llvm

namespace luthier {

class InstrumentationTask;

class LiftedRepresentation;

namespace hsa {

class ISA;

} // namespace hsa

/// \brief Singleton in charge of generating instrumented machine code
/// \details <tt>CodeGenerator</tt> performs the following tasks:
/// 1. Create calls to hooks inside an instrumentation
/// <tt>llvm::Module</tt> (<tt>IModule</tt>), creating a collection of
/// injected payload <tt>llvm::Function</tts> inside the <tt>IModule</tt>. \n
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
};

} // namespace luthier

#endif