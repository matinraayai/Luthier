//===-- InstrumentationContext.h --------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This file describes the <tt>InstrumentationContext</tt> class. It contains
/// the necessary LLVM constructs to create <tt>llvm::Module</tt>s and
/// <tt>llvm::MachineModuleInfo</tt>s for both inspection and instrumentation
/// tasks.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INSTRUMENTATION_CONTEXT_H
#define LUTHIER_TOOLING_INSTRUMENTATION_CONTEXT_H
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Target/TargetMachine.h>

namespace llvm::object {

class ObjectFile;

}

namespace luthier {

/// \brief holds on to the \c LLVMContext and \c TargetMachine instances
/// used to create <tt>llvm::Module</tt>s and <tt>llvm::MachineModuleInfo</tt>s
/// for both inspection and instrumentation tasks
class InstrumentationContext {
private:
  /// A mapping between the ISA strings and their target machines
  llvm::StringMap<std::unique_ptr<llvm::TargetMachine>> TMs{};

  /// A thread-safe context; Its lock is used to protect the contents
  /// inside the instrumentation context
  llvm::orc::ThreadSafeContext Context;

public:

  explicit InstrumentationContext(std::unique_ptr<llvm::LLVMContext> Ctx)
      : Context(std::move(Ctx)) {};

  /// Default constructor
  InstrumentationContext() : Context(std::make_unique<llvm::LLVMContext>()) {};

  /// \return a scoped lock protecting the contents of the instrumentation
  /// context
  llvm::orc::ThreadSafeContext::Lock getLock() const {
    return Context.getLock();
  }

  llvm::LLVMContext &getLLVMContext() { return *Context.getContext(); }

  [[nodiscard]] const llvm::LLVMContext &getLLVMContext() const {
    return *Context.getContext();
  }

  /// Queries the context for the \c TargetMachine for the ISA specified by
  /// <tt>TT</tt>, <tt>CPU</tt>, and <tt>STF</tt>.
  /// If the context has not yet created a \c TargetMachine for the ISA it
  /// will create one and caches it
  /// \param TT the \c llvm::TargetTriple of the ISA
  /// \param CPU the processor name of the ISA; Must be set to
  /// <tt>"unknown"</tt> when no CPU name is available
  /// \param STF the \c llvm::SubTargetFeatures of the ISA
  /// \return on success, a reference to the \c llvm::TargetMachine of the ISA;
  /// an \c llvm::Error indicating the issue encountered during the process
  llvm::Expected<llvm::TargetMachine &>
  getOrCreateTargetMachine(const llvm::Triple &TT, llvm::StringRef CPU,
                           const llvm::SubtargetFeatures &STF);

  /// Queries the context for the \c TargetMachine for the ISA specified by
  /// <tt>TT</tt>, <tt>CPU</tt>, and <tt>STF</tt>.
  /// \param TT the \c llvm::TargetTriple of the ISA
  /// \param CPU the processor name of the ISA; Must be set to
  /// <tt>"unknown"</tt> when no CPU name is available
  /// \param STF the \c llvm::SubTargetFeatures of the ISA
  /// \return if a \c llvm::TargetMachine for the ISA is found, a pointer
  /// to it is returned; Otherwise, \c nullptr is returned
  llvm::TargetMachine *getTargetMachine(const llvm::Triple &TT,
                                        llvm::StringRef CPU,
                                        const llvm::SubtargetFeatures &STF);
};

} // namespace luthier

#endif