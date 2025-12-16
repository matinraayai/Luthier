//===-- InstrumentationTask.h - Instrumentation Task ------------*- C++ -*-===//
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
/// \file
/// This file describes the instrumentation task, an interface for tools to
/// describe how a Lifted Representation should be instrumented.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INSTRUMENTATION_TASK_H
#define LUTHIER_TOOLING_INSTRUMENTATION_TASK_H
#include "luthier/types.h"
#include <functional>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <utility>
#include <variant>

namespace luthier {

class LiftedRepresentation;

class InstrumentationModule;

/// \brief keeps track of modifications to be performed on on a
/// \c LiftedRepresentation in order to create an instrumented version of an HSA
/// execution primitive (i.e. <tt>hsa_executable_symbol_t</tt>, or
/// <tt>hsa_executable_t</tt>)
/// \details instrumentation task consists of the following:\n
/// 1. A "preset" name, identifying the instrumentation task. An HSA primitive
/// cannot be instrumented under the same name. Different instrumented versions
/// of the same HSA primitive must have different preset names.\n
/// 2. A function allowing the user to describe how to instrument a
/// <tt>LiftedRepresentation</tt>. This function is applied to a cloned version
/// of a <tt>LiftedRepresentation</tt>, and it is designed to be the only
/// user-facing place which allows directly mutating the
/// <tt>LiftedRepresentation</tt> (e.g. adding <tt>llvm::MachineInstr</tt>'s
/// using the Machine Instruction builder API). Outside of this function, the
/// only way to modify the \c InstrumentationTask is to insert hooks via the
/// \c InstrumentationTask::insertHookBefore function.\n
/// Objects of this class as well as a <tt>LiftedRepresentation<tt> of an
/// HSA primitive are passed to the \c luthier::instrumentAndLoad function.
class InstrumentationTask {
public:
  typedef struct {
    /// Name of the hook to be inserted
    llvm::StringRef HookName;
    /// List of arguments passed to the hook
    llvm::SmallVector<std::variant<llvm::Constant *, llvm::MCRegister>, 1> Args;
  } hook_invocation_descriptor;

  /// A mapping of a \c llvm::MachineInstr to the hooks + their arguments
  /// to be inserted before it
  typedef llvm::DenseMap<llvm::MachineInstr *,
                         llvm::SmallVector<hook_invocation_descriptor, 1>>
      hook_insertion_tasks;

private:
  /// The \c LiftedRepresentation being instrumented
  LiftedRepresentation &LR;
  /// The instrumentation module used to instrument the
  /// <tt>LiftedRepresentation</tt>s
  const InstrumentationModule &IM;
  /// A list of hooks to be inserted at each \c llvm::MachineInstr of the
  /// <tt>LiftedRepresentation</tt>
  hook_insertion_tasks HookInsertionTasks{};

public:
  /// InstrumentationTask constructor
  /// \param LR the \c LiftedRepresentation being instrumented
  explicit InstrumentationTask(LiftedRepresentation & LR);

  /// Queues a hook insertion task, which will insert a hook before the
  /// \p MI \n
  /// There is no "<tt>insertHookAfter</tt> variant to prevent insertion of
  /// instructions after the block's terminator instruction
  /// \param MI the \c llvm::MachineInstr the hook will be inserted before
  /// \param Hook handle of the hook obtained from \c LUTHIER_GET_HOOK_HANDLE
  /// \param Args A list of arguments to be passed to the hook; An empty list
  /// by default
  /// \returns an \c llvm::Error indicating the success of the operation or
  /// its failure
  llvm::Error insertHookBefore(
      llvm::MachineInstr &MI, const void *Hook,
      llvm::ArrayRef<std::variant<llvm::Constant *, llvm::MCRegister>> Args =
          {});

  /// \return a const reference to the hook insertion tasks
  [[nodiscard]] const hook_insertion_tasks &getHookInsertionTasks() const {
    return HookInsertionTasks;
  }

  /// \return a const reference to the instrumentation module of this task
  [[nodiscard]] const InstrumentationModule &getModule() const { return IM; }
};

} // namespace luthier

#endif