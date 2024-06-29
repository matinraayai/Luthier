//===-- instrumentation_task.h - Instrumentation Task -----------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
///
/// This file describes the instrumentation task, which allows a tool writer to
/// describe how a Lifted Representation is instrumented.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_TASK_H
#define LUTHIER_INSTRUMENTATION_TASK_H
#include "types.h"
#include <functional>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineInstr.h>

namespace luthier {

class LiftedRepresentation;

/// \brief keeps track of modifications to be performed on on a
/// \c LiftedRepresentation in order to create an instrumented version of an HSA
/// primitive
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
/// \c InstrumentationTask::insertHookAt function.\n
/// Objects of this class as well as a <tt>LiftedRepresentation<tt> of an
/// HSA primitive are passed to the \c luthier::instrumentAndLoad function.
class InstrumentationTask {
public:
  typedef llvm::DenseMap<llvm::MachineInstr *,
                         std::tuple<const void *, InstrPoint>>
      hook_insertion_tasks;

  typedef std::function<llvm::Error(luthier::InstrumentationTask &)>
      mutator_func_t;

private:
  std::string Preset;
  hook_insertion_tasks HookInsertionTasks{};
  const mutator_func_t MutatorFunction;

public:
  /// InstrumentationTask constructor
  /// \param Preset a name which a primitive will be instrumented
  /// under. The same HSA primitive cannot be instrumented under the same
  /// preset name. To have multiple instrumented versions of the same primitive,
  /// one can use different preset name to identify and keep track of them.
  /// \param Mutator a function which allows iteration and mutation of the
  /// <tt>LiftedRepresentation</tt>
  InstrumentationTask(llvm::StringRef Preset, mutator_func_t Mutator)
      : Preset(Preset), MutatorFunction(std::move(Mutator)){};

  /// Queues a hook insertion task, which will insert a hook over the
  /// <tt>MI</tt>
  /// \param MI the \c llvm::MachineInstr the hook will be inserted over
  /// \param Hook handle of the hook obtained from \c LUTHIER_GET_HOOK_HANDLE
  /// \param IPoint where the hook will be inserted with respect to
  /// the <tt>MI</tt>
  void insertHookAt(llvm::MachineInstr &MI, const void *Hook,
                    InstrPoint IPoint);

  /// \return a const reference to the hook insertion tasks
  [[nodiscard]] const hook_insertion_tasks &getInsertCallTasks() const {
    return HookInsertionTasks;
  }

  /// \return a const reference to the mutator function
  [[nodiscard]] const mutator_func_t &getMutator() { return MutatorFunction; }
};

} // namespace luthier

#endif