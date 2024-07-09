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
#include <utility>

namespace luthier {

class LiftedRepresentation;

class InstrumentationModule;

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
  /// Type of arguments that can be passed to a hook
  enum class ArgType : uint8_t {
    REGISTER,
    INT64,
  };

  typedef struct {
    /// Name of the hook to be inserted
    llvm::StringRef HookName;
    /// List of arguments passed to the hook
    llvm::SmallVector<std::pair<ArgType, uint64_t>, 1> Args;
  } mi_hook_insertion_task;

  typedef struct {
    /// The instrumented \c llvm::MachineInstr
    llvm::MachineInstr *MI;

    /// List of hooks inserted before the \c MI
    llvm::SmallVector<mi_hook_insertion_task, 1> BeforeIPointTasks;

    /// List of hooks inserted after the \c MI
    llvm::SmallVector<mi_hook_insertion_task, 1> AfterIPointTasks;

  } mi_hook_insertion_tasks;

  typedef llvm::DenseMap<llvm::MachineInstr *, mi_hook_insertion_tasks>
      hook_insertion_tasks;

private:
  /// The preset to instrument <tt>LiftedRepresentation</tt>s under
  std::string Preset;
  /// The instrumentation module used to instrument the
  /// <tt>LiftedRepresentation</tt>s
  const InstrumentationModule &IM;
  /// A list of hooks to be inserted at each \c llvm::MachineInstr of the
  /// <tt>LiftedRepresentation</tt>
  hook_insertion_tasks HookInsertionTasks{};

public:
  /// InstrumentationTask constructor
  /// \param Preset a name which a primitive will be instrumented
  /// under. The same HSA primitive cannot be instrumented under the same
  /// preset name. To have multiple instrumented versions of the same primitive,
  /// one can use different preset name to identify and keep track of them
  explicit InstrumentationTask(llvm::StringRef Preset);

  /// Queues a hook insertion task, which will insert a hook over the
  /// <tt>MI</tt>
  /// \param MI the \c llvm::MachineInstr the hook will be inserted over
  /// \param Hook handle of the hook obtained from \c LUTHIER_GET_HOOK_HANDLE
  /// \param IPoint where the hook will be inserted with respect to
  /// the <tt>MI</tt>
  /// \param Args A list of arguments to be passed to the hook; An empty list
  /// by default
  /// \returns an \c llvm::Error indicating the success of the operation or
  /// its failure
  llvm::Error
  insertHookAt(llvm::MachineInstr &MI, const void *Hook, InstrPoint IPoint,
               llvm::ArrayRef<std::pair<ArgType, uint64_t>> Args = {});

  /// \return a const reference to the hook insertion tasks
  [[nodiscard]] const hook_insertion_tasks &getHookInsertionTasks() const {
    return HookInsertionTasks;
  }

  /// \return a const reference to the instrumentation module of this task
  [[nodiscard]] const InstrumentationModule &getModule() const { return IM; }
};

} // namespace luthier

#endif