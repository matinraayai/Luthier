#ifndef LUTHIER_INSTRUMENTATION_TASK_H
#define LUTHIER_INSTRUMENTATION_TASK_H

namespace luthier {
/// \brief Contains all the instrumentation tasks that needs to be performed
/// on a Module
class InstrumentationTask {
public:
  typedef llvm::DenseMap<llvm::MachineInstr *,
                         std::tuple<const void *, InstrPoint>>
      insert_call_tasks;

private:
  insert_call_tasks InsertCallTasks;

public:
  InstrumentationTask() = default;

  void insertCallTo(llvm::MachineInstr &MI, const void *DevFunc,
                    InstrPoint IPoint);

  [[nodiscard]] const insert_call_tasks &getInsertCallTasks() const {
    return InsertCallTasks;
  }
};

}

#endif