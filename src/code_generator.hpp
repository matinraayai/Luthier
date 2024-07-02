//===-- code_generator.hpp - Luthier Code Generator  ----------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's code generator, which instruments lifted
/// representation given an instrumentation task.
//===----------------------------------------------------------------------===//
#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "object_utils.hpp"
#include "singleton.hpp"
#include <luthier/instrumentation_task.h>
#include <luthier/lifted_representation.h>
#include <luthier/types.h>

namespace luthier {

namespace hsa {

class ISA;

} // namespace hsa

class CodeGenerator : public Singleton<CodeGenerator> {
public:
  llvm::Error instrument(const LiftedRepresentation &LR,
                         InstrumentationTask &ITask);

private:

  static llvm::Error
  insertHooks(LiftedRepresentation &LR,
              const InstrumentationTask::hook_insertion_tasks &Tasks);

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
} // namespace luthier

#endif