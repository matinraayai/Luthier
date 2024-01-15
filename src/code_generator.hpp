#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP
#include <unordered_map>

#include "code_view.hpp"
#include "hsa_agent.hpp"
#include "hsa_instr.hpp"
#include "hsa_isa.hpp"
#include "luthier_types.h"
#include "target_manager.hpp"

namespace luthier {
class CodeGenerator {
 public:
  CodeGenerator(const CodeGenerator &) = delete;
  CodeGenerator &operator=(const CodeGenerator &) = delete;

  static inline CodeGenerator &instance() {
      static CodeGenerator instance;
      return instance;
  }

  static luthier::byte_string_t assemble(const std::string &instList, const hsa::GpuAgent &agent);

  static luthier::byte_string_t assemble(const std::vector<std::string> &instList, const hsa::GpuAgent &agent);

  static luthier::byte_string_t assembleToRelocatable(const std::string &instList, const hsa::GpuAgent &agent);

  static luthier::byte_string_t assembleToRelocatable(const std::vector<std::string> &instList,
                                                      const hsa::GpuAgent &agent);

  static luthier::byte_string_t compileRelocatableToExecutable(const luthier::byte_string_t &code,
                                                               const hsa::GpuAgent &agent);

  void instrument(hsa::Instr &instr, const void *devFunc, luthier_ipoint_t point);

 private:

  unsigned int getTargetSpecificOpcode(const hsa::Isa &isa, unsigned int targetAgnosticOpcode) {
      return llvmTargetInstructions_.at(isa).at(targetAgnosticOpcode);
  }

  template<typename... Ops, typename = std::enable_if_t<std::conjunction_v<std::is_same<llvm::MCOperand, Ops>...>>>
  llvm::MCInst makeInstruction(const luthier::hsa::Isa &isa, unsigned int opcodeId, const Ops &... operands) {
      llvm::MCInst out;
      unsigned int targetOperand = llvmTargetInstructions_.at(isa).at(opcodeId);
      const auto &targetInfo = luthier::TargetManager::instance().getTargetInfo(isa);

      out.setOpcode(targetOperand);
      for (const auto &op: {operands...}) {
          out.addOperand(op);
      }
      return out;
  }

  /**
   * A map of agent to its empty relocatable. Empty relocatables have only an s_nop instructions.
   * The relocatables get assembled when the CodeGenerator first gets called
   */
  std::unordered_map<hsa::GpuAgent, luthier::byte_string_t> emptyRelocatableMap_;

  /**
   * \brief LLVM tablegen creates two types of enumerators for each instructions; One without specifying the target ISA,
   * and one that does (usually suffixed at the end of the instruction name e.g. _gfx8, _vi).
   * If the target-agnostic enum is used to create an llvm::MCInst, it cannot be emitted to the correct
   * machine instruction by the llvm::MCCodeEmitter. This map is initialized by the code generator, to let developers
   * create instructions without using the target-specific opcode variant.
   * There are some instructions that only have instructions with targets, without a target-agnostic counterpart.
   * We have no choice but to use the target-specific opcodes for now.
   * TODO: Eventually, this map should be generated before/during compile time for all/specified possible targets
   * TODO: Create an enum for the target-specific opcodes that don't have a target-agnostic enum
   */
  std::unordered_map<hsa::Isa, std::unordered_map<unsigned int, unsigned int>> llvmTargetInstructions_;

  CodeGenerator();
  ~CodeGenerator() {}
};
}// namespace luthier

#endif