#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "luthier_types.h"
#include "object_utils.hpp"

namespace luthier {

namespace hsa {
class GpuAgent;

class ISA;

class Instr;
} // namespace hsa

class CodeGenerator {
public:
  CodeGenerator(const CodeGenerator &) = delete;
  CodeGenerator &operator=(const CodeGenerator &) = delete;

  static inline CodeGenerator &instance() {
    static CodeGenerator instance;
    return instance;
  }

  static llvm::Error
  compileRelocatableToExecutable(const llvm::ArrayRef<uint8_t> &code,
                                 const hsa::GpuAgent &agent,
                                 llvm::SmallVectorImpl<uint8_t> &out);

  static llvm::Error
  compileRelocatableToExecutable(const llvm::ArrayRef<uint8_t> &code,
                                 const hsa::ISA &isa,
                                 llvm::SmallVectorImpl<uint8_t> &out);

  llvm::Error instrument(hsa::Instr &instr, const void *devFunc,
                         luthier_ipoint_t point);

private:
  CodeGenerator() = default;
  ~CodeGenerator() = default;
};
} // namespace luthier

#endif