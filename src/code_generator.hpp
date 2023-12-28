#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP
#include "code_view.hpp"
#include "hsa_agent.hpp"
#include "hsa_instr.hpp"
#include "luthier_types.h"
#include <unordered_map>

namespace luthier {
class CodeGenerator {
 public:
    CodeGenerator(const CodeGenerator &) = delete;
    CodeGenerator &operator=(const CodeGenerator &) = delete;

    static inline CodeGenerator &instance() {
        static CodeGenerator instance;
        return instance;
    }

    static luthier::byte_string_t assemble(const std::string &instList, const hsa::GpuAgent& agent);

    static luthier::byte_string_t assemble(const std::vector<std::string> &instList, const hsa::GpuAgent& agent);

    static luthier::byte_string_t assembleToRelocatable(const std::string &instList, const hsa::GpuAgent& agent);

    static luthier::byte_string_t assembleToRelocatable(const std::vector<std::string> &instList, const hsa::GpuAgent&  agent);

    static luthier::byte_string_t compileRelocatableToExecutable(const luthier::byte_string_t &code, const hsa::GpuAgent&  agent);

    void instrument(hsa::Instr &instr, const void *dev_func,
                    luthier_ipoint_t point);

 private:
    /**
     * A map of agent to its empty relocatable. Empty relocatables have only an s_nop instructions.
     * The relocatables get assembled when the CodeGenerator first gets called
     */
    std::unordered_map<hsa::GpuAgent, luthier::byte_string_t> emptyRelocatableMap_;

    CodeGenerator();
    ~CodeGenerator() {}
};
}// namespace luthier

#endif