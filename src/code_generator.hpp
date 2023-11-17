#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP
#include "code_object_manipulation.hpp"
#include "instr.hpp"
#include "luthier_types.hpp"
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

    static luthier::co_manip::code_t assemble(const std::string &instList, hsa_agent_t agent);

    static luthier::co_manip::code_t assemble(const std::vector<std::string> &instList, hsa_agent_t agent);

    static luthier::co_manip::code_t assembleToRelocatable(const std::string &instList, hsa_agent_t agent);

    static luthier::co_manip::code_t assembleToRelocatable(const std::vector<std::string> &instList, hsa_agent_t agent);

    static luthier::co_manip::code_t compileRelocatableToExecutable(const luthier::co_manip::code_t &code, hsa_agent_t agent);

    void instrument(Instr &instr, const void *dev_func,
                    luthier_ipoint_t point);
    void modify(Instr &instr, void *my_addr);

 private:
    /**
     * A map of agent to its empty relocatable. Empty relocatables have only an s_nop instructions.
     * The relocatables get assembled when the CodeGenerator first gets called
     */
    std::unordered_map<decltype(hsa_agent_t::handle), luthier::co_manip::code_t> emptyRelocatableMap_;

    CodeGenerator();
    ~CodeGenerator() {}
};
}// namespace luthier

#endif