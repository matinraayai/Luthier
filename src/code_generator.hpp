#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP
#include "instr.hpp"
#include "luthier_types.hpp"

namespace luthier {
class CodeGenerator {
 public:
    CodeGenerator(const CodeGenerator &) = delete;
    CodeGenerator &operator=(const CodeGenerator &) = delete;

    static inline CodeGenerator &instance() {
        static CodeGenerator instance;
        return instance;
    }

    static void instrument(Instr &instr, const std::string &instrumentationFunction, luthier_ipoint_t point);

    static void modify(Instr &instr, void *my_addr);

 private:
    typedef struct {
        const std::string name;
        const void *hostFunction;
        const std::string deviceName;
        const void *parentFatBinary;
    } function_info_t;

    CodeGenerator() {}
    ~CodeGenerator() {}
};
}// namespace luthier

#endif