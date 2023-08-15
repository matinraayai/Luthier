#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP
#include "code_object_manipulation.hpp"
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

    static void instrument(Instr &instr, const void* dev_func,
                           luthier_ipoint_t point);

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
}

#endif