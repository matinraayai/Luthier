#ifndef CODE_GENERATOR_H
#define CODE_GENERATOR_H
#include "sibir_types.h"
#include "instr.h"

namespace sibir {
class CodeGenerator {
 public:
    CodeGenerator(const CodeGenerator &) = delete;
    CodeGenerator &operator=(const CodeGenerator &) = delete;

    static inline CodeGenerator &Instance() {
        static CodeGenerator instance;
        return instance;
    }

    static void instrument(Instr& instr, const std::string&instrumentationFunction, sibir_ipoint_t point);

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