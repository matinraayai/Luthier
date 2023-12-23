#ifndef INSTRUMENTATION_FUNCTION_HPP
#define INSTRUMENTATION_FUNCTION_HPP

#include "luthier_types.h"

namespace luthier {

/**
 * @brief
 */
class InstrumentationFunction {
 private:
    const kernel_descriptor_t *kd_;
    const hsa_executable_t executable_;
    const hsa_agent_t agent_;
    const hsa_executable_symbol_t
};
}// namespace luthier

#endif