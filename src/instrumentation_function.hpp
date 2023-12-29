#ifndef INSTRUMENTATION_FUNCTION_HPP
#define INSTRUMENTATION_FUNCTION_HPP

#include <utility>

#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "luthier_types.h"

namespace luthier {

/**
 * @brief Contains information required for the tool's instrumentation function.
 */
class InstrumentationFunction {
 private:
    const hsa::ExecutableSymbol instrumentationFunction_;
    const hsa::ExecutableSymbol instrumentationKernel_;

 public:
    InstrumentationFunction(hsa::ExecutableSymbol instrumentationFunction, hsa::ExecutableSymbol instrumentationKernel)
        : instrumentationFunction_(std::move(instrumentationFunction)),
          instrumentationKernel_(std::move(instrumentationKernel)) {
        assert(instrumentationKernel_.getAgent() == instrumentationFunction_.getAgent());
        assert(instrumentationKernel_.getExecutable() == instrumentationFunction_.getExecutable());
    };

    [[nodiscard]] hsa::GpuAgent getAgent() const { return instrumentationFunction_.getAgent(); }

    [[nodiscard]] hsa::Executable getExecutable() const { return instrumentationFunction_.getExecutable(); }

    [[nodiscard]] const hsa::ExecutableSymbol& getInstrumentationFunction() const { return instrumentationFunction_; };

    [[nodiscard]] const hsa::ExecutableSymbol& getInstrumentationKernel() const { return instrumentationKernel_; };
};
}// namespace luthier

#endif