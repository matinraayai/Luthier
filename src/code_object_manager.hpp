#ifndef CODE_OBJECT_MANAGER_HPP
#define CODE_OBJECT_MANAGER_HPP
#include "code_object_manipulation.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_handle_type.hpp"
#include "instrumentation_function.hpp"
#include "luthier_types.h"
#include <amd_comgr/amd_comgr.h>
#include <hsa/hsa.h>
#include <memory>
#include <set>
#include <vector>

namespace luthier {
/**
 * @brief A singleton object that keeps track of instrumentation functions and instrumented kernels
 * (per un-instrumented application kernel) in Luthier.
 */
class CodeObjectManager {
 public:
    CodeObjectManager(const CodeObjectManager &) = delete;
    CodeObjectManager &operator=(const CodeObjectManager &) = delete;

    static inline CodeObjectManager &instance() {
        static CodeObjectManager instance;
        return instance;
    }

    /**
     * Registers the wrapper kernels of Luthier tool's instrumentation functions.
     * The wrapper kernel body is not used. Their host shadow pointer and their names are used instead.
     * Their kernel descriptors are used to keep track of the instrumentation function register requirements
     * @param instrumentationFunctionInfo a list of tuples, the first element being the host shadow pointer of the wrapper kernels,
     * the second being the name of the wrapper kernel
     */
    void registerHipWrapperKernelsOfInstrumentationFunctions(const std::vector<std::tuple<const void *, const char *>>
                                                                 &instrumentationFunctionInfo);

    /**
     * Returns the hsa::ExecutableSymbol of the instrumentation function, given its HSA agent and wrapper kernel host ptr
     * @param wrapperKernelHostPtr shadow host pointer of the wrapper kernel used to force compilation of the
     * instrumentation function
     * @param agent the GPU HSA agent where the instrumentation function is loaded on
     * @return the instrumentation function symbol
     */
    const hsa::ExecutableSymbol& getInstrumentationFunction(const void *wrapperKernelHostPtr, hsa::GpuAgent agent) const;

    /**
     * Returns the kernel descriptor of the wrapper kernel associated with an instrumentation function, given its wrapper kernel
     * shadow host pointer and the HSA GPU Agent it is loaded on
     * @param wrapperKernelHostPtr shadow host pointer of the wrapper kernel used to force compilation of the
     * instrumentation function
     * @param agent the HSA GPU Agent the wrapper kernel is loaded on
     * @return a pointer to the kernel descriptor located in host-accessible device memory
     */
    const hsa::ExecutableSymbol& getInstrumentationKernel(const void *wrapperKernelHostPtr, hsa::GpuAgent agent) const;


    /**
     * Registers an instrumented version of a target application kernel. \class CodeObjectManager keeps track of instrumented kernels
     * via their kernel descriptors located in device memory. Kernel descriptors can then be used to query information about the kernel
     * (e.g. start of the kernel)
     * Both kernel descriptors should be located on the same device
     * @param originalCodeKD device address of the target application kernel's descriptor
     * @param instrumentedCodeKD device address of the instrumented kernel's descriptor
     */
    void registerInstrumentedKernel(const hsa::ExecutableSymbol& originalKernel,
                                    const hsa::ExecutableSymbol& instrumentedKernel);

    /**
     * Returns the instrumented kernel's KD given its original un-instrumented version's KD
     * @param originalKernelKD KD of the un-instrumented original kernel
     * @return pointer to the KD of the instrumented kernel, which should be located on the same device
     * @throws std::runtime_error if the originalKernelKD is not found internally
     */
    const hsa::ExecutableSymbol& getInstrumentedKernel(const hsa::ExecutableSymbol& originalKernel) const;

 private:

    CodeObjectManager() {}
    ~CodeObjectManager() {
    }

    /**
     * Iterates over all the frozen HSA executables in the HSA Runtime and registers the ones that belong to the Luthier tool
     */
    void registerLuthierHsaExecutables();

    /**
     * A set of all hsa_executable_t handles that belong to the Luthier tool, containing the instrumentation function
     * and their wrapper kernels
     */
    std::set<hsa::Executable> toolExecutables_{};

    std::unordered_map<const void*, std::unordered_map<hsa::GpuAgent, luthier::InstrumentationFunction>> functions_{};

    std::unordered_map<hsa::ExecutableSymbol, hsa::ExecutableSymbol> instrumentedKernels_{};
};
};// namespace luthier

#endif
