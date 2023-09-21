#ifndef CODE_OBJECT_MANAGER_HPP
#define CODE_OBJECT_MANAGER_HPP
#include "code_object_manipulation.hpp"
#include "luthier_types.hpp"
#include <amd_comgr/amd_comgr.h>
#include <hsa/hsa.h>
#include <memory>
#include <set>
#include <unordered_map>
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
     * Returns the content of the instrumentation function, given its HSA agent and wrapper kernel host ptr
     * @param wrapperKernelHostPtr shadow host pointer of the wrapper kernel used to force compilation of the
     * instrumentation function
     * @param agent the GPU HSA agent where the instrumentation function is loaded on
     * @return the instrumentation function body
     */
    co_manip::code_view_t getInstrumentationFunction(const void *wrapperKernelHostPtr, hsa_agent_t agent) const;

    /**
     * Returns the kernel descriptor of the wrapper kernel associated with an instrumentation function, given its wrapper kernel
     * shadow host pointer and the HSA GPU Agent it is loaded on
     * @param wrapperKernelHostPtr shadow host pointer of the wrapper kernel used to force compilation of the
     * instrumentation function
     * @param agent the HSA GPU Agent the wrapper kernel is loaded on
     * @return a pointer to the kernel descriptor located in host-accessible device memory
     */
    kernel_descriptor_t *getKernelDescriptorOfInstrumentationFunction(const void *wrapperKernelHostPtr, hsa_agent_t agent) const;

    /**
     * Registers an instrumented version of a target application kernel. \class CodeObjectManager keeps track of instrumented kernels
     * via their kernel descriptors located in device memory. Kernel descriptors can then be used to query information about the kernel
     * (e.g. start of the kernel)
     * Both kernel descriptors should be located on the same device
     * @param originalCodeKD device address of the target application kernel's descriptor
     * @param instrumentedCodeKD device address of the instrumented kernel's descriptor
     */
    void registerInstrumentedKernel(const kernel_descriptor_t *originalCodeKD,
                                    const kernel_descriptor_t *instrumentedCodeKD);

    /**
     * Returns the instrumented kernel's KD given its original un-instrumented version's KD
     * @param originalKernelKD KD of the un-instrumented original kernel
     * @return pointer to the KD of the instrumented kernel, which should be located on the same device
     * @throws std::runtime_error if the originalKernelKD is not found internally
     */
    const kernel_descriptor_t *getInstrumentedKernelKD(const kernel_descriptor_t *originalKernelKD);

 private:
    typedef struct {
        luthier::co_manip::code_view_t function;
        kernel_descriptor_t *kd;
    } per_agent_instrumentation_function_entry_t;

    typedef struct {
        std::unordered_map<decltype(hsa_agent_t::handle), per_agent_instrumentation_function_entry_t> agentToExecMap;
        const std::string globalFunctionName;
        const std::string deviceFunctionName;
    } instrumentation_function_info_t;

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
    std::set<decltype(hsa_executable_t::handle)> executables_{};

    std::unordered_map<const void *, instrumentation_function_info_t> functions_{};

    std::unordered_map<const kernel_descriptor_t *, const kernel_descriptor_t *> instrumentedKernels_{};
};
};// namespace luthier

#endif
