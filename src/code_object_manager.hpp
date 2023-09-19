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
 * \brief A singleton object that keeps track of instrumentation functions and instrumented kernels
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
     * The wrapper kernel ISA is not used to keep track of instrumentation functions. Their host shadow pointer and
     * their names are used instead.
     * Their kernel descriptors are used to keep track of the instrumentation function register requirements
     * \param instrumentationFunctionInfo a list of tuples, the first element being the host shadow pointer of the wrapper kernels,
     * the second being the name of the wrapper kernel
     */
    void registerHipWrapperKernelsOfInstrumentationFunctions(const std::vector<std::tuple<const void *, const char *>>
                                                                 &instrumentationFunctionInfo);

    co_manip::code_object_region_t getCodeObjectOfInstrumentationFunction(const void *function, hsa_agent_t agent) const;

    kernel_descriptor_t *getKernelDescriptorOfInstrumentationFunction(const void *function, hsa_agent_t agent) const;

    void registerInstrumentedKernel(const kernel_descriptor_t *originalCode,
                                      const kernel_descriptor_t *instrumentedCode);

    const kernel_descriptor_t *getInstrumentedFunctionOfKD(const kernel_descriptor_t *kd) {
        std::cout << "Is in instrumented kernels? " << instrumentedKernels_.contains(kd) << std::endl;
        std::cout << "Instrumented kernel address: " << std::hex << instrumentedKernels_[kd] << std::dec << std::endl;
        return instrumentedKernels_[kd];
    }

 private:
    typedef struct function_agent_entry_s {
        luthier::co_manip::code_object_region_t function{};
        kernel_descriptor_t *kd{};
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
     * Iterates over all the frozen HSA executables and registers the ones that belong to the Luthier tool
     */
    void registerLuthierHsaExecutables();

    std::set<decltype(hsa_executable_t::handle)> executables_{};

    std::unordered_map<const void *, instrumentation_function_info_t> functions_{};

    std::unordered_map<const kernel_descriptor_t *, const kernel_descriptor_t *> instrumentedKernels_{};
};
};// namespace luthier

#endif
