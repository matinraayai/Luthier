#ifndef CODE_OBJECT_MANAGER_HPP
#define CODE_OBJECT_MANAGER_HPP
#include "luthier_types.hpp"
#include "code_object_manipulation.hpp"
#include <vector>
#include <amd_comgr/amd_comgr.h>
#include <hsa/hsa.h>
#include <set>
#include <memory>
#include <unordered_map>

namespace luthier {
class CodeObjectManager {
 public:
    CodeObjectManager(const CodeObjectManager &) = delete;
    CodeObjectManager &operator=(const CodeObjectManager &) = delete;

    static inline CodeObjectManager &Instance() {
        static CodeObjectManager instance;
        return instance;
    }



    void registerFunctions(const std::vector<std::tuple<const void *, const char *>>& instrumentationFunctionInfo);

    co_manip::code_object_region_t getCodeObjectOfInstrumentationFunction(const void *function, hsa_agent_t agent) const;

    kernel_descriptor_t* getKernelDescriptorOfInstrumentationFunction(const void* function, hsa_agent_t agent) const;

    void registerKD(luthier_address_t originalCode, luthier_address_t instrumentedCode);

    luthier_address_t getInstrumentedFunctionOfKD(const luthier_address_t kd) {
        std::cout << "Is in instrumented kernels? " << instrumentedKernels_.contains(kd) << std::endl;
        std::cout << "Instrumented kernel address: " << std::hex << instrumentedKernels_[kd] << std::dec << std::endl;
        return instrumentedKernels_[kd];
    }

 private:
    typedef struct function_agent_entry_s {
        luthier::co_manip::code_object_region_t function{};
        kernel_descriptor_t* kd{};
    } per_agent_instrumentation_function_entry_t;

    typedef struct {
        std::unordered_map<decltype(hsa_agent_t::handle), per_agent_instrumentation_function_entry_t> agentToExecMap;
        const std::string globalFunctionName;
        const std::string deviceFunctionName;
    } instrumentation_function_info_t;

    CodeObjectManager() {}
    ~CodeObjectManager() {
    }

    void registerLuthierExecutables();


    std::set<decltype(hsa_executable_t::handle)> executables_{};

    std::unordered_map<const void*, instrumentation_function_info_t> functions_{};

    std::unordered_map<luthier_address_t, luthier_address_t> instrumentedKernels_;
};
};// namespace luthier

#endif
