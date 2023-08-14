#ifndef CODE_OBJECT_MANAGER_H
#define CODE_OBJECT_MANAGER_H
#include "luthier_types.hpp"
#include <amd_comgr/amd_comgr.h>
#include <hsa/hsa.h>
#include <map>
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

    void registerFatBinary(const void *data);

    void registerFunction(const void *fbWrapper,
                          const char *funcName,
                          const void *hostFunction,
                          const char *deviceName);

    std::string getCodeObjectOfInstrumentationFunction(const void *function, hsa_agent_t agent);

    void registerKD(luthier_address_t originalCode, luthier_address_t instrumentedCode);

    luthier_address_t getInstrumentedFunctionOfKD(const luthier_address_t kd) {
        std::cout << "Is in instrumented kernels? " << instrumentedKernels_.contains(kd) << std::endl;
        std::cout << "Instrumented kernel address: " << std::hex << instrumentedKernels_[kd] << std::dec << std::endl;
        return instrumentedKernels_[kd];
    }

 private:
    typedef struct {
        std::unordered_map<decltype(hsa_agent_t::handle), hsa_executable_t> agentToExecMap;
        const std::string name;
        const std::string deviceName;
        const void *parentFatBinary;
    } function_info_t;

    CodeObjectManager() {}
    ~CodeObjectManager() {
        for (auto it: fatBinaries_)
            amd_comgr_release_data(it.second);
    }

    std::unordered_map<const void*, function_info_t> functions_{};

    std::unordered_map<const void *, amd_comgr_data_t> fatBinaries_{};

    std::unordered_map<luthier_address_t, luthier_address_t> instrumentedKernels_;
};
};// namespace luthier

#endif
