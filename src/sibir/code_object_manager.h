#ifndef CODE_OBJECT_MANAGER_H
#define CODE_OBJECT_MANAGER_H
#include <unordered_map>
#include <amd_comgr.h>
#include <hsa/hsa.h>

class SibirCodeObjectManager {
 public:

    SibirCodeObjectManager(const SibirCodeObjectManager &) = delete;
    SibirCodeObjectManager & operator=(const SibirCodeObjectManager &) = delete;

    static inline SibirCodeObjectManager & Instance() {
        static SibirCodeObjectManager instance;
        return instance;
    }

    void registerFatBinary(const void* data);

    void registerFunction(const void* fatBinary,
                          const char* funcName,
                          const void* hostFunction,
                          const char* deviceName);

    hsa_executable_t getInstrumentationFunction(const char* functionName);

 private:
    typedef struct {
        const char* name;
        const void* hostFunction;
        const char* deviceName;
        const void* parentFatBinary;
    } function_info_t;

    SibirCodeObjectManager() {}
    ~SibirCodeObjectManager() {
        for (auto it: fatBinaries_)
            amd_comgr_release_data(it.second);
    }

    std::unordered_map<const void*, amd_comgr_data_t> fatBinaries_{};
    std::unordered_map<const char*, function_info_t> functions_{};

//    //Populated during __hipRegisterVars
//    std::unordered_map<const void*, Var*> vars_;
//    //Populated during __hipRegisterManagedVar
//    std::vector<Var*> managedVars_;
//    std::unordered_map<int, bool> managedVarsDevicePtrInitalized_;

};

#endif
