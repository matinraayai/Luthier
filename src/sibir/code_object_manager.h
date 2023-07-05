#ifndef CODE_OBJECT_MANAGER_H
#define CODE_OBJECT_MANAGER_H
#include <unordered_map>
#include <amd_comgr.h>

class SibirCodeObjectManager {
 public:

    SibirCodeObjectManager(const SibirCodeObjectManager &) = delete;
    SibirCodeObjectManager & operator=(const SibirCodeObjectManager &) = delete;

    static inline SibirCodeObjectManager & Instance() {
        static SibirCodeObjectManager instance;
        return instance;
    }

    void setLastFatBinary(const void* data);

    void saveLastFatBinary();

    void registerFunction(const char* funcName,
                          const void* hostFunction,
                          const char* deviceName);

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

    const void* lastFatBinary_{nullptr};

    std::unordered_map<const void*, amd_comgr_data_t> fatBinaries_{};
    std::unordered_map<const char*, function_info_t> functions_{};

//    //Populated during __hipRegisterVars
//    std::unordered_map<const void*, Var*> vars_;
//    //Populated during __hipRegisterManagedVar
//    std::vector<Var*> managedVars_;
//    std::unordered_map<int, bool> managedVarsDevicePtrInitalized_;

};

#endif
