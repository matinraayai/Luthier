#ifndef HSA_AGENT_HPP
#define HSA_AGENT_HPP
#include "hsa_handle_type.hpp"
#include "hsa_intercept.hpp"
#include <hsa/hsa.h>
#include <string>

//#define AGENT_COMGR_META_ACCESSOR(metaName, metaType, typeConvertor) \
//    metaType get##metaName##fromComgrMeta() {                                 \
//        std::string key{#metaName};\
//        if (!amdComgrMetaDataMap_.contains(key)) {             \
//            amd_comgr_metadata_node_t valueMetaData;           \
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_metadata_lookup(metaDataRootNode_, key.c_str(), &valueMetaData)); \
//            std::string value;                                 \
//            size_t size;                                       \
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaData, &size, nullptr));      \
//            value.resize(size);                                \
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaData, &size, value.data())); \
//            amdComgrMetaDataMap_.insert({key, typeConvertor(value.c_str())});                                     \
//        }                                                      \
//        return std::any_cast<metaType>(amdComgrMetaDataMap_.at(key));                                 \
//    }
//
//#define AGENT_HSA_META_ACCESSOR(metaName, metaReturnType, hsaInfoType) \
//    metaReturnType getAgent##metaName##fromHsa() { \
//        if (!hsaMetaDataMap_.contains((hsa_agent_info_t) hsaInfoType)) {                  \
//            const auto &coreApi = luthier::HsaInterceptor::instance().getSavedHsaTables().core; \
//            metaReturnType out;                                        \
//            LUTHIER_HSA_CHECK(coreApi.hsa_agent_get_info_fn(agent_, (hsa_agent_info_t) hsaInfoType, &out)); \
//            hsaMetaDataMap_.insert({(hsa_agent_info_t) hsaInfoType, out});\
//        }                                                              \
//        return std::any_cast<metaReturnType>(hsaMetaDataMap_.at((hsa_agent_info_t) hsaInfoType));         \
//    }
//
//namespace luthier {
//
//class AgentMetaData {
// private:
//    std::unordered_map<std::string, std::any> amdComgrMetaDataMap_;
//    std::unordered_map<hsa_agent_info_t, std::any> hsaMetaDataMap_;
//    std::string isaName_;
//    hsa_agent_t agent_;
//    amd_comgr_metadata_node_t metaDataRootNode_;
//
//
// public:
//    explicit AgentMetaData(hsa_agent_t agent);
//    AgentMetaData() = delete;
//
//    ~AgentMetaData() {
//        amdComgrMetaDataMap_.clear();
//        hsaMetaDataMap_.clear();
//    }
//
//    std::string getIsaName() const { return isaName_; };
//
//    AGENT_HSA_META_ACCESSOR(DriverNodeId, uint32_t, HSA_AMD_AGENT_INFO_DRIVER_NODE_ID);
//
//    AGENT_COMGR_META_ACCESSOR(AddressableNumSGPRs, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(AddressableNumVGPRs, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(Architecture, std::string, std::string)
//    AGENT_COMGR_META_ACCESSOR(EUsPerCU, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(LDSBankCount, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(LocalMemorySize, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(MaxFlatWorkGroupSize, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(MaxWavesPerCU, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(TotalNumSGPRs, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(TotalNumVGPRs, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(VGPRAllocGranule, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(SGPRAllocGranule, int, std::atoi)
//    AGENT_COMGR_META_ACCESSOR(Name, std::string, std::string)
//    AGENT_COMGR_META_ACCESSOR(Os, std::string, std::string)
//    AGENT_COMGR_META_ACCESSOR(Processor, std::string, std::string)
//    AGENT_COMGR_META_ACCESSOR(TrapHandlerEnabled, bool, bool)
//    AGENT_COMGR_META_ACCESSOR(Vendor, std::string, std::string)
//    AGENT_COMGR_META_ACCESSOR(Version, std::string, std::string)
//};
////typedef struct luthier_hsa_agent_info_entry_s {
////    // metadata information obtained from AMD Comgr library
////    struct  {
////        bool sramecc;
////        bool xnack;
////        unsigned int lDSBankCount; /
////        unsigned int localMemorySize; /
////        unsigned int maxFlatWorkGroupSize; /
////        unsigned int maxWavesPerCU; /
////        std::string name; /
////        std::string os; /
////        std::string processor; /
////        unsigned int sGPRAllocGranule; /
////        unsigned int vGPRAllocGranule; /
////        unsigned int totalNumSGPRs; /
////        unsigned int totalNumVGPRs; /
////        bool trapHandlerEnabled; /
////        std::string vendor;
////        std::string version;
////    } amd_comgr;
////    struct  {
////        std::string isaName;
////    } hsa;
////} luthier_hsa_agent_info_entry_t;
//#undef AGENT_COMGR_META_ACCESSOR
//#undef AGENT_HSA_META_ACCESSOR

namespace luthier::hsa {

//#define AGENT_HSA_META_ACCESSOR(metaName, metaReturnType, hsaInfoType) \
//    metaReturnType get##metaName() { \
//        if (!hsaMetaDataMap_.contains((hsa_agent_info_t) hsaInfoType)) {                  \
//            const auto &coreApi = luthier::HsaInterceptor::instance().getSavedHsaTables().core; \
//            metaReturnType out;                                        \
//            LUTHIER_HSA_CHECK(coreApi.hsa_agent_get_info_fn(agent_, (hsa_agent_info_t) hsaInfoType, &out)); \
//            hsaMetaDataMap_.insert({(hsa_agent_info_t) hsaInfoType, out});\
//        }                                                              \
//        return std::any_cast<metaReturnType>(hsaMetaDataMap_.at((hsa_agent_info_t) hsaInfoType));         \
//    }

class Isa;

class GpuAgent : public HandleType<hsa_agent_t> {
 public:
    explicit GpuAgent(hsa_agent_t agent) : HandleType<hsa_agent_t>(agent){};

    std::vector<Isa> getIsa() const;
};

}// namespace luthier::hsa

namespace std {

template<>
struct hash<luthier::hsa::GpuAgent> {
    size_t operator()(const luthier::hsa::GpuAgent &obj) const {
        return hash<unsigned long>()(obj.hsaHandle());
    }
};

template<>
struct less<luthier::hsa::GpuAgent> {
    bool operator()(const luthier::hsa::GpuAgent &lhs, const luthier::hsa::GpuAgent &rhs) const {
        return lhs.hsaHandle() < rhs.hsaHandle();
    }
};

template<>
struct equal_to<luthier::hsa::GpuAgent> {
    bool operator()(const luthier::hsa::GpuAgent &lhs, const luthier::hsa::GpuAgent &rhs) const {
        return lhs.hsaHandle() == rhs.hsaHandle();
    }
};

}// namespace std

//#undef AGENT_HSA_META_ACCESSOR

#endif