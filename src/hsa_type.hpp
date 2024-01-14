#ifndef HSA_TYPE_HPP
#define HSA_TYPE_HPP
#include "hsa_intercept.hpp"

namespace luthier::hsa {
template<typename HT>
class Type {
 private:
    const HT hsaType_;
    const HsaApiTableContainer &hsaApiTable_;//< This is saved to reduce the number of calls to the HSA interceptor
    const hsa_ven_amd_loader_1_03_pfn_t &hsaLoaderTable_;

 protected:
    explicit Type(HT hsaType)
        : hsaApiTable_(HsaInterceptor::instance().getSavedHsaTables()),
          hsaLoaderTable_(HsaInterceptor::instance().getHsaVenAmdLoaderTable()),
          hsaType_(hsaType){};

    [[nodiscard]] inline const HsaApiTableContainer &getApiTable() const { return hsaApiTable_; }

    [[nodiscard]] inline const hsa_ven_amd_loader_1_03_pfn_t &getLoaderTable() const { return hsaLoaderTable_; }

 public:
    HT asHsaType() const { return hsaType_; }
};

}// namespace luthier::hsa

#endif