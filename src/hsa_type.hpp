#ifndef HSA_TYPE_HPP
#define HSA_TYPE_HPP
#include "hsa_intercept.hpp"

namespace luthier::hsa {
template<typename HT>
class Type {
 private:
    const HT hsaType_;
    const HsaApiTableContainer &hsaApiTable_;//< This is saved to reduce the number of calls to the HSA interceptor
 protected:
    explicit Type(HT hsaType) : hsaApiTable_(HsaInterceptor::instance().getSavedHsaTables()), hsaType_(hsaType){};

    [[nodiscard]] const HsaApiTableContainer &getApiTable() const { return hsaApiTable_; }

 public:
    HT asHsaType() const { return hsaType_; }
};

}// namespace luthier::hsa

#endif