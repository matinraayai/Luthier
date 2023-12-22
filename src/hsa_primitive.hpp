#ifndef HSA_PRIMITIVE_HPP
#define HSA_PRIMITIVE_HPP
#include "hsa_intercept.hpp"

namespace luthier::hsa {

template<typename HT>
class HsaPrimitive {
 private:
    const HT primitive_;
    const HsaApiTableContainer &hsaApiTable_;//< This is saved to reduce the number of calls to the HSA interceptor
 protected:
    explicit HsaPrimitive(HT primitive) : hsaApiTable_(HsaInterceptor::instance().getSavedHsaTables()), primitive_(primitive) {};

    const HsaApiTableContainer &getApiTable() { return hsaApiTable_;}
 public:
    HT getPrimitive() { return primitive_;};

};

}

#endif