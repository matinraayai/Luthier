#ifndef HSA_TYPE_HPP
#define HSA_TYPE_HPP
#include "hsa_intercept.hpp"

namespace luthier::hsa {
template <typename HT> class Type {
private:
  const HT HsaType;
  const HsaApiTableContainer &HsaApiTable; //< This is saved to reduce the
                                           // number of calls to the HSA
                                           // interceptor
  const hsa_ven_amd_loader_1_03_pfn_t &HsaLoaderTable;

protected:
  explicit Type(HT hsaType)
      : HsaApiTable(HsaInterceptor::instance().getSavedHsaTables()),
        HsaLoaderTable(HsaInterceptor::instance().getHsaVenAmdLoaderTable()),
        HsaType(hsaType){};

  [[nodiscard]] inline const HsaApiTableContainer &getApiTable() const {
    return HsaApiTable;
  }

  [[nodiscard]] inline const hsa_ven_amd_loader_1_03_pfn_t &
  getLoaderTable() const {
    return HsaLoaderTable;
  }

public:
  HT asHsaType() const { return HsaType; }
};

} // namespace luthier::hsa

#endif