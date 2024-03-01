#ifndef HSA_TYPE_HPP
#define HSA_TYPE_HPP
#include "hsa_intercept.hpp"

namespace luthier::hsa {
template <typename HT> class Type {
private:
  HT HsaType; //< Should be trivially copyable

protected:
  explicit Type(HT HsaType) : HsaType(HsaType){};

  [[nodiscard]] inline const HsaApiTableContainer &getApiTable() const {
    return HsaInterceptor::instance().getSavedHsaTables();
  }

  [[nodiscard]] inline const hsa_ven_amd_loader_1_03_pfn_t &
  getLoaderTable() const {
    return HsaInterceptor::instance().getHsaVenAmdLoaderTable();
  }

public:
  HT asHsaType() const { return HsaType; }

  Type(const Type &Type) : HsaType(Type.asHsaType()){};

  Type &operator=(const Type &Other) {
    this->HsaType = Other.HsaType;
    return *this;
  }

  Type &operator=(Type &&Other) noexcept {
    this->HsaType = Other.HsaType;
    return *this;
  }
};

} // namespace luthier::hsa

#endif