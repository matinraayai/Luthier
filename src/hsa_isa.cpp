#include "hsa_isa.hpp"

namespace luthier::hsa {

Isa Isa::fromName(const char *isaName) {
    hsa_isa_t isa;
    const auto &coreApi = luthier::HsaInterceptor::instance().getSavedHsaTables().core;
    LUTHIER_HSA_CHECK(
        coreApi.hsa_isa_from_name_fn(
            isaName,
            &isa));
    return Isa(isa);
}
std::string Isa::getName() const {
    uint32_t isaNameSize;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(this->asHsaType(),
                                                                 HSA_ISA_INFO_NAME_LENGTH,
                                                                 &isaNameSize));
    std::string isaName;
    isaName.resize(isaNameSize);
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_isa_get_info_alt_fn(this->asHsaType(),
                                                                 HSA_ISA_INFO_NAME,
                                                                 isaName.data()));
    return isaName;
}

}// namespace luthier::hsa
