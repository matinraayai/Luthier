#include "hsa/hsa_agent.hpp"

#include "hsa/hsa_isa.hpp"

namespace luthier::hsa {

llvm::Error GpuAgent::getIsa(llvm::SmallVectorImpl<ISA> &isaList) const {
  auto iterator = [](hsa_isa_t isa, void *data) {
    auto supportedIsaList =
        reinterpret_cast<llvm::SmallVectorImpl<ISA> *>(data);
    supportedIsaList->emplace_back(isa);
    return HSA_STATUS_SUCCESS;
  };

  return LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_agent_iterate_isas_fn(
      this->asHsaType(), iterator, &isaList));
}
llvm::Expected<hsa::ISA> GpuAgent::getIsa() const {
  hsa_isa_t out;
  auto iterator = [](hsa_isa_t isa, void *data) {
    *reinterpret_cast<hsa_isa_t *>(data) = isa;
    return HSA_STATUS_INFO_BREAK;
  };
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_ERROR_CHECK(getApiTable().core.hsa_agent_iterate_isas_fn(
                                  this->asHsaType(), iterator, &out),
                              HSA_STATUS_INFO_BREAK));
  return hsa::ISA(out);
}

} // namespace luthier::hsa
