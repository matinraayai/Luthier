#include "hsa_cache.hpp"
#include "error.h"
#include "hsa_type.hpp"

std::string luthier::hsa::Cache::getName() {
    uint32_t nameLength;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_cache_get_info_fn(asHsaType(), HSA_CACHE_INFO_NAME_LENGTH, &nameLength));
    std::string out;
    out.resize(nameLength);
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_cache_get_info_fn(asHsaType(), HSA_CACHE_INFO_NAME, out.data()));
    return out;
}
uint8_t luthier::hsa::Cache::getLevel() {
    uint8_t level;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_cache_get_info_fn(asHsaType(), HSA_CACHE_INFO_LEVEL, &level));
    return level;
}
unsigned int luthier::hsa::Cache::getSize() {
    unsigned int size;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_cache_get_info_fn(asHsaType(), HSA_CACHE_INFO_LEVEL, &size));
    return size;
}
