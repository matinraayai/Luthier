#include "hsa_loaded_code_object.hpp"
#include "hsa_executable.hpp"
#include "hsa_intercept.hpp"

namespace luthier::hsa {

LoadedCodeObject::LoadedCodeObject(hsa_loaded_code_object_t lco) : HandleType<hsa_loaded_code_object_t>(lco) {}

Executable LoadedCodeObject::getExecutable() const {
    hsa_executable_t exec;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
        this->asHsaType(),
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE,
        &exec));
    return Executable(exec);
}

GpuAgent LoadedCodeObject::getAgent() const {
    hsa_agent_t agent;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
        this->asHsaType(),
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
        &agent));
    return GpuAgent(agent);
}

hsa_ven_amd_loader_code_object_storage_type_t LoadedCodeObject::getStorageType() const {
    hsa_ven_amd_loader_code_object_storage_type_t storageType;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
        this->asHsaType(),
        HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
        &storageType));
    return storageType;
}

luthier::co_manip::code_view_t LoadedCodeObject::getStorageMemory() const {
    luthier_address_t storageBase;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                                      HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                                                                                      &storageBase));

    uint64_t storageSize;
    getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                                                                    &storageSize);

    return {reinterpret_cast<std::byte *>(storageBase), storageSize};
}

int LoadedCodeObject::getStorageFile() const {
    int fd;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                                      HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
                                                                                      &fd));
    return fd;
}

long LoadedCodeObject::getLoadDelta() const {
    long loadDelta;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                                      HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
                                                                                      &loadDelta));
    return loadDelta;
}

luthier::co_manip::code_view_t LoadedCodeObject::getLoadedMemory() const {
    luthier_address_t loadBase;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                                      HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                                                                      &loadBase));

    uint64_t loadSize;
    getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                                                    &loadSize);

    return {reinterpret_cast<std::byte *>(loadBase), loadSize};
}

std::string LoadedCodeObject::getUri() const {
    unsigned int uriLength;
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                                      HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
                                                                                      &uriLength));

    std::string uri;
    uri.resize(uriLength);
    getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(this->asHsaType(),
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
                                                                    uri.data());

    return uri;
}

}// namespace luthier::hsa
