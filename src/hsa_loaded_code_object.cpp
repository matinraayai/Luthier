#include "hsa_loaded_code_object.hpp"

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_intercept.hpp"

namespace luthier::hsa {

LoadedCodeObject::LoadedCodeObject(hsa_loaded_code_object_t lco)
    : HandleType<hsa_loaded_code_object_t>(lco) {}

llvm::Expected<Executable> LoadedCodeObject::getExecutable() const {
  hsa_executable_t exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE, &exec)));

  return Executable(exec);
}

llvm::Expected<GpuAgent> LoadedCodeObject::getAgent() const {
  hsa_agent_t agent;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
          &agent)));
  return GpuAgent(agent);
}

llvm::Expected<hsa_ven_amd_loader_code_object_storage_type_t>
LoadedCodeObject::getStorageType() const {
  hsa_ven_amd_loader_code_object_storage_type_t storageType;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
          &storageType)));
  return storageType;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObject::getStorageMemory() const {
  luthier_address_t storageBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
          &storageBase)));

  uint64_t storageSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
          &storageSize)));

  return llvm::ArrayRef<uint8_t>{reinterpret_cast<uint8_t *>(storageBase),
                                 storageSize};
}

llvm::Expected<int> LoadedCodeObject::getStorageFile() const {
  int fd;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
          &fd)));
  return fd;
}

llvm::Expected<long> LoadedCodeObject::getLoadDelta() const {
  long loadDelta;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA, &loadDelta)));
  return loadDelta;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObject::getLoadedMemory() const {
  luthier_address_t loadBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &loadBase)));

  uint64_t loadSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &loadSize)));

  return llvm::ArrayRef<uint8_t>{reinterpret_cast<uint8_t *>(loadBase),
                                 loadSize};
}

llvm::Expected<std::string> LoadedCodeObject::getUri() const {
  unsigned int uriLength;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH, &uriLength)));

  std::string uri;
  uri.resize(uriLength);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
          uri.data())));

  return uri;
}

} // namespace luthier::hsa
