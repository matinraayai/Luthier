#include "hsa_loaded_code_object.hpp"

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_intercept.hpp"
#include "hsa_platform.hpp"
#include "object_utils.hpp"

namespace luthier::hsa {

LoadedCodeObject::LoadedCodeObject(hsa_loaded_code_object_t LCO)
    : HandleType<hsa_loaded_code_object_t>(LCO) {}

llvm::Expected<Executable> LoadedCodeObject::getExecutable() const {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE, &Exec)));

  return Executable(Exec);
}

llvm::Expected<GpuAgent> LoadedCodeObject::getAgent() const {
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
          &Agent)));
  return GpuAgent(Agent);
}

llvm::Expected<hsa_ven_amd_loader_code_object_storage_type_t>
LoadedCodeObject::getStorageType() const {
  hsa_ven_amd_loader_code_object_storage_type_t StorageType;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
          &StorageType)));
  return StorageType;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObject::getStorageMemory() const {
  luthier::address_t StorageBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
          &StorageBase)));

  uint64_t StorageSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
          &StorageSize)));

  return llvm::ArrayRef<uint8_t>{reinterpret_cast<uint8_t *>(StorageBase),
                                 StorageSize};
}

llvm::Expected<int> LoadedCodeObject::getStorageFile() const {
  int FD;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
          &FD)));
  return FD;
}

llvm::Expected<long> LoadedCodeObject::getLoadDelta() const {
  long LoadDelta;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA, &LoadDelta)));
  return LoadDelta;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObject::getLoadedMemory() const {
  luthier::address_t LoadBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &LoadBase)));

  uint64_t LoadSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &LoadSize)));

  return llvm::ArrayRef<uint8_t>{reinterpret_cast<uint8_t *>(LoadBase),
                                 LoadSize};
}

llvm::Expected<std::string> LoadedCodeObject::getUri() const {
  unsigned int UriLength;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH, &UriLength)));

  std::string URI;
  URI.resize(UriLength);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
          URI.data())));

  return URI;
}

llvm::Expected<hsa_ven_amd_loader_loaded_code_object_kind_t>
LoadedCodeObject::getKind() {
  hsa_ven_amd_loader_loaded_code_object_kind_t Kind;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND,
          &Kind)));
  return Kind;
}
llvm::Expected<llvm::object::ELF64LEObjectFile &>
LoadedCodeObject::getStorageELF() const {
  std::lock_guard Lock(getMutex());
  if (!StorageELFOfLCOs.contains(hsaHandle()))
    LUTHIER_RETURN_ON_ERROR(cache());
  return *StorageELFOfLCOs.at(hsaHandle());
}

llvm::Expected<ISA> LoadedCodeObject::getISA() const {
  std::lock_guard Lock(getMutex());
  if (!ISAOfLCOs.contains(hsaHandle())) {
    LUTHIER_RETURN_ON_ERROR(cache());
  }
  return ISA(ISAOfLCOs.at(hsaHandle()));
}

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               std::unique_ptr<llvm::object::ELF64LEObjectFile>>
    LoadedCodeObject::StorageELFOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle), hsa_isa_t>
    LoadedCodeObject::ISAOfLCOs;

llvm::Error LoadedCodeObject::cache() const {
  std::lock_guard Lock(getMutex());
  // Cache the Storage ELF
  auto StorageMemory = this->getStorageMemory();
  LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
  auto StorageELF = getAMDGCNObjectFile(*StorageMemory);
  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());
  auto &CachedELF =
      *(StorageELFOfLCOs.insert({this->hsaHandle(), std::move(*StorageELF)})
            .first->second);
  // Cache the ISA of the ELF
  llvm::Triple TT = CachedELF.makeTriple();
  std::optional<llvm::StringRef> CPU = CachedELF.tryGetCPUName();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CPU.has_value()));
  llvm::SubtargetFeatures Features;
  LUTHIER_RETURN_ON_ERROR(CachedELF.getFeatures().moveInto(Features));
  auto ISA = hsa::ISA::fromLLVM(TT, *CPU, Features);
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());
  ISAOfLCOs.insert({this->hsaHandle(), ISA->asHsaType()});
  return llvm::Error::success();
}
llvm::Error LoadedCodeObject::invalidate() const {
  std::lock_guard Lock(getMutex());
  StorageELFOfLCOs.erase(this->hsaHandle());
  ISAOfLCOs.erase(this->hsaHandle());
  return llvm::Error::success();
}

} // namespace luthier::hsa
