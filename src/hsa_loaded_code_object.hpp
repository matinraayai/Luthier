#ifndef HSA_LOADED_CODE_OBJECT_HPP
#define HSA_LOADED_CODE_OBJECT_HPP
#include "hsa_handle_type.hpp"

namespace luthier::hsa {
class Executable;

class GpuAgent;

class LoadedCodeObject : public HandleType<hsa_loaded_code_object_t> {
 public:
    explicit LoadedCodeObject(hsa_loaded_code_object_t lco);

    [[nodiscard]] Executable getExecutable() const;

    [[nodiscard]] GpuAgent getAgent() const;

    [[nodiscard]] hsa_ven_amd_loader_code_object_storage_type_t getStorageType() const;

    [[nodiscard]] llvm::ArrayRef<uint8_t> getStorageMemory() const;

    [[nodiscard]] int getStorageFile() const;

    [[nodiscard]] long getLoadDelta() const;

    [[nodiscard]] llvm::ArrayRef<uint8_t> getLoadedMemory() const;

    [[nodiscard]] std::string getUri() const;
};

}// namespace luthier::hsa

#endif