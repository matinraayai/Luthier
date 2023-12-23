#ifndef HSA_LOADED_CODE_OBJECT_HPP
#define HSA_LOADED_CODE_OBJECT_HPP
#include "hsa_primitive.hpp"
#include "code_object_manipulation.hpp"

namespace luthier::hsa {
class Executable;

class GpuAgent;

class LoadedCodeObject : public HandleType<hsa_loaded_code_object_t> {
 private:
    const hsa_ven_amd_loader_1_03_pfn_t& loaderApi_;
 public:
    explicit LoadedCodeObject(hsa_loaded_code_object_t lco);

    [[nodiscard]] Executable getExecutable() const;

    [[nodiscard]] GpuAgent getAgent() const;

    [[nodiscard]] hsa_ven_amd_loader_code_object_storage_type_t getStorageType() const;

    [[nodiscard]] luthier::co_manip::code_view_t getStorageMemory() const;

    [[nodiscard]] int getStorageFile() const;

    [[nodiscard]] long getLoadDelta() const;

    [[nodiscard]] luthier::co_manip::code_view_t getLoadedMemory() const;

    [[nodiscard]] std::string getUri() const;

};

}


#endif