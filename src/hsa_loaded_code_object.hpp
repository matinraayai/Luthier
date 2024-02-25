#ifndef HSA_LOADED_CODE_OBJECT_HPP
#define HSA_LOADED_CODE_OBJECT_HPP
#include "hsa_handle_type.hpp"
#include <llvm/Support/Error.h>

namespace luthier::hsa {
class Executable;

class GpuAgent;

class LoadedCodeObject : public HandleType<hsa_loaded_code_object_t> {
public:
  explicit LoadedCodeObject(hsa_loaded_code_object_t lco);

  [[nodiscard]] llvm::Expected<Executable> getExecutable() const;

  [[nodiscard]] llvm::Expected<GpuAgent> getAgent() const;

  [[nodiscard]] llvm::Expected<hsa_ven_amd_loader_code_object_storage_type_t>
  getStorageType() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>>
  getStorageMemory() const;

  [[nodiscard]] llvm::Expected<int> getStorageFile() const;

  [[nodiscard]] llvm::Expected<long> getLoadDelta() const;

  [[nodiscard]] llvm::Expected<llvm::ArrayRef<uint8_t>> getLoadedMemory() const;

  [[nodiscard]] llvm::Expected<std::string> getUri() const;
};

} // namespace luthier::hsa

#endif