#include "hsa_code_object_reader.hpp"

#include <llvm/ADT/StringExtras.h>

#include "error.hpp"
#include "hsa_intercept.hpp"

namespace luthier::hsa {

llvm::Expected<CodeObjectReader>
CodeObjectReader::createFromMemory(llvm::StringRef elf) {
  const auto &coreTable = HsaInterceptor::instance().getSavedHsaTables().core;
  hsa_code_object_reader_t reader;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      coreTable.hsa_code_object_reader_create_from_memory_fn(
          elf.data(), elf.size(), &reader)));
  return CodeObjectReader{reader};
}
llvm::Expected<CodeObjectReader>
CodeObjectReader::createFromMemory(llvm::ArrayRef<uint8_t> elf) {
  return createFromMemory(llvm::toStringRef(elf));
}

llvm::Error CodeObjectReader::destroy() {
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_code_object_reader_destroy_fn(asHsaType()));
}

} // namespace luthier::hsa
