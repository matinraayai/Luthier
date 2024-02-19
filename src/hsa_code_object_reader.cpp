#include "hsa_code_object_reader.hpp"

#include <llvm/ADT/StringExtras.h>

#include "hsa_intercept.hpp"

luthier::hsa::CodeObjectReader luthier::hsa::CodeObjectReader::createFromMemory(llvm::StringRef elf) {
    const auto& coreTable = HsaInterceptor::instance().getSavedHsaTables().core;
    hsa_code_object_reader_t reader;
    LUTHIER_HSA_CHECK(coreTable.hsa_code_object_reader_create_from_memory_fn(elf.data(), elf.size(), &reader));
    return CodeObjectReader{reader};
}
luthier::hsa::CodeObjectReader luthier::hsa::CodeObjectReader::createFromMemory(llvm::ArrayRef<uint8_t> elf) {
    return createFromMemory(llvm::toStringRef(elf));
}

void luthier::hsa::CodeObjectReader::destroy() {
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_code_object_reader_destroy_fn(asHsaType()));
}
