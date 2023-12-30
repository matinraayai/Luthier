#include "hsa_code_object_reader.hpp"

#include "hsa_intercept.hpp"

luthier::hsa::CodeObjectReader luthier::hsa::CodeObjectReader::createFromMemory(luthier::byte_string_view elf) {
    const auto& coreTable = HsaInterceptor::instance().getSavedHsaTables().core;
    hsa_code_object_reader_t reader;
    LUTHIER_HSA_CHECK(coreTable.hsa_code_object_reader_create_from_memory_fn(elf.data(), elf.size(), &reader));
    return CodeObjectReader{reader};
}
luthier::hsa::CodeObjectReader luthier::hsa::CodeObjectReader::createFromMemory(const luthier::byte_string_t& elf) {
    return createFromMemory(luthier::byte_string_view(elf));
}
void luthier::hsa::CodeObjectReader::destroy() {
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_code_object_reader_destroy_fn(asHsaType()));
}
