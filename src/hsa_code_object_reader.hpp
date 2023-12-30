#ifndef HSA_CODE_OBJECT_READER_HPP
#define HSA_CODE_OBJECT_READER_HPP
#include "code_view.hpp"
#include "hsa_handle_type.hpp"

namespace luthier {

class CodeObjectManager;

namespace hsa {

class CodeObjectReader : public HandleType<hsa_code_object_reader_t> {
    friend class luthier::CodeObjectManager;

 private:
    static CodeObjectReader createFromMemory(luthier::byte_string_view elf);

    static CodeObjectReader createFromMemory(const luthier::byte_string_t& elf);

    void destroy();

 public:
    explicit CodeObjectReader(hsa_code_object_reader_t reader) : HandleType(reader){};
};

}// namespace hsa
}// namespace luthier

#endif