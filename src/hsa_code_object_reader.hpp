#ifndef HSA_CODE_OBJECT_READER_HPP
#define HSA_CODE_OBJECT_READER_HPP
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/ArrayRef.h>
#include "hsa_handle_type.hpp"

namespace luthier {

class CodeObjectManager;

namespace hsa {

class CodeObjectReader : public HandleType<hsa_code_object_reader_t> {
    friend class luthier::CodeObjectManager;

 private:
    static CodeObjectReader createFromMemory(llvm::StringRef elf);

    static CodeObjectReader createFromMemory(llvm::ArrayRef<uint8_t> elf);

    void destroy();

 public:
    explicit CodeObjectReader(hsa_code_object_reader_t reader) : HandleType(reader){};
};

}// namespace hsa
}// namespace luthier

#endif