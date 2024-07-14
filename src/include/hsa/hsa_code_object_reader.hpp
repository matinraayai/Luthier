#ifndef HSA_CODE_OBJECT_READER_HPP
#define HSA_CODE_OBJECT_READER_HPP
#include "hsa/hsa_handle_type.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

// I made createFromMemory public - mark this down somewhere

namespace luthier {

class ToolExecutableManager;

namespace hsa {

class CodeObjectReader : public HandleType<hsa_code_object_reader_t> {
  friend class luthier::ToolExecutableManager;

// private:
public:
  static llvm::Expected<CodeObjectReader> createFromMemory(llvm::StringRef elf);

  static llvm::Expected<CodeObjectReader>
  createFromMemory(llvm::ArrayRef<uint8_t> elf);

  llvm::Error destroy();

  explicit CodeObjectReader(hsa_code_object_reader_t reader)
      : HandleType(reader){};
};

} // namespace hsa
} // namespace luthier

#endif
