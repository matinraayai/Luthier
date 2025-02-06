#include "common/Lift.hpp"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/Error.h>

namespace luthier {

llvm::Error disassemble(llvm::ArrayRef<uint8_t> Code,
                        const llvm::MCDisassembler &DisAsm,
                        const llvm::MCAsmInfo &MAI,
                        llvm::SmallVectorImpl<llvm::MCInst> &Insts,
                        llvm::SmallVectorImpl<uint64_t> *Offsets) {
  size_t MaxReadSize = MAI.getMaxInstLength();
  size_t Idx = 0;
  uint64_t CurrentOffset = 0;

  while (Idx < Code.size()) {
    size_t ReadSize =
        (Idx + MaxReadSize) < Code.size() ? MaxReadSize : Code.size() - Idx;
    size_t InstSize{};
    llvm::MCInst Inst;
    auto ReadBytes =
        arrayRefFromStringRef(toStringRef(Code).substr(Idx, ReadSize));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        DisAsm.getInstruction(Inst, InstSize, ReadBytes, CurrentOffset,
                              llvm::nulls()) == llvm::MCDisassembler::Success,
        "Failed to disassemble instruction at offset {0:x}", CurrentOffset));
    if (Offsets)
      Offsets->push_back(CurrentOffset);
    Idx += InstSize;
    CurrentOffset += InstSize;
    Insts.push_back(Inst);
  }
  return llvm::Error::success();
}

} // namespace luthier