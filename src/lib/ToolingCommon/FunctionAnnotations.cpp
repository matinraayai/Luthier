
#include "luthier/Tooling/FunctionAnnotations.h"
#include <llvm/IR/Function.h>
#include <llvm/Support/ScopedPrinter.h>

namespace luthier {

void setFunctionEntryPoint(llvm::Function &F, EntryPoint EP) {
  F.addFnAttr(EntryPointAddrAttr, llvm::to_string(EP.getRawAddress()));
}

std::optional<EntryPoint> getFunctionEntryPoint(llvm::Function &F) {
  uint64_t Addr = F.getFnAttributeAsParsedInteger(EntryPointAddrAttr);
  if (Addr == 0) {
    return std::nullopt;
  }
  if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
    return EntryPoint(
        *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(Addr));
  } else {
    return EntryPoint{Addr};
  }
}

} // namespace luthier