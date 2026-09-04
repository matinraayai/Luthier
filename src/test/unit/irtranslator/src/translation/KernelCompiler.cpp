#include "KernelCompiler.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>

namespace luthier::test {

/// Create an AMDGPUTargetMachine for the given GPU target string.
static llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
createAMDGPUTargetMachine(llvm::StringRef GpuTarget) {
  std::string Error;
  llvm::Triple TT("amdgcn-amd-amdhsa");
  const llvm::Target *T =
      llvm::TargetRegistry::lookupTarget(TT.str(), Error);
  if (!T)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to find AMDGPU target: " + Error);

  llvm::TargetOptions Opts;
  std::unique_ptr<llvm::TargetMachine> TM(T->createTargetMachine(
      TT.str(), GpuTarget, "", Opts,
      /*RM=*/std::nullopt, /*CM=*/std::nullopt,
      llvm::CodeGenOptLevel::Default));
  if (!TM)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to create TargetMachine");
  return std::move(TM);
}

llvm::Expected<llvm::SmallVector<char, 0>>
KernelCompiler::compileIR(llvm::Module &M, llvm::StringRef GpuTarget) {
  auto TMOrErr = createAMDGPUTargetMachine(GpuTarget);
  if (!TMOrErr)
    return TMOrErr.takeError();
  auto &TM = *TMOrErr;

  M.setDataLayout(TM->createDataLayout());

  llvm::SmallVector<char, 0> ObjBuffer;
  llvm::raw_svector_ostream ObjStream(ObjBuffer);
  llvm::legacy::PassManager PM;

  if (TM->addPassesToEmitFile(PM, ObjStream, nullptr,
                              llvm::CodeGenFileType::ObjectFile))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Target does not support object emission");

  PM.run(M);
  return ObjBuffer;
}

} // namespace luthier::test
