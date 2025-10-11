
#ifndef LUTHIER_TOOLING_CODE_DISCOVERY_PASS_H
#define LUTHIER_TOOLING_CODE_DISCOVERY_PASS_H
#include <llvm/IR/PassManager.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>

namespace luthier {

class CodeDiscoveryPass : public llvm::PassInfoMixin<CodeDiscoveryPass> {

  using EntryPointType =
      std::variant<const llvm::amdhsa::kernel_descriptor_t *, uint64_t>;

  EntryPointType InitialEntryPoint;

public:
  explicit CodeDiscoveryPass(EntryPointType InitialEntryPoint)
      : InitialEntryPoint(InitialEntryPoint) {};

  llvm::PreservedAnalyses run(llvm::Module &TargetModule,
                              llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif