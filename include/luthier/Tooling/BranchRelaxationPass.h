
#ifndef LUTHIER_TOOLING_BRANCH_RELAXATION_PASS_H
#define LUTHIER_TOOLING_BRANCH_RELAXATION_PASS_H

#include <llvm/IR/PassManager.h>
#include <llvm/CodeGen/RegisterScavenging.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/CodeGen/MachineFunction.h>

namespace luthier{

    class BranchRelaxationPass : public PassInfoMixin<BranchRelaxationPass> {
        llvm::MachineFunction *MF = nullptr;
        const llvm::TargetRegisterInfo *TRI = nullptr;
        const llvm::TargetInstrInfo *TII = nullptr;
        const llvm::TargetMachine *TM = nullptr;
        public:
            llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM);
    };
}

#endif 