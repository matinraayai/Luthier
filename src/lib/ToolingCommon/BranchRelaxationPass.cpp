//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a branch relaxation pass for the code we are instrumenting
//===----------------------------------------------------------------------===//

#include <luthier/Tooling/BranchRelaxationPass.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Target/TargetMachine.h>
#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-branch-relaxation"

namespace luthier{

    llvm::PreservedAnalyses BranchRelaxationPass::run(llvm::MachineFunction &TargetMF, llvm::MachineFunctionAnalysisManager &TargetMFAM) {
        MF = &TargetMF;
        bool MadeChanges;
        LLVM_DEBUG(dbgs() << "***** BranchRelaxation *****\n");

        
        const llvm::TargetSubtargetInfo &ST = TargetMF.getSubtarget();
        if(MadeChanges) return llvm::PreservedAnalyses::none();
        return llvm::PreservedAnalyses::all();
    }

} // namespace luthier