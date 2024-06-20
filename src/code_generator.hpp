#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "luthier/instr.h"
#include "luthier/types.h"
#include "object_utils.hpp"
#include "singleton.hpp"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include <luthier/pass.h>

#include <llvm/IR/Type.h>

#include <llvm/Pass.h>
#include <queue>

#include "llvm/IR/LegacyPassManager.h"
// #include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <llvm/CodeGen/LivePhysRegs.h>

namespace luthier {

namespace hsa {

class GpuAgent;

class ISA;

} // namespace hsa

class CodeGenerator: public Singleton<CodeGenerator> {
public:

  static llvm::Error
  compileRelocatableToExecutable(const llvm::ArrayRef<uint8_t> &Code,
                                 const hsa::ISA &ISA,
                                 llvm::SmallVectorImpl<uint8_t> &Out);

  llvm::Error
  instrument(std::unique_ptr<llvm::Module> Module,
             std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP,
             const LiftedSymbolInfo &LSO, luthier::InstrumentationTask &ITask);

private:

  static llvm::Expected<std::vector<LiftedSymbolInfo>>
  applyInstrumentation(llvm::Module &Module, llvm::MachineModuleInfo &MMI,
                       const LiftedSymbolInfo &LSO,
                       const InstrumentationTask &ITask);

  static llvm::Expected<std::vector<LiftedSymbolInfo>>
  insertFunctionCalls(llvm::Module &Module, llvm::MachineModuleInfo &MMI,
                      const LiftedSymbolInfo &TargetLSI,
                      const InstrumentationTask::insert_call_tasks &Tasks);
};

} // namespace luthier

namespace llvm {

namespace {

struct LivenessCopy : public MachineFunctionPass {
  static char ID;
  llvm::LivePhysRegs LiveRegs;

  LivenessCopy() : MachineFunctionPass(ID) {}

  explicit LivenessCopy(const llvm::MachineBasicBlock* IPointBlock) : 
      MachineFunctionPass(ID) {
    llvm::computeLiveIns(LiveRegs, *IPointBlock);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (LiveRegs.empty()) {
      llvm::outs() << "No lives to add\n";
    return true;
    }

    /* Frame offsets somewhere around here 
     * Somwhere we'll need to parse inline Asm and turn it into normal instructions
     */

    for (auto &IPointMBB : MF) {
      // if (IPointMBB.getName() == "InstruPoint") {
        llvm::outs() << "Add LiveIns to Block: " 
                     << IPointMBB.getName() << "\n";
        llvm::addLiveIns(IPointMBB, LiveRegs);
      // }
    }
    llvm::outs() << "=====> Liveness Copy finished\n";
    return true;
  }
};
} // namespace anonymous 

char LivenessCopy::ID = 0;
// static llvm::RegisterPass<LivenessCopy> X("getlivenss", "Liveness Copy Pass",
//                                           false, false);

} // namespace llvm

#endif
