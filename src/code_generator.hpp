#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "SIRegisterInfo.h"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "luthier/instr.h"
#include "luthier/types.h"
#include "object_utils.hpp"
#include "singleton.hpp"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "SIFrameLowering.h"
#include <luthier/pass.h>

#include <llvm/IR/Type.h>

#include <llvm/Pass.h>
#include <queue>

#include "llvm/IR/LegacyPassManager.h"
// #include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <llvm/CodeGen/LivePhysRegs.h>
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"

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
    LiveRegs.addReg(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3);
    for (auto &MBB : MF) {
      llvm::outs() << "Add LiveIns to Block: " << MBB.getName() << "\n";
      llvm::addLiveIns(MBB, LiveRegs);
      MBB.sortUniqueLiveIns();
    }
    return true;
  }
};

struct StackFrameOffset : public MachineFunctionPass {
  static char ID;
  // const MachineFunction *RefMF;

  StackFrameOffset() : MachineFunctionPass(ID) {}
  // explicit StackFrameOffset(const MachineFunction* RMF) : 
  //     MachineFunctionPass(ID) {
  //   RefMF = RMF;
  // }

  bool runOnMachineFunction(MachineFunction &MF) override {
    for (auto &MBB : MF) {
      // llvm::SIFrameLowering::emitPrologue(MF, MBB);
      // llvm::SIFrameLowering::emitEpilogue(MF, MBB);
      // if (!RefMF) RefMF = &MF;
      auto &MFI = MF.getFrameInfo();
      // auto &MFI = RefMF->getFrameInfo();
      
      llvm::outs() << "machine function " << MF.getName() << "\n"
                   << "\tStack Size of: " << MFI.getStackSize() << "\n"
      // llvm::outs() << "machine function " << RefMF->getName() << " contains "
                   << "\tContains " << MFI.getNumObjects() << " Stack Objects\n";
      
      for (int SOIdx = 0; SOIdx < MFI.getNumObjects(); ++SOIdx) {
        llvm::outs() << " - Stack Object Num "      << SOIdx << "\n"
                     << "   Stack ID:             " << MFI.getStackID(SOIdx)      << "\n"
                     << "   Stack object Size:    " << MFI.getObjectSize(SOIdx)   << "\n";
        // Add to Stack Frame object offset
        // void setObjectOffset(int ObjectIdx, int64_t SPOffset) {
        auto SOoffset = MFI.getObjectOffset(SOIdx);
        // SOoffset += 100; // value to add: amount of stack the original app is using
        llvm::outs() << "   Stack Pointer Offset: " << SOoffset << "\n";
      }
    }
    return false;
  }
};
} // namespace anonymous 

char LivenessCopy::ID = 0;
char StackFrameOffset::ID = 0;
// static llvm::RegisterPass<LivenessCopy> X("getlivenss", "Liveness Copy Pass",
//                                           false, false);

} // namespace llvm

#endif
