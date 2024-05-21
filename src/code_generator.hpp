#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "luthier/instr.h"
#include "luthier/types.h"
#include "object_utils.hpp"
#include "singleton.hpp"
#include "llvm/CodeGen/TargetPassConfig.h"
#include <luthier/pass.h>

#include <llvm/IR/Type.h>

#include <llvm/Pass.h>
#include <queue>

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

  static llvm::Error convertToVirtual(llvm::Module &Module,
                                      llvm::MachineModuleInfo &MMI,
                                      const LiftedSymbolInfo &LSI);
    
  // AnalysisID TargetPassConfig::addPass(AnalysisID PassID) {
  // llvm::Pass* addPass(llvm::TargetPassConfig *TPC, llvm::AnalysisID PassID) {
  llvm::Pass* getPass(llvm::TargetPassConfig *TPC, llvm::AnalysisID PassID) {
    llvm::IdentifyingPassPtr TargetID = TPC->getPassSubstitution(PassID);
    // llvm::IdentifyingPassPtr FinalPtr = TPC->overridePass(PassID, TargetID);
    // if (!FinalPtr.isValid())
    //   return nullptr;
  
    llvm::Pass *P;
    // if (FinalPtr.isInstance())
    //   P = FinalPtr.getInstance();
    // else {
      // P = llvm::Pass::createPass(FinalPtr.getID());
      
    P = llvm::Pass::createPass(TargetID.getID());
      
    if (!P)
      llvm_unreachable("Pass ID not registered");

    return P;
    // }
    // addPass(P); // Ends the lifetime of P.
  }
};
} // namespace luthier

#endif
