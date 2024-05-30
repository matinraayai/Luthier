#include "cloning.hpp"

#include <llvm/Analysis/CallGraph.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace luthier {

llvm::Function *
cloneFunctionIntoModule(llvm::ArrayRef<llvm::Function *> OldFuncs,
                        llvm::Module &NewModule) {
  // Walk over the function's instructions, and identify all global objects
  // it uses from the parent module
  llvm::Function *CurrentF{nullptr};
  llvm::DenseSet<llvm::Function *> VisitedFunctions;
  llvm::DenseSet<llvm::GlobalVariable *> VisitedGlobalVariables;
  llvm::DenseSet<llvm::GlobalAlias *> VisitedGlobalAliases;
  llvm::DenseSet<llvm::GlobalIFunc *> VisitedGlobalIFunc;
  llvm::DenseSet<llvm::Function *> FunctionsToBeVisited{OldFuncs.begin(),
                                                        OldFuncs.end()};
  while (!FunctionsToBeVisited.empty()) {
    CurrentF = *FunctionsToBeVisited.begin();
    llvm::outs() << "Dumping function " << CurrentF->getName() << "\n";
    for (const auto &BB : *CurrentF) {
      for (const auto &I : BB) {
        llvm::outs() << "Dumping instruction\n";
        I.dump();
        for (const auto &Op : I.operands()) {
          if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(&Op)) {
            VisitedGlobalVariables.insert(GV);
          }
          if (auto *Func = llvm::dyn_cast<llvm::Function>(&Op)) {
            if (Func != CurrentF && !VisitedFunctions.contains(Func)) {
              FunctionsToBeVisited.insert(Func);
            }
          }
          if (auto *GA = llvm::dyn_cast<llvm::GlobalAlias>(&Op)) {
            VisitedGlobalAliases.insert(GA);
          }
          if (auto *IFunc = llvm::dyn_cast<llvm::GlobalIFunc>(&Op)) {
            VisitedGlobalIFunc.insert(IFunc);
          }
        }
      }
    }
    FunctionsToBeVisited.erase(CurrentF);
    VisitedFunctions.insert(CurrentF);
  }

  llvm::ValueToValueMapTy VMap;

  // Clone the visited Global Variables, and make them external
  // Also clone their Metadata here
  for (const auto GV : VisitedGlobalVariables) {
    auto *NewGV = new llvm::GlobalVariable(
        NewModule, GV->getValueType(), GV->isConstant(),
        llvm::GlobalValue::ExternalLinkage, (llvm::Constant *)nullptr,
        GV->getName(), (llvm::GlobalVariable *)nullptr,
        GV->getThreadLocalMode(), GV->getType()->getAddressSpace());
    NewGV->copyAttributesFrom(GV);

    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 1> MDs;
    GV->getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(MD.first, *MapMetadata(MD.second, VMap));

    VMap[GV] = NewGV;
  }

  // Loop over the functions in the module, making external functions as before
  for (auto *F : VisitedFunctions) {
    llvm::Function *NF = llvm::Function::Create(
        cast<llvm::FunctionType>(F->getValueType()), F->getLinkage(),
        F->getAddressSpace(), F->getName(), &NewModule);
    NF->copyAttributesFrom(F);
    VMap[F] = NF;
  }

  for (const auto &OldFunc : VisitedFunctions) {
    if (OldFunc->isDeclaration()) {
      // Copy over metadata for declarations since we're not doing it below in
      // CloneFunctionInto().
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 1> MDs;
      OldFunc->getAllMetadata(MDs);
      for (auto MD : MDs)
        cast<llvm::Function>(VMap[OldFunc])->addMetadata(MD.first, *MapMetadata(MD.second, VMap));
      continue;
    }
    std::vector<llvm::Type *> ArgTypes;

//    // Unlike cloneFunction, all arguments must be preserved
//    for (const llvm::Argument &I : OldFunc->args())
//      ArgTypes.push_back(I.getType());
//
//    // Create a new function type...
//    llvm::FunctionType *FTy = llvm::FunctionType::get(
//        OldFunc->getFunctionType()->getReturnType(), ArgTypes,
//        OldFunc->getFunctionType()->isVarArg());
//
//    // Create the new function...
//    llvm::Function *NewF = llvm::Function::Create(
//        FTy, OldFunc->getLinkage(), OldFunc->getAddressSpace(),
//        OldFunc->getName(), &NewModule);
//    NewF->setIsNewDbgInfoFormat(OldFunc->IsNewDbgInfoFormat);

    auto *NewF = cast<llvm::Function>(VMap[OldFunc]);
    // Loop over the arguments, copying the names of the mapped arguments
    // over...
    llvm::Function::arg_iterator DestI = NewF->arg_begin();
    for (const llvm::Argument &I : OldFunc->args())
      if (VMap.count(&I) == 0) {     // Is this argument preserved?
        DestI->setName(I.getName()); // Copy the name over...
        VMap[&I] = &*DestI++;        // Add mapping to VMap
      }

    llvm::SmallVector<llvm::ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(NewF, OldFunc, VMap,
                      llvm::CloneFunctionChangeType::DifferentModule, Returns);
  }
  NewModule.dump();
  return nullptr;
}

} // namespace luthier