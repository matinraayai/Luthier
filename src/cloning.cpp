#include "cloning.hpp"

#include <llvm/Transforms/Utils/Cloning.h>

llvm::Function *cloneFunctionIntoModule(const llvm::Function &F,
                                        llvm::Module &Module) {
  std::vector<llvm::Type *> ArgTypes;

  for (const llvm::Argument &I : F.args())
    ArgTypes.push_back(I.getType());

  // Create a new function type...
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(F.getFunctionType()->getReturnType(), ArgTypes,
                              F.getFunctionType()->isVarArg());

  // Create the new function...
  llvm::Function *NewF = llvm::Function::Create(
      FTy, F.getLinkage(), F.getAddressSpace(), F.getName(), &Module);
  NewF->setIsNewDbgInfoFormat(F.IsNewDbgInfoFormat);

  llvm::ValueToValueMapTy VMap;
  // Loop over the arguments, copying the names of the mapped arguments over...
  llvm::Function::arg_iterator DestI = NewF->arg_begin();
  for (const llvm::Argument &I : F.args())
    if (VMap.count(&I) == 0) {     // Is this argument preserved?
      DestI->setName(I.getName()); // Copy the name over...
      VMap[&I] = &*DestI++;        // Add mapping to VMap
    }

  llvm::SmallVector<llvm::ReturnInst *, 8> Returns; // Ignore returns cloned.
  CloneFunctionInto(NewF, &F, VMap,
                    llvm::CloneFunctionChangeType::DifferentModule, Returns);

  return NewF;
}