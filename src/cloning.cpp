#include "cloning.hpp"
#include "error.hpp"

#include <llvm/Analysis/CallGraph.h>
#include <llvm/Transforms/Utils/Cloning.h>

// Copied over from LLVM's Module cloning
static void copyComdat(llvm::GlobalObject *Dst, const llvm::GlobalObject *Src) {
  const llvm::Comdat *SC = Src->getComdat();
  if (!SC)
    return;
  llvm::Comdat *DC = Dst->getParent()->getOrInsertComdat(SC->getName());
  DC->setSelectionKind(SC->getSelectionKind());
  Dst->setComdat(DC);
}

namespace luthier {

void cloneModuleAttributes(const llvm::Module &OldModule,
                                           llvm::Module &NewModule) {
  NewModule.setModuleIdentifier(OldModule.getModuleIdentifier());
  NewModule.setSourceFileName(OldModule.getSourceFileName());
  NewModule.setDataLayout(OldModule.getDataLayout());
  NewModule.setTargetTriple(OldModule.getTargetTriple());
  NewModule.setModuleInlineAsm(OldModule.getModuleInlineAsm());
  NewModule.IsNewDbgInfoFormat = OldModule.IsNewDbgInfoFormat;
}

llvm::Error cloneGlobalValuesIntoModule(
    llvm::ArrayRef<llvm::GlobalValue *> DeepCloneOldValues,
    llvm::Module &NewModule) {
  llvm::outs() << "\n=====> Here in cloneGlobalValuesIntoModule\n";
  // The deep clone old Value list shouldn't be empty
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(!DeepCloneOldValues.empty()));
  // All deep clone old Values need to have the same parent Module
  llvm::Module &OldModule = *DeepCloneOldValues[0]->getParent();

  for (const auto *DeepCloneOldValue : DeepCloneOldValues) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(
        DeepCloneOldValue->getParent() == &OldModule));
  }
  // A set of visited values that need to be deep cloned
  llvm::DenseSet<llvm::GlobalValue *> VisitedDeepCloneOldValues{};

  // Iterate over all the deep clone values, and find all related values
  // Visited Global Values means they are related to the deep clone content,
  // but not necessarily will be deep cloned
  llvm::DenseSet<llvm::Function *> VisitedOldFunctions;
  llvm::DenseSet<llvm::GlobalVariable *> VisitedOldGlobalVariables;
  llvm::DenseSet<llvm::GlobalAlias *> VisitedOldGlobalAliases;
  llvm::DenseSet<llvm::GlobalIFunc *> VisitedOldGlobalIFunc;
  llvm::DenseSet<llvm::GlobalValue *> DeepCloneOldGlobalValuesToBeVisited{
      DeepCloneOldValues.begin(), DeepCloneOldValues.end()};
  llvm::GlobalValue *CurrentOldDeepCloneValue{nullptr};
  while (!DeepCloneOldGlobalValuesToBeVisited.empty()) {
    // Pop the Current deep clone value
    CurrentOldDeepCloneValue = *DeepCloneOldGlobalValuesToBeVisited.begin();
    // A sanity check for the algorithm; Make sure we don't revisit the same
    // deep clone old value
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        !VisitedDeepCloneOldValues.contains(CurrentOldDeepCloneValue)));
    // Function-specific logic
    if (auto *CurrentDeepCloneFunction =
            llvm::dyn_cast<llvm::Function>(CurrentOldDeepCloneValue)) {
      // Walk over the function's instructions, and identify all global objects
      // it uses from the parent module
      // It is a depth-first search
      // llvm::outs() << "Dumping function " << CurrentDeepCloneFunction->getName()
      //              << "\n";
      for (const auto &BB : *CurrentDeepCloneFunction) {
        for (const auto &I : BB) {
          // llvm::outs() << "Dumping instruction\n";
//          I.dump();
          for (const auto &Op : I.operands()) {
            if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(&Op)) {
              VisitedOldGlobalVariables.insert(GV);
            }
            if (auto *Func = llvm::dyn_cast<llvm::Function>(&Op)) {
              VisitedOldFunctions.insert(Func);
              // The if statement is there to avoid recursion
              if (Func != CurrentOldDeepCloneValue) {
                auto Linkage = Func->getLinkage();
                auto Visibility = Func->getVisibility();
                if (Linkage == llvm::GlobalValue::PrivateLinkage ||
                    Linkage == llvm::GlobalValue::InternalLinkage) {
                  if (!VisitedDeepCloneOldValues.contains(Func))
                    DeepCloneOldGlobalValuesToBeVisited.insert(Func);
                }
                // llvm::outs() << "Func name: " << Func->getName()
                //              << " Linkage: " << Linkage
                //              << " Visiblity: " << Visibility << "\n";
              }
            }
            if (auto *GA = llvm::dyn_cast<llvm::GlobalAlias>(&Op)) {
              VisitedOldGlobalAliases.insert(GA);
              if (auto *FuncAliasee =
                      llvm::dyn_cast<llvm::Function>(GA->getAliasee())) {
                auto Linkage = FuncAliasee->getLinkage();
                if (Linkage == llvm::GlobalValue::PrivateLinkage ||
                    Linkage == llvm::GlobalValue::InternalLinkage) {
                  if (!VisitedDeepCloneOldValues.contains(FuncAliasee))
                    DeepCloneOldGlobalValuesToBeVisited.insert(FuncAliasee);
                }
              }
            }
            if (auto *IFunc = llvm::dyn_cast<llvm::GlobalIFunc>(&Op)) {
              VisitedOldGlobalIFunc.insert(IFunc);
              if (auto ResolverFunc = IFunc->getResolverFunction()) {
                auto ResolverLinkage = ResolverFunc->getLinkage();
                if (ResolverLinkage == llvm::GlobalValue::PrivateLinkage ||
                    ResolverLinkage == llvm::GlobalValue::InternalLinkage) {
                  if (!VisitedDeepCloneOldValues.contains(ResolverFunc))
                    DeepCloneOldGlobalValuesToBeVisited.insert(ResolverFunc);
                }
              }
            }
          }
        }
      }
      VisitedOldFunctions.insert(CurrentDeepCloneFunction);
    } else if (auto *CurrentDeepCloneOldVariable =
                   llvm::dyn_cast<llvm::GlobalVariable>(
                       CurrentOldDeepCloneValue)) {
      VisitedOldGlobalVariables.insert(CurrentDeepCloneOldVariable);
    } else if (auto *CurrentDeepCloneOldGlobalAlias =
                   llvm::dyn_cast<llvm::GlobalAlias>(
                       CurrentOldDeepCloneValue)) {
      VisitedOldGlobalAliases.insert(CurrentDeepCloneOldGlobalAlias);
      auto *Aliasee = llvm::dyn_cast<llvm::GlobalValue>(
          CurrentDeepCloneOldGlobalAlias->getAliasee());
      if (Aliasee && !VisitedDeepCloneOldValues.contains(Aliasee))
        DeepCloneOldGlobalValuesToBeVisited.insert(Aliasee);
    } else if (auto *CurrentDeepCloneGlobalIFunc =
                   llvm::dyn_cast<llvm::GlobalIFunc>(
                       CurrentOldDeepCloneValue)) {
      auto ResolverFunc = CurrentDeepCloneGlobalIFunc->getResolverFunction();
      if (ResolverFunc && !VisitedDeepCloneOldValues.contains(ResolverFunc))
        DeepCloneOldGlobalValuesToBeVisited.insert(ResolverFunc);
    }
    DeepCloneOldGlobalValuesToBeVisited.erase(CurrentOldDeepCloneValue);
    VisitedDeepCloneOldValues.insert(CurrentOldDeepCloneValue);
  }

  // Create the declaration of all related variables and create a mapping
  // of them for the cloning functions to work with
  llvm::ValueToValueMapTy VMap;

  // Loop over all the visited global variables, making corresponding globals
  // in the new module.  Here we add them to the VMap and to the new Module.
  // Attributes and initializers will be added later.
  for (const llvm::GlobalVariable *OldGV : VisitedOldGlobalVariables) {
    auto *NewGV = new llvm::GlobalVariable(
        NewModule, OldGV->getValueType(), OldGV->isConstant(),
        OldGV->getLinkage(), (llvm::Constant *)nullptr, OldGV->getName(),
        (llvm::GlobalVariable *)nullptr, OldGV->getThreadLocalMode(),
        OldGV->getType()->getAddressSpace());
    NewGV->copyAttributesFrom(OldGV);
    VMap[OldGV] = NewGV;
  }

  // Loop over the visited functions and create their declarations
  for (const llvm::Function *OldFunc : VisitedOldFunctions) {
    auto *NewFunc = llvm::Function::Create(
        llvm::cast<llvm::FunctionType>(OldFunc->getValueType()),
        OldFunc->getLinkage(), OldFunc->getAddressSpace(), OldFunc->getName(),
        &NewModule);
    NewFunc->copyAttributesFrom(OldFunc);
    VMap[OldFunc] = NewFunc;
  }

  // Loop over the visited aliases
  for (const llvm::GlobalAlias *OldGA : VisitedOldGlobalAliases) {
    auto *Aliasee = llvm::dyn_cast<llvm::GlobalValue>(OldGA->getAliasee());
    if (!VisitedDeepCloneOldValues.contains(OldGA) &&
        !VisitedDeepCloneOldValues.contains(Aliasee)) {
      // An alias cannot act as an external reference, so we need to create
      // either a function or a global variable depending on the value type.
      llvm::GlobalValue *GV;
      if (OldGA->getValueType()->isFunctionTy())
        GV = llvm::Function::Create(
            cast<llvm::FunctionType>(OldGA->getValueType()),
            llvm::GlobalValue::ExternalLinkage, OldGA->getAddressSpace(),
            OldGA->getName(), &NewModule);
      else
        GV = new llvm::GlobalVariable(NewModule, OldGA->getValueType(), false,
                                      llvm::GlobalValue::ExternalLinkage,
                                      nullptr, OldGA->getName(), nullptr,
                                      OldGA->getThreadLocalMode(),
                                      OldGA->getType()->getAddressSpace());
      VMap[OldGA] = GV;
      // Don't copy over the attributes
      continue;
    }
    auto *NewGA = llvm::GlobalAlias::create(
        OldGA->getValueType(), OldGA->getType()->getPointerAddressSpace(),
        OldGA->getLinkage(), OldGA->getName(), &NewModule);
    NewGA->copyAttributesFrom(OldGA);
    VMap[OldGA] = NewGA;
  }

  for (const llvm::GlobalIFunc *OldIFunc : VisitedOldGlobalIFunc) {
    // Defer setting the resolver function until after functions are cloned.
    auto *NewIFunc = llvm::GlobalIFunc::create(
        OldIFunc->getValueType(), OldIFunc->getAddressSpace(),
        OldIFunc->getLinkage(), OldIFunc->getName(), nullptr, &NewModule);
    NewIFunc->copyAttributesFrom(OldIFunc);
    VMap[OldIFunc] = NewIFunc;
  }

  // Now that all the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  for (const llvm::GlobalVariable *OldGV : VisitedOldGlobalVariables) {
    auto *NewGV = llvm::cast<llvm::GlobalVariable>(VMap[OldGV]);

    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 1> MDs;
    OldGV->getAllMetadata(MDs);
    for (auto MD : MDs)
      NewGV->addMetadata(MD.first, *MapMetadata(MD.second, VMap));

    if (OldGV->isDeclaration())
      continue;
    if (!VisitedDeepCloneOldValues.contains(OldGV)) {
      NewGV->setLinkage(llvm::GlobalValue::ExternalLinkage);
    }
    if (OldGV->hasInitializer())
      NewGV->setInitializer(MapValue(OldGV->getInitializer(), VMap));

    copyComdat(NewGV, OldGV);
  }

  for (const llvm::Function *OldFunc : VisitedOldFunctions) {
    auto *NewFunc = llvm::cast<llvm::Function>(VMap[OldFunc]);

    if (OldFunc->isDeclaration()) {
      // Copy over metadata for declarations since we're not doing it below in
      // CloneFunctionInto().
      llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 1> MDs;
      OldFunc->getAllMetadata(MDs);
      for (auto MD : MDs)
        NewFunc->addMetadata(MD.first, *MapMetadata(MD.second, VMap));
      continue;
    }

    if (!VisitedDeepCloneOldValues.contains(OldFunc)) {
      // Skip after setting the correct linkage for an external reference.
      NewFunc->setLinkage(llvm::GlobalValue::ExternalLinkage);
      // Personality function is not valid on a declaration.
      NewFunc->setPersonalityFn(nullptr);
      continue;
    }

    llvm::Function::arg_iterator DestI = NewFunc->arg_begin();
    for (const llvm::Argument &J : OldFunc->args()) {
      DestI->setName(J.getName());
      VMap[&J] = &*DestI++;
    }

    llvm::SmallVector<llvm::ReturnInst *, 8> Returns; // Ignore returns cloned.

    CloneFunctionInto(NewFunc, OldFunc, VMap,
                      llvm::CloneFunctionChangeType::DifferentModule, Returns);

    if (OldFunc->hasPersonalityFn())
      NewFunc->setPersonalityFn(MapValue(OldFunc->getPersonalityFn(), VMap));

    copyComdat(NewFunc, OldFunc);
  }

  for (const auto GV : VisitedOldGlobalVariables) {
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

  // And aliases
  for (const llvm::GlobalAlias *OldGA : VisitedOldGlobalAliases) {
    // We already dealt with undefined aliases above.
    auto *Aliasee = llvm::dyn_cast<llvm::GlobalValue>(OldGA->getAliasee());
    if (!VisitedDeepCloneOldValues.contains(OldGA) &&
        !VisitedDeepCloneOldValues.contains(Aliasee))
      continue;
    auto *NewGA = llvm::cast<llvm::GlobalAlias>(VMap[OldGA]);
    if (const llvm::Constant *C = OldGA->getAliasee())
      NewGA->setAliasee(MapValue(C, VMap));
  }

  for (const llvm::GlobalIFunc *OldIFunc : VisitedOldGlobalIFunc) {
    auto *NewIFunc = llvm::cast<llvm::GlobalIFunc>(VMap[OldIFunc]);
    if (const llvm::Constant *Resolver = OldIFunc->getResolver())
      NewIFunc->setResolver(MapValue(Resolver, VMap));
  }

  // And named metadata....
  for (const llvm::NamedMDNode &NMD : OldModule.named_metadata()) {
    llvm::NamedMDNode *NewNMD = NewModule.getOrInsertNamedMetadata(NMD.getName());
    NewNMD->clearOperands();
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapMetadata(NMD.getOperand(i), VMap));
  }

  llvm::outs() << "\n=====> End of cloneGlobalValuesIntoModule\n";
  return llvm::Error::success();
}

} // namespace luthier
