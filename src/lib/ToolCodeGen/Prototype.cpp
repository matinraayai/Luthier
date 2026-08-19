//===-- Prototype.cpp -------------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file
/// Implements out-of-line definitions for \c Prototype.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/Prototype.h"

#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicCalls.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InjectedPayloadSideEffectsAnalysis.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"

#include <cassert>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetOpcodes.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManagerImpl.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

namespace luthier {

static llvm::Error assignToInject(llvm::Function &PayloadFn,
                                  llvm::Module &TargetModule,
                                  llvm::MachineInstr &TargetMI,
                                  llvm::FunctionAnalysisManager &IFAM) {
  if (!PayloadFn.getReturnType()->isVoidTy() || PayloadFn.arg_size() != 0)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Injected payload function must be void() with no arguments: '" +
        PayloadFn.getName().str() + "'");

  PayloadFn.addFnAttr(InjectedPayloadAttribute);

  PayloadFn.addFnAttr(llvm::Attribute::Naked);

  llvm::appendToCompilerUsed(*PayloadFn.getParent(), {&PayloadFn});

  llvm::Function *ExternHandle = llvm::cast<llvm::Function>(
      TargetModule
          .getOrInsertFunction(PayloadFn.getName(), PayloadFn.getFunctionType())
          .getCallee());

  auto &PayloadSideEffects =
      IFAM.getResult<InjectedPayloadSideEffectsAnalysis>(PayloadFn);

  // Emit the PATCHPOINT marker immediately before the target instruction.
  // Operand layout (see llvm::PatchpointOpers): ID, NBytes, Target, NArgs,
  // CC, then args, then implicit uses/defs. The marker is transient — the
  // patcher rewrites it away before final code emission — so a zero ID and
  // zero shadow are sufficient; the extern handle is what identifies the
  // payload downstream.
  llvm::MachineFunction &MF = *TargetMI.getMF();
  const llvm::TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  auto MIB = llvm::BuildMI(*TargetMI.getParent(), TargetMI, llvm::DebugLoc(),
                           TII.get(llvm::TargetOpcode::PATCHPOINT))
                 .addImm(0)
                 .addImm(0)
                 .addGlobalAddress(ExternHandle)
                 .addImm(0)
                 .addImm(llvm::CallingConv::C);
  for (llvm::MCRegister R : PayloadSideEffects.reads())
    (void)MIB.addReg(R, llvm::RegState::Implicit);
  for (llvm::MCRegister R : PayloadSideEffects.writes())
    (void)MIB.addReg(R, llvm::RegState::ImplicitDefine);
  LUTHIER_RETURN_ON_ERROR(
      TargetMachineInstrMDNode::initializeMDNode(*MIB).takeError());
  MF.getFrameInfo().setHasPatchPoint(true);

  return llvm::Error::success();
}

Prototype::Prototype(
    std::unique_ptr<llvm::Module> Target,
    std::unique_ptr<llvm::Module> IModule)
    : TargetModule(std::move(Target)), IModule(std::move(IModule)) {
  assert(this->TargetModule && this->IModule &&
         "Prototype modules must be non-null");
  assert(&this->TargetModule->getContext() == &this->IModule->getContext() &&
         "Prototype modules must share an LLVMContext");
}

llvm::Expected<llvm::Function *> Prototype::createInjectedPayload(
    llvm::MachineInstr &TargetMI, llvm::FunctionAnalysisManager &IFAM,
    llvm::function_ref<llvm::Error(llvm::IRBuilderBase &)> Build) {
  auto *FTy = llvm::FunctionType::get(
      llvm::Type::getVoidTy(IModule->getContext()), /*isVarArg=*/false);

  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                                   "luthier.payload", *IModule);

  if (!F->getReturnType()->isVoidTy() || F->arg_size() != 0)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Injected payload function must be void() with no arguments: '" +
        F->getName().str() + "'");

  llvm::BasicBlock *BB = llvm::BasicBlock::Create(IModule->getContext(), "", F);
  llvm::IRBuilder<> Builder(BB);

  if (auto Err = Build(Builder))
    return std::move(Err);

  Builder.CreateRetVoid();

  if (auto Err = assignToInject(*F, *TargetModule, TargetMI, IFAM))
    return std::move(Err);

  return F;
}

llvm::Expected<llvm::Function *> Prototype::createInjectedPayload(
    llvm::Function &HookFn, llvm::MachineInstr &TargetMI,
    llvm::FunctionAnalysisManager &IFAM, llvm::ArrayRef<PayloadArg> Args) {

  return createInjectedPayload(
      TargetMI, IFAM, [&](llvm::IRBuilderBase &Builder) -> llvm::Error {
        // Materialize each PayloadArg into a Value* the hook can consume.
        // RegArg entries emit a luthier::readReg intrinsic call whose result
        // (of the requested type) becomes the argument; Value* entries are
        // forwarded verbatim.
        llvm::SmallVector<llvm::Value *, 4> HookArgs;
        HookArgs.reserve(Args.size());
        for (const PayloadArg &A : Args) {
          if (auto *const *V = std::get_if<llvm::Value *>(&A)) {
            HookArgs.push_back(*V);
          } else {
            const RegArg &R = std::get<RegArg>(A);
            HookArgs.push_back(insertCallToIntrinsic(
                *IModule, Builder, "luthier::readReg", *R.Ty,
                static_cast<uint32_t>(R.Reg.id())));
          }
        }
        llvm::CallInst *HookCall = Builder.CreateCall(&HookFn, HookArgs);
        // Force-inline the hook function for now
        // TODO: Add arg that prevent force inlining the hook function
        llvm::InlineFunctionInfo IFI;
        llvm::InlineResult IR = llvm::InlineFunction(*HookCall, IFI);
        return !IR.isSuccess() ? LUTHIER_MAKE_GENERIC_ERROR(
                                     "Failed to force-inline hook '" +
                                     HookFn.getName().str() +
                                     "' into payload: " + IR.getFailureReason())
                               : llvm::Error::success();
      });
}

void Prototype::forEachTargetMF(
    PrototypeAnalysisManager &PAM,
    llvm::function_ref<void(llvm::MachineFunction &)> Fn) {
  llvm::FunctionAnalysisManager &FAM =
      PAM.getResult<TargetFunctionAnalysisManagerPrototypeProxy>(*this)
          .getManager();
  for (llvm::Function &F : *TargetModule) {
    if (auto *MFRes = FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F))
      Fn(MFRes->getMF());
  }
}

/// Runs \p Pass over \p M, which is the module of \p IP selected by the caller,
/// against \p MAM, that module's own \c llvm::ModuleAnalysisManager. Mirrors
/// LLVM's ModuleToFunctionPassAdaptor::run / the machinery in the other LLVM
/// Pass adaptors.
static llvm::PreservedAnalyses
runModulePass(RunOnTargetModuleAdaptor::PassConceptT &Pass, llvm::Module &M,
              llvm::ModuleAnalysisManager &MAM, Prototype &IP,
              PrototypeAnalysisManager &IPAM) {
  // Request PassInstrumentation from the *module* analysis manager; it drives
  // the instrumenting callbacks around the pass below. Deliberately not taken
  // from IPAM: the Prototype level runs on a separate, empty PIC because LLVM's
  // StandardInstrumentations cannot name a Prototype IR unit (see
  // InstrumentationPassBuilder::PrototypePIC). The pass being wrapped here is a
  // plain module pass, so the module-level callbacks apply to it and keep
  // -print-after-all / -time-passes working.
  llvm::PassInstrumentation PI =
      MAM.getResult<llvm::PassInstrumentationAnalysis>(M);

  // Check the BeforePass callbacks; if asked to skip, do not run the pass and
  // report that everything is preserved.
  if (!PI.runBeforePass<llvm::Module>(Pass, M))
    return llvm::PreservedAnalyses::all();

  // Which functions of M exist going in, so results belonging to any the pass
  // deletes can be evicted below. Held as bare addresses: a deleted Function
  // must not be dereferenced afterwards, and is only ever needed as a cache key.
  llvm::SmallPtrSet<const llvm::Function *, 16> FunctionsBefore;
  for (llvm::Function &F : M)
    FunctionsBefore.insert(&F);

  llvm::PreservedAnalyses PA = Pass.run(M, MAM);

  // Reconcile the inner managers for M. Querying the proxy here also ensures it
  // is cached for M, without which MAM.invalidate has nothing to descend
  // through and M's own function analyses would silently go stale.
  llvm::FunctionAnalysisManager &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Evict results for functions the pass deleted. Preserving the proxy below
  // moves that duty here: LLVM's function-deleting module passes handle their
  // own (both inliners, ArgumentPromotion, GlobalOpt, FunctionSpecialization
  // all call FAM.clear), but GlobalDCEPass does not — it reports
  // PreservedAnalyses::none() and leans on the blanket clear. Left alone, its
  // deleted functions would strand results that any later Function reusing the
  // address would pick up. AnalysisManager::clear only uses the argument's
  // address as a key, so a freed Function is never read.
  for (llvm::Function &F : M)
    FunctionsBefore.erase(&F);
  for (const llvm::Function *Dead : FunctionsBefore)
    FAM.clear(*const_cast<llvm::Function *>(Dead),
              "<function deleted by module pass>");

  // FunctionAnalysisManagerModuleProxy is preserved deliberately, so LLVM's
  // proxy takes its per-function invalidation branch over the functions of M
  // rather than calling InnerAM->clear() on M's whole FunctionAnalysisManager
  // (see PassManager.cpp). The pass's own PA still drives that walk, so nothing
  // it really invalidated survives; what is spared is M's lifted MIR, which
  // MachineFunctionAnalysis is built to bring through such a walk intact —
  // "unless it is invalidated explicitly, it should remain preserved". The
  // instrumentation module needs that: ISel and the machine-pass half of the
  // codegen pipeline are separate wrapped pipelines, and its MIR has to survive
  // from one to the other.
  llvm::PreservedAnalyses ModulePA = PA;
  ModulePA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
  MAM.invalidate(M, ModulePA);

  PI.runAfterPass(Pass, M, PA);

  // MAM.invalidate above already reconciled module-level analyses for M and,
  // through MAM's own FunctionAnalysisManagerModuleProxy, the function-level
  // ones too. So from the Prototype pass manager's point of view nothing is
  // left to invalidate at any inner level, and keeping the proxies live is what
  // stops it from clearing managers that were just brought up to date. The
  // other module's three proxies are preserved for the stronger reason that a
  // pass over M cannot have touched it at all.
  PA.preserveSet<llvm::AllAnalysesOn<llvm::Module>>();
  PA.preserve<TargetModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<TargetMachineFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleFunctionAnalysisManagerPrototypeProxy>();
  PA.preserve<IModuleMachineFunctionAnalysisManagerPrototypeProxy>();
  return PA;
}

llvm::PreservedAnalyses
RunOnTargetModuleAdaptor::run(Prototype &IP,
                              PrototypeAnalysisManager &IPAM) {
  return runModulePass(
      *Pass, IP.getTargetModule(),
      IPAM.getResult<TargetModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager(),
      IP, IPAM);
}

void RunOnTargetModuleAdaptor::printPipeline(
    llvm::raw_ostream &OS,
    llvm::function_ref<llvm::StringRef(llvm::StringRef)> MapClassName) {
  OS << "target(";
  Pass->printPipeline(OS, MapClassName);
  OS << ")";
}

llvm::PreservedAnalyses RunOnInstrumentationModuleAdaptor::run(
    Prototype &IP, PrototypeAnalysisManager &IPAM) {
  return runModulePass(
      *Pass, IP.getInstrumentationModule(),
      IPAM.getResult<IModuleAnalysisManagerPrototypeProxy>(IP).getManager(),
      IP, IPAM);
}

void RunOnInstrumentationModuleAdaptor::printPipeline(
    llvm::raw_ostream &OS,
    llvm::function_ref<llvm::StringRef(llvm::StringRef)> MapClassName) {
  OS << "instrumentation(";
  Pass->printPipeline(OS, MapClassName);
  OS << ")";
}

} // namespace luthier

// Explicit template instantiation for IP
namespace llvm {
// PassManager::run emits a stack-trace entry that calls this per-IR-unit hook;
// LLVM only provides specializations for its own IR units (Module, Function),
// so IP must supply its own before the PassManager instantiation below.
template <>
void printIRUnitNameForStackTrace<luthier::Prototype>(
    raw_ostream &OS, const luthier::Prototype &IR) {
  OS << "prototype for \"" << IR.getName() << "\"";
}

template class PassManager<luthier::Prototype>;
template class AnalysisManager<luthier::Prototype>;
} // namespace llvm
