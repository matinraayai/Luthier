//===-- InjectedPayloadCreationPass.h -----------------------------*-C++-*-===//
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
/// Defines the \c InjectedPayloadCreationPass CRTP base class, which prvoides
/// its derived convenience helpers for constructing injected payload functions
/// in the prototype's instrumentation module.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_CREATION_PASS_H
#define LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_CREATION_PASS_H
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicCalls.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/Twine.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <variant>

namespace luthier {

/// \brief CRTP base for Luthier passes that construct injected payload
/// functions before target <tt>llvm::MachineInstr</tt>s.
template <typename Derived>
class InjectedPayloadCreationPass : public llvm::PassInfoMixin<Derived> {
public:
  /// Describes a register argument passed to a hook: the register to read
  /// and the LLVM type to interpret it as.
  struct RegArg {
    llvm::MCRegister Reg;
    llvm::Type *Ty;
  };

  /// An argument to a hook invocation: either a pre-computed \c Value* or
  /// a register read via \c luthier::readReg.
  using PayloadArg = std::variant<llvm::Value *, RegArg>;

protected:
  /// \brief Creates a new injected-payload function in \p IModule for \p
  /// TargetMI.
  ///
  /// \details A single entry \c BasicBlock is created and an \c IRBuilderBase
  /// pointing into it is passed to \p Build.  After \p Build returns, a \c ret
  /// void is appended and \c assignToInject is called to mark the function and
  /// attach
  /// \c !luthier.target_instr_point metadata.
  ///
  /// Name collisions with existing payloads for the same MI are resolved
  /// automatically by LLVM (appending \c .1, \c .2, …).
  ///
  /// \returns the newly created function, or an error if \p Build or metadata
  /// attachment fails
  llvm::Expected<llvm::Function *> createInjectedPayload(
      llvm::Module &IModule, const llvm::MachineInstr &TargetMI,
      llvm::function_ref<llvm::Error(llvm::IRBuilderBase &)> Build) {
    auto *FTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(IModule.getContext()), /*isVarArg=*/false);

    const llvm::MachineBasicBlock *MBB = TargetMI.getParent();
    const llvm::MachineFunction *MF = MBB->getParent();
    unsigned MBBNum = MBB->getNumber();
    unsigned InstrIdx = 0;
    for (const auto &I : *MBB) {
      if (&I == &TargetMI)
        break;
      ++InstrIdx;
    }
    std::string Name = ("luthier.payload." + MF->getName() + "." +
                        llvm::Twine(MBBNum) + "." + llvm::Twine(InstrIdx))
                           .str();

    auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                                     Name, IModule);

    if (!F->getReturnType()->isVoidTy() || F->arg_size() != 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Injected payload function must be void() with no arguments: '" +
          F->getName().str() + "'");

    llvm::BasicBlock *BB =
        llvm::BasicBlock::Create(IModule.getContext(), "", F);
    llvm::IRBuilder<> Builder(BB);

    if (auto Err = Build(Builder))
      return std::move(Err);

    Builder.CreateRetVoid();

    if (auto Err = assignToInject(*F, TargetMI))
      return std::move(Err);

    return F;
  }

  /// Convenience overload: creates an injected-payload function for \p TargetMI
  /// that calls \p HookFn with \p Args
  /// \c RegArg entries in \p Args are lowered to \c luthier::readReg intrinsic
  /// calls; \c Value* entries are forwarded directly
  llvm::Error createInjectedPayload(llvm::Function &HookFn,
                                    const llvm::MachineInstr &TargetMI,
                                    llvm::ArrayRef<PayloadArg> Args = {}) {
    llvm::Module *IModule = HookFn.getParent();
    assert(IModule && "HookFn has no parent module");

    auto FnOrErr = createInjectedPayload(
        *IModule, TargetMI, [&](llvm::IRBuilderBase &Builder) -> llvm::Error {
          llvm::SmallVector<llvm::Value *, 8> ResolvedArgs;
          for (const auto &Arg : Args) {
            if (std::holds_alternative<llvm::Value *>(Arg)) {
              ResolvedArgs.push_back(std::get<llvm::Value *>(Arg));
            } else {
              const RegArg &RA = std::get<RegArg>(Arg);
              llvm::CallInst *ReadVal = insertCallToIntrinsic(
                  *IModule, Builder, "luthier::readReg", *RA.Ty, RA.Reg.id());
              ResolvedArgs.push_back(ReadVal);
            }
          }
          llvm::CallInst *HookCall = Builder.CreateCall(&HookFn, ResolvedArgs);
          // A payload is spliced into the middle of a target MachineFunction,
          // so it must never tail-call. Left alone, this call sits in tail
          // position ahead of the `ret void` below and TailCallElimPass marks
          // it `tail`; ISel then lowers it to SI_TCRETURN, a terminator that
          // jumps rather than calls. TargetModulePatcherPass drops that
          // terminator when it inlines the payload -- correctly, since a return
          // cannot survive mid-block -- and the call to the hook disappears,
          // leaving only a dangling PC-relative reference to it. `notail` keeps
          // the call a real call all the way through ISel.
          HookCall->setTailCallKind(llvm::CallInst::TCK_NoTail);
          return llvm::Error::success();
        });

    if (FnOrErr) {
      llvm::Function *PayloadFn = *FnOrErr;
      if (HookFn.hasFnAttribute("target-cpu"))
        PayloadFn->addFnAttr(HookFn.getFnAttribute("target-cpu"));
      PayloadFn->addFnAttr("target-features", "+wavefrontsize64");
    }

    if (!FnOrErr)
      return FnOrErr.takeError();
    return llvm::Error::success();
  }

  /// Marks \p PayloadFn as an injected payload for \p TargetMI.
  /// \p PayloadFn must be \c void() with no arguments; an error is returned
  /// otherwise
  llvm::Error assignToInject(llvm::Function &PayloadFn,
                             const llvm::MachineInstr &TargetMI) {
    if (!PayloadFn.getReturnType()->isVoidTy() || PayloadFn.arg_size() != 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Injected payload function must be void() with no arguments: '" +
          PayloadFn.getName().str() + "'");

    PayloadFn.addFnAttr(InjectedPayloadAttribute);
    // Injected payloads carry their own custom prologue/epilogue emitted by
    // InjectedPayloadPEIPass — for state-value-array load/store + frame-reg
    // spill/restore — so suppress LLVM's stock PEI emission for them. RA
    // still runs normally on Naked functions; Naked only gates the
    // prologue/epilogue + CSR-save/restore emission inside
    // PrologEpilogInserter (see PrologEpilogInserter.cpp:259, 679).
    PayloadFn.addFnAttr(llvm::Attribute::Naked);

    // Payloads have internal linkage and no static caller (the patcher
    // inlines them at MIR time, after the IR pipeline runs). Anchor them
    // against `@llvm.compiler.used` so the IModule's default `-O` pipeline
    // (specifically GlobalDCE) doesn't strip them before the patcher gets
    // a chance to inline them into the target MFs.
    llvm::appendToCompilerUsed(*PayloadFn.getParent(), {&PayloadFn});

    if (!TargetMachineInstrMDNode::getInstrMDNodeIfExists(TargetMI)) {
      auto MDOrErr = TargetMachineInstrMDNode::initializeMDNode(
          *const_cast<llvm::MachineInstr *>(&TargetMI));
      if (!MDOrErr)
        return MDOrErr.takeError();
    }

    PayloadFn.setMetadata(TargetInstrPointAttr, TargetMI.getPCSections());
    return llvm::Error::success();
  }
};

} // namespace luthier

#endif
