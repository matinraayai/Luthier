//===-- ProcessIntrinsicsAtIRLevelPass.cpp --------------------------------===//
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
/// Implements the \c ProcessIntrinsicsAtIRLevelPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/Intrinsic/ReadReg.h"
#include "luthier/Intrinsic/ReadSVA.h"
#include "luthier/Intrinsic/WriteReg.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/Twine.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/ScopedPrinter.h>
#include <sstream>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-process-intrinsics-at-ir-level-pass"

llvm::PreservedAnalyses
luthier::ProcessIntrinsicsAtIRLevelPass::run(llvm::Module &IModule,
                                             llvm::ModuleAnalysisManager &MAM) {

  LLVM_DEBUG(luthier::dbgs() << "=== ProcessIntrinsicsAtIRLevelPass: module '"
                             << IModule.getName() << "' ===\n");

  const auto &IntrinsicsProcessors =
      MAM.getResult<IntrinsicsProcessorsAnalysis>(IModule);

  auto &TM = reinterpret_cast<const llvm::GCNTargetMachine &>(
      MAM.getResult<llvm::MachineModuleAnalysis>(IModule).getMMI().getTarget());

  // Per-run dedup map: signature -> opaque key. Two semantically identical
  // intrinsic invocations share the same key and named-MD entry.
  llvm::StringMap<std::string> SignatureToKey;
  unsigned NextKeyId = 0;

  // Iterate over all functions and find the ones marked as a Luthier
  // intrinsic. Early increment since we remove the intrinsic function once
  // we have processed all its users.
  for (auto &F : llvm::make_early_inc_range(IModule.functions())) {
    if (F.hasFnAttribute(IntrinsicAttribute)) {
      // Find the processor for this intrinsic
      auto IntrinsicName =
          F.getFnAttribute(IntrinsicAttribute).getValueAsString();


      // Built processors not registered with the intrinsic processor
      IntrinsicIRProcessorFunc IRProcessor;
      if (IntrinsicName == "luthier::readReg") {
        IRProcessor = readRegIRProcessor;
      } else if (IntrinsicName == "luthier::writeReg") {
        IRProcessor = writeRegIRProcessor;
      } else if (IntrinsicName == "luthier::readSVA") {
        IRProcessor = readSVAIRProcessor;
      } else {
        // Every other intrinsic in the registry
        std::optional<IntrinsicProcessor> Processor =
            IntrinsicsProcessors.getProcessorIfRegistered(IntrinsicName);
        if (!Processor.has_value()) {
          IModule.getContext().emitError(
              llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                  "Intrinsic {0} is not registered", IntrinsicName))));
          return llvm::PreservedAnalyses::all();
        }
        IRProcessor = Processor->IRProcessor;
      }

      LLVM_DEBUG({
        luthier::dbgs() << "\n--- Intrinsic '" << IntrinsicName << "' ("
                        << F.getNumUses() << " use(s)) ---\n";
      });

      // Iterate over all users of the intrinsic
      // Early increment the loop range since we will replace and delete the
      // user in the process
      for (auto *User : llvm::make_early_inc_range(F.users())) {
        // Ensure the user is a Call instruction; Anything other usage is
        // illegal
        auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User);
        if (!CallInst) {
          IModule.getContext().emitError(
              llvm::formatv("Found a user of intrinsic {0} which is not a "
                            "call instruction: {1}.",
                            IntrinsicName, *User));
          return llvm::PreservedAnalyses::all();
        }

        LLVM_DEBUG({
          luthier::dbgs() << "  Call site in '"
                          << CallInst->getFunction()->getName() << "': ";
          CallInst->print(luthier::dbgs());
          luthier::dbgs() << "\n";
        });

        // Call the IR processor of the intrinsic on the user
        llvm::Expected<IntrinsicIRLoweringInfo> IRLoweringInfoOrErr =
            IRProcessor(F, *CallInst, TM);
        if (auto Err = IRLoweringInfoOrErr.takeError()) {
          IModule.getContext().emitError(CallInst,
                                         llvm::toString(std::move(Err)));
        }

        // Build the inline-asm constraint string and operand type/value
        // vectors from the IR processor's lowering info.
        const auto &ReturnValInfo = IRLoweringInfoOrErr->getReturnValueInfo();
        std::stringstream ConstraintSS;

        if (!ReturnValInfo.Val->getType()->isVoidTy())
          ConstraintSS << "=" << ReturnValInfo.Constraint;
        // Construct argument type vector
        llvm::SmallVector<llvm::Type *, 4> ArgTypes;
        llvm::SmallVector<llvm::Value *, 4> ArgValues;
        ArgTypes.reserve(IRLoweringInfoOrErr->getArgsInfo().size());
        ArgValues.reserve(IRLoweringInfoOrErr->getArgsInfo().size());
        for (const auto &[I, ArgInfo] :
             llvm::enumerate(IRLoweringInfoOrErr->getArgsInfo())) {
          if (I != 0 || (I == 0 && !ReturnValInfo.Val->getType()->isVoidTy()))
            ConstraintSS << ",";
          ArgTypes.push_back(ArgInfo.Val->getType());
          ArgValues.push_back(const_cast<llvm::Value *>(ArgInfo.Val));
          ConstraintSS << ArgInfo.Constraint;
        }

        // Create the inline-asm placeholder. The opaque key lives in the
        // template-string position (operand 0 of the eventual INLINEASM
        // MachineInstr after ISel), giving the MIR lowering pass a stable
        // handle that survives SelectionDAG
        auto *PlaceHolderInlineAsm = llvm::InlineAsm::get(
            llvm::FunctionType::get(ReturnValInfo.Val->getType(), ArgTypes,
                                    /*isVarArg=*/false),
            IntrinsicName, ConstraintSS.str(),
            /*hasSideEffects=*/true);
        auto *InlineAsmPlaceholderCall =
            llvm::CallInst::Create(PlaceHolderInlineAsm, ArgValues);
        InlineAsmPlaceholderCall->insertBefore(*CallInst->getParent(),
                                               CallInst->getIterator());
        // Replace all occurrences of the user with the inline assembly
        // placeholder
        CallInst->replaceAllUsesWith(InlineAsmPlaceholderCall);

        // Transfer debug info of the original use to the inline assembly
        // placeholder
        InlineAsmPlaceholderCall->copyMetadata(*CallInst);
        InlineAsmPlaceholderCall->setDebugLoc(CallInst->getDebugLoc());

        LLVM_DEBUG({
          luthier::dbgs() << "\n  Placeholder: ";
          InlineAsmPlaceholderCall->print(luthier::dbgs());
          luthier::dbgs() << "\n";
        });

        CallInst->eraseFromParent();
      }
      F.dropAllReferences();
      F.eraseFromParent();
    }
  }

  return llvm::PreservedAnalyses::all();
}
