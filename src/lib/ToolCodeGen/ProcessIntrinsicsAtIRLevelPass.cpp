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
/// \file ProcessIntrinsicsAtIRLevelPass.cpp
/// Implements the \c ProcessIntrinsicsAtIRLevelPass class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
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

namespace {

/// Build a deterministic content-hash signature for an intrinsic call. Two
/// calls whose signatures match share a placeholder key — their lowering is
/// identical because everything the MIR processor looks at (intrinsic name +
/// aux MDNode contents) is captured in the signature, and the inline-asm
/// operand layout is captured by the return/argument types and any constant
/// argument values.
std::string
buildIntrinsicSignature(llvm::StringRef IntrinsicName, llvm::Type &ReturnType,
                        llvm::ArrayRef<llvm::Type *> ArgTypes,
                        llvm::ArrayRef<llvm::Value *> ArgValues,
                        llvm::ArrayRef<llvm::Metadata *> AuxOperands) {
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << "name=" << IntrinsicName << ";";
  OS << "ret=";
  ReturnType.print(OS, /*IsForDebug=*/false, /*NoDetails=*/true);
  OS << ";";
  for (size_t I = 0; I < ArgTypes.size(); ++I) {
    OS << "arg" << I << "=";
    ArgTypes[I]->print(OS, /*IsForDebug=*/false, /*NoDetails=*/true);
    if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(ArgValues[I]))
      OS << "@" << CI->getValue();
    OS << ";";
  }
  for (size_t I = 0; I < AuxOperands.size(); ++I) {
    OS << "aux" << I << "=";
    if (auto *CAM = llvm::dyn_cast<llvm::ConstantAsMetadata>(AuxOperands[I])) {
      if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(CAM->getValue())) {
        OS << "ci=" << CI->getValue();
      } else {
        // Best-effort fallback for non-int constants; never hit in current
        // intrinsic set, but keep the encoding total.
        CAM->getValue()->printAsOperand(OS, /*PrintType=*/false);
      }
    } else if (auto *MDS = llvm::dyn_cast<llvm::MDString>(AuxOperands[I])) {
      OS << "s=" << MDS->getString();
    } else {
      // Opaque pointer fallback. Two semantically equivalent MDNodes hash to
      // different values here; not ideal but never triggered today.
      OS << "p=" << AuxOperands[I];
    }
    OS << ";";
  }
  return Out;
}

/// Encode an \c IntrinsicISAStateEffects record as a 3-operand MDNode:
/// \c !{ !{SVAs...}, !{ReadRegs...}, !{WrittenRegs...} }, with each inner
/// list being an MDNode of \c ConstantAsMetadata i32 values. An effects
/// record with all three lists empty is normalized to an empty MDNode.
llvm::MDNode *
encodeEffectsAsMDNode(llvm::LLVMContext &Ctx,
                      const luthier::IntrinsicISAStateEffects &Eff) {
  if (Eff.ReadSVAs.empty() && Eff.ReadPhysRegs.empty() &&
      Eff.WrittenPhysRegs.empty())
    return llvm::MDNode::get(Ctx, {});
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  auto MakeList = [&](auto &&Range) -> llvm::MDNode * {
    llvm::SmallVector<llvm::Metadata *, 4> Ops;
    for (auto V : Range)
      Ops.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(I32, static_cast<unsigned>(V))));
    return llvm::MDNode::get(Ctx, Ops);
  };
  llvm::MDNode *SVANode = MakeList(Eff.ReadSVAs);
  llvm::SmallVector<unsigned, 4> ReadIds, WriteIds;
  for (llvm::MCRegister R : Eff.ReadPhysRegs)
    ReadIds.push_back(R.id());
  for (llvm::MCRegister R : Eff.WrittenPhysRegs)
    WriteIds.push_back(R.id());
  llvm::MDNode *ReadNode = MakeList(ReadIds);
  llvm::MDNode *WriteNode = MakeList(WriteIds);
  return llvm::MDNode::get(Ctx, {SVANode, ReadNode, WriteNode});
}

/// Append a 4-tuple entry \c !{!"<key>", !"<name>", <aux>, <effects>} to the
/// module's \c !luthier.intrinsic.placeholders named-MDNode. The \c aux node
/// is always present (empty MDNode if the intrinsic forwarded no aux data);
/// likewise \c effects is always present (empty MDNode if the intrinsic has
/// no callee-visible ISA-state effects).
void appendPlaceholderNamedMDEntry(
    llvm::Module &M, llvm::StringRef Key, llvm::StringRef IntrinsicName,
    llvm::ArrayRef<llvm::Metadata *> AuxOperands,
    const luthier::IntrinsicISAStateEffects &Effects) {
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::NamedMDNode *NamedMD =
      M.getOrInsertNamedMetadata(luthier::LuthierIntrinsicNamedMDName);
  llvm::Metadata *AuxNode = AuxOperands.empty()
                                ? llvm::MDNode::get(Ctx, {})
                                : llvm::MDNode::get(Ctx, AuxOperands);
  llvm::MDNode *EffNode = encodeEffectsAsMDNode(Ctx, Effects);
  NamedMD->addOperand(llvm::MDNode::get(
      Ctx, {llvm::MDString::get(Ctx, Key),
            llvm::MDString::get(Ctx, IntrinsicName), AuxNode, EffNode}));
}

} // namespace

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
      // Ensure the processor is indeed registered with the Code Generator
      std::optional<IntrinsicProcessor> Processor =
          IntrinsicsProcessors.getProcessorIfRegistered(IntrinsicName);
      if (!Processor.has_value()) {
        IModule.getContext().emitError(
            llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                "Intrinsic {0} is not registered", IntrinsicName))));
        return llvm::PreservedAnalyses::all();
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
            Processor->IRProcessor(F, *CallInst, TM);
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

        // Compute the signature and reuse an existing key for any
        // previously-seen identical invocation; otherwise assign a fresh
        // opaque key and register a named-MD entry.
        llvm::ArrayRef<llvm::Metadata *> AuxOperands =
            IRLoweringInfoOrErr->getExtraLoweringValues();
        std::string Signature = buildIntrinsicSignature(
            IntrinsicName, *ReturnValInfo.Val->getType(), ArgTypes, ArgValues,
            AuxOperands);
        auto [SigIt, Inserted] = SignatureToKey.try_emplace(Signature, "");
        if (Inserted) {
          SigIt->second = (llvm::Twine(LuthierIntrinsicPlaceholderKeyPrefix) +
                           llvm::Twine(NextKeyId++))
                              .str();
          appendPlaceholderNamedMDEntry(IModule, SigIt->second, IntrinsicName,
                                        AuxOperands,
                                        IRLoweringInfoOrErr->getEffects());
        }
        const std::string &Key = SigIt->second;

        // Create the inline-asm placeholder. The opaque key lives in the
        // template-string position (operand 0 of the eventual INLINEASM
        // MachineInstr after ISel), giving the MIR lowering pass a stable
        // handle that survives SelectionDAG
        auto *PlaceHolderInlineAsm = llvm::InlineAsm::get(
            llvm::FunctionType::get(ReturnValInfo.Val->getType(), ArgTypes,
                                    /*isVarArg=*/false),
            Key, ConstraintSS.str(),
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
          luthier::dbgs() << "  Placeholder key: '" << Key << "'";
          if (!AuxOperands.empty())
            luthier::dbgs() << " (" << AuxOperands.size() << " aux value(s))";
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
