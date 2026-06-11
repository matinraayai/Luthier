//===-- TranslationStateTestPasses.h ------------------------------*-C++-*-===//
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
/// \file TranslationStateTestPasses.h
/// Test-only module passes that exercise the \c TraceIRTranslatorAnalysis
/// dirty-mark/flush cycle from lit pipelines.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TESTING_TRANSLATION_STATE_TEST_PASSES_H
#define LUTHIER_TOOL_CODE_GEN_TESTING_TRANSLATION_STATE_TEST_PASSES_H
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief Test-only pass: marks every MBB of every lifted machine function
/// dirty on the \c TranslationState. Lets lit tests exercise the
/// mark-serialize-flush cycle through the pipeline
struct MarkRetranslateTestPass
    : public llvm::PassInfoMixin<MarkRetranslateTestPass> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

/// \brief Test-only pass: flushes every lifted machine function's translation
struct FlushTranslationTestPass
    : public llvm::PassInfoMixin<FlushTranslationTestPass> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

/// \brief Test-only pass: per MF — flush (initial lift), mark all MBBs, flush
/// again (warm path against the persistent translator). All within one pass
/// run so the pinned \c TranslationState is not recomputed between mark and
/// flush by the pass manager's proxy invalidation
struct WarmMarkFlushTestPass
    : public llvm::PassInfoMixin<WarmMarkFlushTestPass> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif
