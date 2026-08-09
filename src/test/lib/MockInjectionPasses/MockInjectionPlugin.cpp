//===-- MockInjectionPlugin.cpp - plugin entry for mock injection passes --===//
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
/// Luthier-style pass-plugin shim that registers all of the mock injection
/// passes with the \c InstrumentationPassBuilder's pipeline parser.
//===----------------------------------------------------------------------===//
#include "MockInjectionPasses.h"
#include "luthier/PassPlugin/LuthierPassPlugin.h"
#include "luthier/ToolCodeGen/InstrumentationPassBuilder.h"
#include "luthier/ToolCodeGen/Prototype.h"

namespace {

template <typename PassT>
bool tryParsePass(llvm::StringRef Name, luthier::PrototypePassManager &PPM) {
  if (Name != PassT::name())
    return false;
  PPM.addPass(PassT());
  return true;
}

/// Registers every mock injection pass with \p PPB's pipeline parser.
///
/// The mock passes run over a whole \c luthier::Prototype, so they are added
/// straight to the \c PrototypePassManager rather than being wrapped in one of
/// the single-module adaptors. \c InstrumentationPassBuilder only consults its
/// parse callbacks with the text inside a \c target(...) or
/// \c instrumentation(...) block, so a mock pass is spelled
/// \c instrumentation(<pass-name>) in \c -passes — an injection pass is what
/// populates the instrumentation module. Only a block naming a single mock pass
/// is claimed here; anything else falls through to the builder's own parsing.
void registerMockInjectionPasses(luthier::InstrumentationPassBuilder &PPB,
                                 void *) {
  PPB.registerPipelineParsingCallback(
      [](llvm::StringRef InnerText, luthier::PrototypePassManager &PPM,
         bool IsTarget) {
        using namespace luthier::test;
        if (IsTarget)
          return false;
        llvm::StringRef Name = InnerText.trim();
        return tryParsePass<MockInjectAtFunctionEntryPass>(Name, PPM) ||
               tryParsePass<MockInjectAtMBBEntryPass>(Name, PPM) ||
               tryParsePass<MockInjectAtMBBTerminatorPass>(Name, PPM) ||
               tryParsePass<MockInjectAtAllVALUPass>(Name, PPM) ||
               tryParsePass<MockInjectAtAllScalarPass>(Name, PPM) ||
               tryParsePass<MockInjectAtOpcodePass>(Name, PPM) ||
               tryParsePass<MockInjectAtAllVGPRDefsWithRegArgPass>(Name, PPM);
      });
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::luthier::PassPluginLibraryInfo
luthierGetPassPluginInfo() {
  return {LUTHIER_PASS_PLUGIN_API_VERSION,
          /*PluginName=*/"luthier-mock-injection-plugin",
          /*PluginVersion=*/LLVM_VERSION_STRING,
          /*ExtraArgs=*/nullptr,
          /*RegisterPrototypePassBuilderCallback=*/registerMockInjectionPasses};
}
