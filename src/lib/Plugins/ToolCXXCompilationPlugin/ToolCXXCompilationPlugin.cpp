//===-- ToolCXXCompilationPlugin.cpp --------------------------------------===//
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
/// Clang plugin for running frontend actions and attributes required for
/// processing Luthier tool source code.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCXXCompilation/Consumers.h"
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <memory>
#include <string>
#include <vector>

namespace {

class Action : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, llvm::StringRef) override {
    return std::make_unique<luthier::EmitHostHandleForDevFuncConsumer>(CI);
  }

  bool ParseArgs(const clang::CompilerInstance &,
                 const std::vector<std::string> &) override {
    return true;
  }

  /// \c Cmdline (not \c Add*) so the action only runs when explicitly requested
  /// with \c -add-plugin. This lets \c EmitHostHandleForDevFuncConsumer's
  /// pre-parsing stage clear \c AddPluginActions on its cloned invocation and
  /// prevent itself from infinitely recursing into this plugin.
  ActionType getActionType() override { return CmdlineBeforeMainAction; }
};

} // namespace

static clang::FrontendPluginRegistry::Add<Action>
    XAction("luthier-emit-device-function-host-handle",
            "Emits host-side handles for used device functions");
