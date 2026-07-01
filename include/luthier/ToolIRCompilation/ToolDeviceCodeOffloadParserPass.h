//===-- ToolDeviceCodeOffloadParserPass.h -------------------------*-C++-*-===//
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
/// Implements the companion pass for the \c ToolDeviceCodeOffloadParser,
/// which:
/// - Deletes \c __hip_register* host-side functions to prevent them from ever
/// being registered with the HIP runtime.
/// - Stores the registration info in the appropriate static inline fields of
/// (all) \c ToolDeviceCodeOffloadParser concrete instances declared
/// in the current translation unit.
/// - Also stores synthesized host handles of requested device functions by the
/// CXX compiler plugin into the appropriate static inline field of all
/// \c ToolDeviceCodeOffloadParser concrete instances.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_IR_COMPILATION_TOOL_DEVICE_CODE_OFFLOAD_PARSER_PASS_H
#define LUTHIER_TOOL_IR_COMPILATION_TOOL_DEVICE_CODE_OFFLOAD_PARSER_PASS_H
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
} // namespace llvm

namespace luthier {

/// \brief Companion pass for the \c ToolDeviceCodeOffloadParserPass
class ToolDeviceCodeOffloadParserPass
    : public llvm::PassInfoMixin<ToolDeviceCodeOffloadParserPass> {
public:
  ToolDeviceCodeOffloadParserPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }

  static llvm::StringRef name() {
    return "luthier-tool-device-code-offload-parser-pass";
  }
};

} // namespace luthier

#endif
