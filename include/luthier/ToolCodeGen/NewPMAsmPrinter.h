//===-- NewPMAsmPrinter.h -----------------------------------------*-C++-*-===//
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
///
/// \file
/// Defines the \c NewPMAsmPrinter pass, a work-around for printing the assembly
/// using the new PM.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_NEW_PM_ASM_PRINTER_H
#define LUTHIER_TOOL_CODE_GEN_NEW_PM_ASM_PRINTER_H
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CodeGen.h>

namespace llvm {

class TargetMachine;

}

namespace luthier {

class NewPMAsmPrinter : public llvm::RequiredPassInfoMixin<NewPMAsmPrinter> {
private:
  llvm::CodeGenFileType FileType;
  llvm::raw_pwrite_stream &OS;
  bool NoVerify;

public:
  NewPMAsmPrinter(llvm::CodeGenFileType FileType, llvm::raw_pwrite_stream &OS,
                  bool NoVerify = false)
      : FileType(FileType), OS(OS), NoVerify(NoVerify) {};

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif