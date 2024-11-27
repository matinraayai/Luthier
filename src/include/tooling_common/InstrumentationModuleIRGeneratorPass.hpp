//===-- InstrumentationModuleIRGeneratorPass.hpp --------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file describes Luthier's <tt>InstrumentationModuleIRGeneratorPass</tt>,
/// in charge of applying the \c InstrumentationTask to an instrumentation
/// module loaded from a LLVM bitcode. It also describes the
/// \c InjectedPayloadAndInstPoint which is the result of
/// <tt>InstrumentationModuleIRGeneratorPass</tt>
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_INSTRUMENTATION_MODULE_IR_GENERATION_PASS_HPP
#define LUTHIER_TOOLING_COMMON_INSTRUMENTATION_MODULE_IR_GENERATION_PASS_HPP
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class InstrumentationTask;

class InjectedPayloadAndInstPoint {
private:
  // A map to keep track of the injected payload functions inside the
  // instrumentation module given the MI they will be patched into
  llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
      AppMIToInjectedPayloadMap;
  // An inverse mapping of the above DenseMap, relating each injected payload
  // function to its target MI in the application
  llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
      InjectedPayloadToAppMIMap;

public:
  InjectedPayloadAndInstPoint() = default;

  void addEntry(llvm::MachineInstr &AppMI, llvm::Function &InjectedPayload) {
    AppMIToInjectedPayloadMap.insert({&AppMI, &InjectedPayload});
    InjectedPayloadToAppMIMap.insert({&InjectedPayload, &AppMI});
  }

  llvm::MachineInstr *at(llvm::Function &InjectedPayload) const {
    return InjectedPayloadToAppMIMap.at(&InjectedPayload);
  }

  llvm::Function *at(llvm::MachineInstr &AppMI) const {
    return AppMIToInjectedPayloadMap.at(&AppMI);
  }

  using mi_payload_const_iterator =
      llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>::const_iterator;

  [[nodiscard]] mi_payload_const_iterator mi_payload_begin() const {
    return AppMIToInjectedPayloadMap.begin();
  }

  [[nodiscard]] mi_payload_const_iterator mi_payload_end() const {
    return AppMIToInjectedPayloadMap.end();
  }

  [[nodiscard]] llvm::iterator_range<mi_payload_const_iterator>
  mi_payload() const {
    return llvm::make_range(mi_payload_begin(), mi_payload_end());
  }

  using payload_mi_const_iterator =
      llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>::const_iterator;

  [[nodiscard]] payload_mi_const_iterator payload_mi_begin() const {
    return InjectedPayloadToAppMIMap.begin();
  }

  [[nodiscard]] payload_mi_const_iterator payload_mi_end() const {
    return InjectedPayloadToAppMIMap.end();
  }

  [[nodiscard]] llvm::iterator_range<payload_mi_const_iterator>
  payload_mi() const {
    return llvm::make_range(payload_mi_begin(), payload_mi_end());
  }

  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

class InstrumentationModuleIRGeneratorPass
    : public llvm::PassInfoMixin<InstrumentationModuleIRGeneratorPass> {
private:

  const InstrumentationTask &Task;

public:
  using Result = InjectedPayloadAndInstPoint;

  explicit InstrumentationModuleIRGeneratorPass(const InstrumentationTask &Task)
      : Task(Task) {};

  /// Run the analysis pass and produce machine module information.
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);
};

} // namespace luthier

#endif