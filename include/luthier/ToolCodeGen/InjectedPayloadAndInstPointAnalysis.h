//===-- InjectedPayloadAndInstPointAnalysis.h -------------------*- C++ -*-===//
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
/// This file describes the \c InjectedPayloadAndInstPointAnalysis which
/// maps injected payload functions in the instrumentation module to their
/// corresponding target \c MachineInstr instrumentation points.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_AND_INST_POINT_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_AND_INST_POINT_ANALYSIS_H
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class InjectedPayloadAndInstPoint {
private:
  /// Maps each target MI to the ordered list of injected payload functions
  /// that will be patched before it
  llvm::DenseMap<llvm::MachineInstr *, llvm::SmallVector<llvm::Function *, 2>>
      AppMIToInjectedPayloadsMap;
  /// Inverse map: each injected payload function to its single target MI
  llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
      InjectedPayloadToAppMIMap;

public:
  InjectedPayloadAndInstPoint() = default;

  void addEntry(llvm::MachineInstr &AppMI, llvm::Function &InjectedPayload) {
    AppMIToInjectedPayloadsMap[&AppMI].push_back(&InjectedPayload);
    InjectedPayloadToAppMIMap.insert({&InjectedPayload, &AppMI});
  }

  [[nodiscard]] llvm::MachineInstr *
  at(const llvm::Function &InjectedPayload) const {
    return InjectedPayloadToAppMIMap.at(&InjectedPayload);
  }

  [[nodiscard]] unsigned int size() const {
    return InjectedPayloadToAppMIMap.size();
  }

  [[nodiscard]] bool contains(const llvm::Function &InjectedPayload) const {
    return InjectedPayloadToAppMIMap.contains(&InjectedPayload);
  }

  [[nodiscard]] llvm::ArrayRef<llvm::Function *>
  at(const llvm::MachineInstr &AppMI) const {
    return AppMIToInjectedPayloadsMap.at(&AppMI);
  }

  [[nodiscard]] bool contains(const llvm::MachineInstr &AppMI) const {
    return AppMIToInjectedPayloadsMap.contains(&AppMI);
  }

  using mi_payloads_const_iterator =
      llvm::DenseMap<llvm::MachineInstr *,
                     llvm::SmallVector<llvm::Function *, 2>>::const_iterator;

  [[nodiscard]] mi_payloads_const_iterator mi_payloads_begin() const {
    return AppMIToInjectedPayloadsMap.begin();
  }

  [[nodiscard]] mi_payloads_const_iterator mi_payloads_end() const {
    return AppMIToInjectedPayloadsMap.end();
  }

  [[nodiscard]] llvm::iterator_range<mi_payloads_const_iterator>
  mi_payloads() const {
    return llvm::make_range(mi_payloads_begin(), mi_payloads_end());
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

  bool invalidate(llvm::Module &IModule, const llvm::PreservedAnalyses &PA,
                  llvm::ModuleAnalysisManager::Invalidator &PAC);
};

class InjectedPayloadAndInstPointAnalysis
    : public llvm::AnalysisInfoMixin<InjectedPayloadAndInstPointAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<InjectedPayloadAndInstPointAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = InjectedPayloadAndInstPoint;

  InjectedPayloadAndInstPointAnalysis() = default;

  Result run(llvm::Module &IModule, llvm::ModuleAnalysisManager &);
};

} // namespace luthier

#endif
