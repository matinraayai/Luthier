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
/// Describes the \c InjectedPayloadAndInstPointAnalysis which maps injected
/// payload functions in the instrumentation module to their
/// corresponding target \c MachineInstr instrumentation points.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_AND_INST_POINT_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_INJECTED_PAYLOAD_AND_INST_POINT_ANALYSIS_H
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class InjectedPayloadAndInstPoint {
private:
  /// Maps each target MI (the \c PATCHPOINT marker) to the ordered list of
  /// injected payload functions to patch at it
  llvm::DenseMap<llvm::MachineInstr *, llvm::SmallVector<llvm::Function *, 2>>
      AppMIToInjectedPayloadsMap;
  /// Inverse map: each injected payload function to its single target MI
  llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
      InjectedPayloadToAppMIMap;
  /// Maps the target module's extern declaration of a payload (the handle
  /// named by the site's \c PATCHPOINT target operand) to the payload's
  /// definition in the instrumentation module
  llvm::DenseMap<llvm::Function *, llvm::Function *>
      ExternHandleToInjectedPayloadMap;
  /// Inverse of \c ExternHandleToInjectedPayloadMap
  llvm::DenseMap<llvm::Function *, llvm::Function *>
      InjectedPayloadToExternHandleMap;

public:
  InjectedPayloadAndInstPoint() = default;

  void addEntry(llvm::MachineInstr &AppMI, llvm::Function &InjectedPayload,
                llvm::Function &ExternHandle) {
    AppMIToInjectedPayloadsMap[&AppMI].push_back(&InjectedPayload);
    InjectedPayloadToAppMIMap.insert({&InjectedPayload, &AppMI});
    ExternHandleToInjectedPayloadMap.insert({&ExternHandle, &InjectedPayload});
    InjectedPayloadToExternHandleMap.insert({&InjectedPayload, &ExternHandle});
  }

  [[nodiscard]] llvm::Function *
  getInjectedPayloadFromExternHandle(const llvm::Function &Extern) const {
    auto It = ExternHandleToInjectedPayloadMap.find(
        const_cast<llvm::Function *>(&Extern));
    return It == ExternHandleToInjectedPayloadMap.end() ? nullptr : It->second;
  }

  [[nodiscard]] llvm::Function *getExternHandleFromInjectedPayload(
      const llvm::Function &InjectedPayload) const {
    auto It = InjectedPayloadToExternHandleMap.find(
        const_cast<llvm::Function *>(&InjectedPayload));
    return It == InjectedPayloadToExternHandleMap.end() ? nullptr : It->second;
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

  bool invalidate(Prototype &P, const llvm::PreservedAnalyses &PA,
                  PrototypeAnalysisManager::Invalidator &PAC);
};

class InjectedPayloadAndInstPointAnalysis
    : public llvm::AnalysisInfoMixin<InjectedPayloadAndInstPointAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<InjectedPayloadAndInstPointAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = InjectedPayloadAndInstPoint;

  InjectedPayloadAndInstPointAnalysis() = default;

  Result run(Prototype &P, PrototypeAnalysisManager &PAM);
};

} // namespace luthier

#endif
