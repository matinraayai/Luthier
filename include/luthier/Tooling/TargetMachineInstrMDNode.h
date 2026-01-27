//===-- TargetMachineInstrMDNode.h --------------------------------*-C++-*-===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// \file TargetMachineInstrMDNode.h
/// Describes the \c TargetMachineInstrMDNode used to parse and modify the
/// PC sections metadata used by the target module machine instructions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_TARGET_MACHINE_INSTR_MD_NODE_H
#define LUTHIER_TOOLING_TARGET_MACHINE_INSTR_MD_NODE_H
#include <llvm/IR/Metadata.h>
#include <llvm/Support/Error.h>

namespace llvm {
class MachineInstr;
}

namespace luthier {

class TargetMachineInstrMDNode : public llvm::MDTuple {

public:
  /// Initializes a \c TargetMachineInstrMDNode metadata node and assigns it
  /// to the PC sections of \p MI
  /// \pre MI must belong to a \c llvm::MachineFunction and a \c
  /// llvm::MachineBasicBlock
  /// \returns Expects the newly created \c TargetMachineInstrMDNode tracked
  /// by \p MI
  static llvm::Expected<TargetMachineInstrMDNode &>
  initializeMDNode(llvm::MachineInstr &MI);

  static TargetMachineInstrMDNode &create(llvm::LLVMContext &Ctx);

  static TargetMachineInstrMDNode *
  getInstrMDNodeIfExists(const llvm::MachineInstr &MI);

  void setTraceInstrAddress(llvm::LLVMContext &Ctx, uint64_t Addr);

  [[nodiscard]] std::optional<uint64_t> getTraceInstrAddress() const;

  [[nodiscard]] bool isTraceInstr() const {
    return getTraceInstrAddress().has_value();
  }

  llvm::Error setInjectedPayload(llvm::Function &InjectedPayload);

  [[nodiscard]] llvm::Function *getInjectedPayloadIfExists() const;

  void setCanRelaxDirectBranch(llvm::LLVMContext &Ctx, bool CanRelaxBranch);

  [[nodiscard]] bool canRelaxDirectBranch() const;

  void addIndirectBranchOrCallTarget(llvm::Function &F);

  [[nodiscard]] llvm::SmallVector<llvm::Function *>
  getIndirectBranchAndCallTargets() const;

  void removeIndirectBranchTarget(llvm::Function &F);

  void setIndirectBranchOrCallTargetsResolutionStatus(llvm::LLVMContext &Ctx,
                                                      bool Status);

  bool getIndirectBranchOrCallTargetsResolutionStatus() const;

  static bool classof(const Metadata *MD);
};

} // namespace luthier

#endif