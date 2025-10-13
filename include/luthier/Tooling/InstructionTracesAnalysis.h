//===-- EntryPointTraceGroupAnalysis.h ----------------------------*-C++-*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Describes the \c EntryPointTraceGroupAnalysis class which provides access to
/// the list of instruction traces discovered for an entry point (i.e, a
/// machine function in the target module).
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_DISCOVERED_TRACES_ANALYSIS_H
#define LUTHIER_TOOLING_DISCOVERED_TRACES_ANALYSIS_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <map>

namespace luthier {

class TraceInstr {
private:
  /// The MC representation of the instruction
  const llvm::MCInst Inst;
  /// The address on the GPU Agent this instruction is loaded at
  const uint64_t LoadedDeviceAddress;
  /// Size of the instruction
  const size_t Size;

public:
  /// Constructor
  /// \param Inst \c MCInst of the instruction
  /// \param DeviceAddr the device address this instruction is loaded on
  /// \param Size size of the instruction in bytes
  TraceInstr(llvm::MCInst Inst, uint64_t DeviceAddr, size_t Size)
      : Inst(std::move(Inst)), LoadedDeviceAddress(DeviceAddr), Size(Size) {};

  /// \return the MC representation of the instruction
  [[nodiscard]] const llvm::MCInst &getMCInst() const { return Inst; }

  /// \return the loaded address of this instruction on the device
  [[nodiscard]] uint64_t getLoadedDeviceAddress() const {
    return LoadedDeviceAddress;
  }

  /// \return the size of the instruction in bytes
  [[nodiscard]] size_t getSize() const { return Size; }
};

/// \brief A \c llvm::MachineFunction analysis which provides the instruction
/// trace group discovered from the entry point corresponding to the machine
/// function in the target module
class InstructionTracesAnalysis
    : public llvm::AnalysisInfoMixin<InstructionTracesAnalysis> {
  friend llvm::AnalysisInfoMixin<InstructionTracesAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using InstructionTrace = std::map<uint64_t, TraceInstr>;

  using TraceGroup =
      llvm::DenseMap<std::pair<uint64_t, uint64_t>, InstructionTrace>;

  using InstructionAddrSet = llvm::DenseSet<uint64_t>;

  class Result {
    friend InstructionTracesAnalysis;

    TraceGroup Traces;

    InstructionAddrSet IndirectCallInstAddresses;

    InstructionAddrSet DirectCallInstAddresses;

    InstructionAddrSet IndirectBranchAddresses;

    InstructionAddrSet DirectBranchTargets;

    Result() = default;

  public:
    const TraceGroup &getTraceGroup() const { return Traces; }

    const InstructionAddrSet &getIndirectCallInstAddresses() const {
      return IndirectCallInstAddresses;
    }

    const InstructionAddrSet &getIndirectBranchAddresses() const {
      return IndirectBranchAddresses;
    }

    const InstructionAddrSet &getBranchTargets() const { return DirectBranchTargets; }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  InstructionTracesAnalysis() = default;

  Result run(llvm::MachineFunction &TargetMF,
             llvm::MachineFunctionAnalysisManager &TargetMFAM);
};

} // namespace luthier

#endif