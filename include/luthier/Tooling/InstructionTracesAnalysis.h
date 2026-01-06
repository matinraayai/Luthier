//===-- InstructionTracesAnalysis.h -------------------------------*-C++-*-===//
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
/// \file
/// Describes the \c InstructionTracesAnalysis class which provides access to
/// the list of trace instructions discovered for an entry point (i.e, a
/// machine function in the target module), as well as the address for
/// direct/indirect branches, call instructions, and the direct branch targets.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INSTRUCTION_TRACES_ANALYSIS_H
#define LUTHIER_TOOLING_INSTRUCTION_TRACES_ANALYSIS_H
#include "MachineFunctionEntryPoints.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Common/GenericLuthierError.h>
#include <map>

namespace luthier {

/// \brief A class containing information regarding instructions in a trace
/// obtained during the lifting process
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

/// \brief A \c llvm::MachineFunction analysis that provides the
/// instruction traces discovered from the entry point corresponding to a lifted
/// machine function in the target module
class InstructionTracesAnalysis
    : public llvm::AnalysisInfoMixin<InstructionTracesAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  /// Represents a contiguous list of disassembled trace instructions
  /// We use a small map vector to make sure:
  /// - We map each trace instruction to its device address
  /// - Instructions are inserted in the same order they are disassembled in
  /// the underlying map to provide fast iteration
  using Trace = llvm::SmallMapVector<uint64_t, TraceInstr, 64>;

  /// Holds on to a group of traces discovered for the entry point of the
  /// current machine function
  /// It maps the device (start address, end address) of the trace to itself
  /// This is so that we can check if we have already visited and disassembled
  /// it during the analysis
  using TraceGroup =
      std::map<std::pair<uint64_t, uint64_t>, std::unique_ptr<Trace>>;

  /// Holds a set of addresses
  using InstructionAddrSet = llvm::SmallDenseSet<uint64_t, 16>;

  class Result {
    friend InstructionTracesAnalysis;

    /// Group of traces discovered from the entry point of the machine function
    TraceGroup Traces;

    /// The address of all indirect call instructions discovered in the traces
    InstructionAddrSet IndirectCallInstAddresses;

    /// The address of all indirect branch instructions discovered in the trace
    InstructionAddrSet IndirectBranchAddresses;

    /// The address of all direct branch target instructions discovered in the
    /// trace
    InstructionAddrSet DirectBranchTargets;

    /// The initial entry point of the trace
    EntryPoint EP;

    explicit Result(EntryPoint EP) : EP(EP) {};

  public:
    using trace_group_iterator = TraceGroup::const_iterator;
    using trace_group_reverse_iterator = TraceGroup::const_reverse_iterator;

    [[nodiscard]] trace_group_iterator traces_begin() const {
      return Traces.begin();
    }

    [[nodiscard]] trace_group_iterator traces_end() const {
      return Traces.end();
    }

    [[nodiscard]] trace_group_reverse_iterator traces_rbegin() const {
      return Traces.rbegin();
    }

    [[nodiscard]] trace_group_reverse_iterator traces_rend() const {
      return Traces.rend();
    }

    [[nodiscard]] size_t traces_size() const { return Traces.size(); }

    [[nodiscard]] llvm::iterator_range<trace_group_iterator> traces() const {
      return llvm::make_range(traces_begin(), traces_end());
    }

    [[nodiscard]] llvm::iterator_range<trace_group_reverse_iterator>
    traces_reversed() const {
      return llvm::make_range(traces_rbegin(), traces_rend());
    }

    [[nodiscard]] trace_group_iterator findTrace(uint64_t StartAddress,
                                                 uint64_t EndAddress) const {
      return Traces.find(std::make_pair(StartAddress, EndAddress));
    }

    using indirect_call_addr_iterator = InstructionAddrSet::const_iterator;

    [[nodiscard]] indirect_call_addr_iterator indirect_call_addr_begin() const {
      return IndirectCallInstAddresses.begin();
    }

    [[nodiscard]] indirect_call_addr_iterator indirect_call_addr_end() const {
      return IndirectCallInstAddresses.end();
    }

    [[nodiscard]] llvm::iterator_range<indirect_call_addr_iterator>
    indirect_call_addrs() const {
      return llvm::make_range(indirect_call_addr_begin(),
                              indirect_call_addr_end());
    }

    [[nodiscard]] bool isAddressIndirectCall(uint64_t DevAddr) const {
      return IndirectCallInstAddresses.contains(DevAddr);
    }

    using indirect_branch_addr_iterator = InstructionAddrSet::const_iterator;

    [[nodiscard]] indirect_branch_addr_iterator
    indirect_branch_addr_begin() const {
      return IndirectBranchAddresses.begin();
    }

    [[nodiscard]] indirect_branch_addr_iterator
    indirect_branch_addr_end() const {
      return IndirectBranchAddresses.end();
    }

    [[nodiscard]] llvm::iterator_range<indirect_branch_addr_iterator>
    indirect_branch_addrs() const {
      return llvm::make_range(indirect_branch_addr_begin(),
                              indirect_branch_addr_end());
    }

    [[nodiscard]] bool isAddressIndirectBranch(uint64_t DevAddr) const {
      return IndirectBranchAddresses.contains(DevAddr);
    }

    using direct_branch_target_addr_iterator =
        InstructionAddrSet::const_iterator;

    [[nodiscard]] direct_branch_target_addr_iterator
    direct_branch_target_addr_begin() const {
      return DirectBranchTargets.begin();
    }

    [[nodiscard]] direct_branch_target_addr_iterator
    direct_branch_target_addr_end() const {
      return DirectBranchTargets.end();
    }

    [[nodiscard]] llvm::iterator_range<direct_branch_target_addr_iterator>
    direct_branch_target_addrs() const {
      return llvm::make_range(direct_branch_target_addr_begin(),
                              direct_branch_target_addr_end());
    }

    [[nodiscard]] bool isAddressDirectBranchTarget(uint64_t DevAddr) const {
      return DirectBranchTargets.contains(DevAddr);
    }

    [[nodiscard]] EntryPoint getInitialEntryPoint() const { return EP; }

    bool invalidate(llvm::MachineFunction &, const llvm::PreservedAnalyses &,
                    llvm::MachineFunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  InstructionTracesAnalysis() = default;

  Result run(llvm::MachineFunction &TargetMF,
             llvm::MachineFunctionAnalysisManager &TargetMFAM);
};

} // namespace luthier

#endif