//===-- InstructionTracesAnalysis.cpp -------------------------------------===//
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
/// Implements the \c InstructionTracesAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/tooling/InstructionTracesAnalysis.h"
#include <AMDGPUTargetMachine.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/TargetRegistry.h>
#include <luthier/tooling/EntryPointsAnalysis.h>
#include <luthier/tooling/ExecutableMemorySegmentAccessor.h>
#include <unordered_set>

namespace luthier {

llvm::AnalysisKey InstructionTracesAnalysis::Key;

static llvm::Expected<uint64_t>
evaluateBranchOrCallIfDirect(const llvm::MCInst &Inst, uint64_t Addr) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Inst.getOperand(0).isImm(), "Direct branch's first op is not immediate"));
  int64_t Imm = Inst.getOperand(0).getImm();
  // Our branches take a simm16.
  return llvm::SignExtend64<16>(Imm) * 4 + Addr + 4;
}

llvm::Error disassembleTrace(uint64_t StartHostAddr, uint64_t SegSize,
                             uint64_t StartDeviceAddr,
                             llvm::MCDisassembler &Disassembler,
                             size_t MaxReadSize, const llvm::MCInstrInfo &MII,
                             llvm::DenseMap<uint64_t, TraceInstr> &Instructions,
                             uint64_t &LastInstructionAddr) {
  uint64_t CurrentDeviceAddress = StartDeviceAddr;
  uint64_t CurrentHostAddr = StartHostAddr;
  uint64_t SegmentHostEndAddr = StartHostAddr + SegSize;
  bool WasTraceEndEncountered{false};

  while (WasTraceEndEncountered || CurrentHostAddr >= SegmentHostEndAddr) {
    size_t ReadSize = (CurrentHostAddr + MaxReadSize) < SegmentHostEndAddr
                          ? MaxReadSize
                          : SegmentHostEndAddr - CurrentHostAddr;
    llvm::MCInst Inst;
    size_t InstSize{};
    llvm::ArrayRef ReadBytes = {reinterpret_cast<uint8_t *>(CurrentHostAddr),
                                ReadSize};

    LastInstructionAddr = CurrentDeviceAddress;

    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        Disassembler.getInstruction(Inst, InstSize, ReadBytes,
                                    CurrentDeviceAddress, llvm::nulls()) ==
            llvm::MCDisassembler::Success,
        llvm::formatv("Failed to disassemble instruction at address {0:x}",
                      CurrentDeviceAddress)));
    /// Check if the current instruction is a return
    WasTraceEndEncountered =
        MII.get(getPseudoOpcodeFromReal(Inst.getOpcode())).isReturn();
    Instructions.insert({CurrentDeviceAddress,
                         TraceInstr{Inst, CurrentDeviceAddress, InstSize}});
    CurrentDeviceAddress += InstSize;
    CurrentHostAddr += InstSize;
  }

  return llvm::Error::success();
}

InstructionTracesAnalysis::Result InstructionTracesAnalysis::run(
    llvm::MachineFunction &TargetMF,
    llvm::MachineFunctionAnalysisManager &TargetMFAM) {
  llvm::LLVMContext &Ctx = TargetMF.getFunction().getContext();
  llvm::Module &TargetM = *TargetMF.getFunction().getParent();
  auto &TM =
      *reinterpret_cast<const llvm::GCNTargetMachine *>(&TargetMF.getTarget());
  llvm::MCContext &MCCtx = TargetMF.getContext();

  const auto &MAMProxy =
      TargetMFAM.getResult<llvm::ModuleAnalysisManagerMachineFunctionProxy>(
          TargetMF);

  const ExecutableMemorySegmentAccessor &SegAccessor =
      MAMProxy
          .getCachedResult<ExecutableMemorySegmentAccessorAnalysis>(TargetM)
          ->getAccessor();

  EntryPointsAnalysis::EntryPointType EntryPoint =
      MAMProxy.getCachedResult<EntryPointsAnalysis>(TargetM)
          ->find(TargetMF)
          ->second;

  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TM.getTarget().createMCDisassembler(*TM.getMCSubtargetInfo(), MCCtx));

  size_t MaxInstSize = TM.getMCAsmInfo()->getMaxInstLength();

  const llvm::MCInstrInfo &MII = *TM.getMCInstrInfo();

  uint64_t InitialEntryPointAddr{0};
  if (std::holds_alternative<const llvm::amdhsa::kernel_descriptor_t *>(
          EntryPoint)) {
    const auto *KD =
        std::get<const llvm::amdhsa::kernel_descriptor_t *>(EntryPoint);
    InitialEntryPointAddr =
        reinterpret_cast<uint64_t>(KD) + KD->kernel_code_entry_byte_offset;
  } else {
    InitialEntryPointAddr = std::get<uint64_t>(EntryPoint);
  }

  Result Out;

  InstructionAddrSet UnvisitedTraceAddresses{InitialEntryPointAddr};

  InstructionAddrSet VisitedTraceAddresses{};

  while (!UnvisitedTraceAddresses.empty()) {
    uint64_t CurrentDeviceAddr = *UnvisitedTraceAddresses.begin();
    ExecutableMemorySegmentAccessor::SegmentDescriptor SegDesc;

    LUTHIER_EMIT_ERROR_IN_CONTEXT(
        Ctx, SegAccessor.getSegment(CurrentDeviceAddr).moveInto(SegDesc));

    uint64_t EntryPointHostAddr =
        CurrentDeviceAddr -
        reinterpret_cast<uint64_t>(SegDesc.SegmentOnDevice.data()) +
        reinterpret_cast<uint64_t>(SegDesc.SegmentOnHost.data());
    size_t SegmentSize = SegDesc.SegmentOnHost.size();
    llvm::DenseMap<uint64_t, TraceInstr> InstTrace;
    uint64_t TraceDeviceEndAddr{0};
    LUTHIER_EMIT_ERROR_IN_CONTEXT(
        Ctx, disassembleTrace(EntryPointHostAddr, SegmentSize,
                              InitialEntryPointAddr, *DisAsm, MaxInstSize, MII,
                              InstTrace, TraceDeviceEndAddr));

    /// Add the br
    for (const auto &[InstAddr, TraceInst] : InstTrace) {
      const auto &MCInst = TraceInst.getMCInst();
      llvm::MCInstrDesc PseudoOpcodeDesc =
          MII.get(getPseudoOpcodeFromReal(MCInst.getOpcode()));
      bool IsIndirectBranch = PseudoOpcodeDesc.isIndirectBranch();
      bool IsDirectBranch = PseudoOpcodeDesc.isBranch() && !IsIndirectBranch;
      bool IsCall = PseudoOpcodeDesc.isCall();
      bool IsIndirectCall = IsCall && !MCInst.getOperand(0).isImm();
      bool IsDirectCall = IsCall && !IsIndirectCall;

      if (IsDirectBranch) {
        llvm::Expected<uint64_t> TargetOrErr =
            evaluateBranchOrCallIfDirect(MCInst, InstAddr);
        LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, TargetOrErr.takeError());
        Out.DirectBranchTargets.insert(*TargetOrErr);
        /// Find if we have already have the target of this branch in the
        /// current trace; If not, look into the previously discoverd traces; If
        /// it's also not there, add the address to the unvisited list
        if (!InstTrace.contains(*TargetOrErr)) {
          bool HaveVisitedDirectBranchTarget{false};
          for (const auto &[TraceInterval, Trace] : Out.Traces) {
            if (TraceInterval.first <= *TargetOrErr &&
                TraceInterval.second <= *TargetOrErr &&
                Trace.contains(*TargetOrErr)) {
              HaveVisitedDirectBranchTarget = true;
            }
          }
          if (!HaveVisitedDirectBranchTarget) {
            UnvisitedTraceAddresses.insert(*TargetOrErr);
          }
        }
      }

      if (IsIndirectBranch) {
        Out.IndirectBranchAddresses.insert(InstAddr);
      }
      if (IsIndirectCall) {
        Out.IndirectCallInstAddresses.insert(InstAddr);
      }
      if (IsDirectCall) {
        Out.DirectCallInstAddresses.insert(InstAddr);
      }
    };

    /// Put the discovered trace in the map
    Out.Traces.insert({std::make_pair(CurrentDeviceAddr, TraceDeviceEndAddr),
                       std::move(InstTrace)});

    /// Put the current entry point in the visited set
    VisitedTraceAddresses.insert(CurrentDeviceAddr);
    UnvisitedTraceAddresses.erase(CurrentDeviceAddr);
  }
  return Out;
}
} // namespace luthier
