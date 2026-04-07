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
#include "luthier/Tooling/InstructionTracesAnalysis.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Tooling/EntryPoint.h"
#include "luthier/Tooling/MemoryAllocationAccessor.h"
#include "luthier/Tooling/PseudoOpcodeAnRegMapper.h"
#include <AMDGPUTargetMachine.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/TargetRegistry.h>
#include <luthier/Tooling/FunctionAnnotations.h>
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

static llvm::Error disassembleTrace(const MemoryAllocationAccessor &SegAccessor,
                                    uint64_t StartDeviceAddr,
                                    llvm::MCDisassembler &Disassembler,
                                    size_t MaxReadSize,
                                    const llvm::MCInstrInfo &MII,
                                    InstructionTraces::Trace &Instructions,
                                    uint64_t &LastInstructionAddr) {
  bool WasTraceTermInstrEncountered{false};
  /// Indicates whether the last instruction disassembled is a basic block
  /// terminator
  uint64_t CurrentDeviceAddress = StartDeviceAddr;

  while (!WasTraceTermInstrEncountered) {
    MemoryAllocationAccessor::AllocationDescriptor AllocDesc;

    LUTHIER_RETURN_ON_ERROR(
        SegAccessor.getAllocationDescriptor(CurrentDeviceAddress)
            .moveInto(AllocDesc));

    if (AllocDesc.empty()) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Address {0:x} has no allocation associated with it",
                        CurrentDeviceAddress));
    }

    uint64_t EntryPointHostAddr =
        CurrentDeviceAddress -
        reinterpret_cast<uint64_t>(AllocDesc.getDeviceAllocation().data()) +
        reinterpret_cast<uint64_t>(AllocDesc.getDeviceAllocation().data());
    size_t SegmentSize = AllocDesc.getSize();

    uint64_t CurrentHostAddr = EntryPointHostAddr;
    uint64_t SegmentHostEndAddr = EntryPointHostAddr + SegmentSize;

    while (!WasTraceTermInstrEncountered &&
           CurrentHostAddr < SegmentHostEndAddr) {
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
      /// Check if the current instruction is a return instruction; Since
      /// the S_SETPC_B64 itself is not a return instruction, we convert it
      /// to the return variant
      uint16_t PseudoOpcode = getPseudoOpcodeFromReal(Inst.getOpcode());
      if (PseudoOpcode == llvm::AMDGPU::S_SETPC_B64) {
        PseudoOpcode = llvm::AMDGPU::S_SETPC_B64_return;
      }
      WasTraceTermInstrEncountered =
          MII.get(PseudoOpcode).isReturn() || MII.get(PseudoOpcode).isCall();
      Instructions.insert(
          {CurrentDeviceAddress,
           std::move(TraceInstr{Inst, CurrentDeviceAddress, InstSize})});
      CurrentDeviceAddress += InstSize;
      CurrentHostAddr += InstSize;
    }
  }

  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<InstructionTraces>>
InstructionTraces::discoverTraces(EntryPoint EP,
                                  const MemoryAllocationAccessor &MemAccessor,
                                  const llvm::TargetMachine &TM,
                                  llvm::MCContext &MCCtx) {
  auto Out = std::unique_ptr<InstructionTraces>(new InstructionTraces(EP));
  size_t MaxInstSize = TM.getMCAsmInfo()->getMaxInstLength();

  const llvm::MCInstrInfo &MII = *TM.getMCInstrInfo();

  uint64_t InitialEntryPointAddr = EP.getEntryPointAddress();

  InstructionAddrSet UnvisitedTraceAddresses{InitialEntryPointAddr};

  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TM.getTarget().createMCDisassembler(*TM.getMCSubtargetInfo(), MCCtx));

  if (!DisAsm) {
    return LUTHIER_MAKE_GENERIC_ERROR("Failed to create an MC Disassembler");
  }

  while (!UnvisitedTraceAddresses.empty()) {
    uint64_t CurrentDeviceAddr = *UnvisitedTraceAddresses.begin();
    auto InstTrace = std::make_unique<Trace>();
    uint64_t TraceDeviceEndAddr{0};

    LUTHIER_RETURN_ON_ERROR(disassembleTrace(MemAccessor, CurrentDeviceAddr,
                                             *DisAsm, MaxInstSize, MII,
                                             *InstTrace, TraceDeviceEndAddr));

    /// Handle direct branch instructions
    for (const auto &[InstAddr, TraceInst] : *InstTrace) {
      const auto &MCInst = TraceInst.getMCInst();
      llvm::MCInstrDesc PseudoOpcodeDesc =
          MII.get(getPseudoOpcodeFromReal(MCInst.getOpcode()));
      bool IsIndirectBranch = PseudoOpcodeDesc.isIndirectBranch();

      if (PseudoOpcodeDesc.isBranch() && !IsIndirectBranch) {
        llvm::Expected<uint64_t> TargetOrErr =
            evaluateBranchOrCallIfDirect(MCInst, InstAddr);
        LUTHIER_RETURN_ON_ERROR(TargetOrErr.takeError());
        Out->DirectBranchTargets.insert(*TargetOrErr);
        /// Find if we have already have the target of this branch in the
        /// current trace; If not, look into the previously discoverd traces; If
        /// it's also not there, add the address to the unvisited list
        if (!InstTrace->contains(*TargetOrErr)) {
          bool HaveVisitedDirectBranchTarget{false};
          for (const auto &[TraceInterval, Trace] : Out->Traces) {
            if (TraceInterval.first <= *TargetOrErr &&
                TraceInterval.second <= *TargetOrErr &&
                Trace->contains(*TargetOrErr)) {
              HaveVisitedDirectBranchTarget = true;
              break;
            }
          }
          if (!HaveVisitedDirectBranchTarget) {
            UnvisitedTraceAddresses.insert(*TargetOrErr);
          }
        }
      }
    };

    /// Put the discovered trace in the map
    Out->Traces.insert({std::make_pair(CurrentDeviceAddr, TraceDeviceEndAddr),
                        std::move(InstTrace)});

    /// Remove the current entry point from the unvisited set
    UnvisitedTraceAddresses.erase(CurrentDeviceAddr);
  }
  return std::move(Out);
}

bool InstructionTracesAnalysis::Result::invalidate(
    llvm::MachineFunction &MF, const llvm::PreservedAnalyses &PA,
    llvm::MachineFunctionAnalysisManager::Invalidator &) {
  // Unless it is invalidated explicitly, it should remain preserved.
  auto PAC = PA.getChecker<InstructionTracesAnalysis>();
  return !PAC.preservedWhenStateless();
}

InstructionTracesAnalysis::Result InstructionTracesAnalysis::run(
    llvm::MachineFunction &TargetMF,
    llvm::MachineFunctionAnalysisManager &TargetMFAM) {
  if (std::optional<EntryPoint> EP =
          getFunctionEntryPoint(TargetMF.getFunction());
      EP.has_value()) {
    llvm::LLVMContext &Ctx = TargetMF.getFunction().getContext();
    llvm::Module &TargetM = *TargetMF.getFunction().getParent();

    auto &TM = *reinterpret_cast<const llvm::GCNTargetMachine *>(
        &TargetMF.getTarget());
    llvm::MCContext &MCCtx = TargetMF.getContext();

    const auto &MAMProxy =
        TargetMFAM.getResult<llvm::ModuleAnalysisManagerMachineFunctionProxy>(
            TargetMF);

    const auto *MAMRes =
        MAMProxy.getCachedResult<MemoryAllocationAnalysis>(TargetM);

    if (!MAMRes) {
      LUTHIER_CTX_EMIT_ON_ERROR(
          Ctx, LUTHIER_MAKE_GENERIC_ERROR(
                   "Memory Allocation Analysis result is not available"));
    }

    const MemoryAllocationAccessor &SegAccessor = MAMRes->getAccessor();

    llvm::Expected<std::unique_ptr<InstructionTraces>> OutOrErr =
        InstructionTraces::discoverTraces(*EP, SegAccessor, TM, MCCtx);

    LUTHIER_CTX_EMIT_ON_ERROR(Ctx, OutOrErr.takeError());

    return Result{std::move(*OutOrErr)};
  } else
    return Result{nullptr};
}
} // namespace luthier
