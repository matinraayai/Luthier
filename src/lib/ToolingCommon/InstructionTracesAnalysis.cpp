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
#include "AMDGPUTargetMachine.h"
#include "LuthierRealToPseudoOpcodeMap.hpp"
#include "LuthierRealToPseudoRegEnumMap.hpp"
#include "luthier/Tooling/MachineFunctionEntryPoints.h"
#include "luthier/Tooling/MemoryAllocationAccessor.h"
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/TargetRegistry.h>
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

llvm::Error disassembleTrace(uint64_t StartHostAddr, uint64_t AllocationSize,
                             uint64_t StartDeviceAddr,
                             llvm::MCDisassembler &Disassembler,
                             size_t MaxReadSize, const llvm::MCInstrInfo &MII,
                             InstructionTracesAnalysis::Trace &Instructions,
                             uint64_t &LastInstructionAddr) {
  uint64_t CurrentDeviceAddress = StartDeviceAddr;
  uint64_t CurrentHostAddr = StartHostAddr;
  uint64_t SegmentHostEndAddr = StartHostAddr + AllocationSize;
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
    Instructions.insert(
        {CurrentDeviceAddress,
         std::move(TraceInstr{Inst, CurrentDeviceAddress, InstSize})});
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

  const auto *MAMRes =
      MAMProxy.getCachedResult<MemoryAllocationAnalysis>(TargetM);

  if (!MAMRes) {
    LUTHIER_CTX_EMIT_ON_ERROR(
        Ctx, LUTHIER_MAKE_GENERIC_ERROR(
                 "Memory Allocation Analysis result is not available"));
  }

  const MemoryAllocationAccessor &SegAccessor = MAMRes->getAccessor();

  const auto *EPRes =
      MAMProxy.getCachedResult<MachineFunctionEntryPoints>(TargetM);

  if (!EPRes) {
    LUTHIER_CTX_EMIT_ON_ERROR(
        Ctx,
        LUTHIER_MAKE_GENERIC_ERROR(
            "Machine Function Entry Points analysis result is not available"));
  }

  auto EPIt = EPRes->find(TargetMF);

  if (EPIt == EPRes->end()) {
    LUTHIER_CTX_EMIT_ON_ERROR(
        Ctx, LUTHIER_MAKE_GENERIC_ERROR(
                 llvm::formatv("Failed to find the entry point associated with "
                               "machine function {0}",
                               TargetMF.getName())));
  }

  EntryPoint EP = EPIt->second;

  /// Disassemble the instructions

  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TM.getTarget().createMCDisassembler(*TM.getMCSubtargetInfo(), MCCtx));

  size_t MaxInstSize = TM.getMCAsmInfo()->getMaxInstLength();

  const llvm::MCInstrInfo &MII = *TM.getMCInstrInfo();

  uint64_t InitialEntryPointAddr = EP.getEntryPointAddress();

  Result Out{EntryPoint{InitialEntryPointAddr}};

  InstructionAddrSet UnvisitedTraceAddresses{InitialEntryPointAddr};

  InstructionAddrSet VisitedTraceAddresses{};

  while (!UnvisitedTraceAddresses.empty()) {
    uint64_t CurrentDeviceAddr = *UnvisitedTraceAddresses.begin();
    MemoryAllocationAccessor::AllocationDescriptor AllocDesc;

    LUTHIER_CTX_EMIT_ON_ERROR(
        Ctx, SegAccessor.getAllocationDescriptor(CurrentDeviceAddr)
                 .moveInto(AllocDesc));

    uint64_t EntryPointHostAddr =
        CurrentDeviceAddr -
        reinterpret_cast<uint64_t>(AllocDesc.AllocationOnDevice.data()) +
        reinterpret_cast<uint64_t>(AllocDesc.AllocationOnHost.data());
    size_t SegmentSize = AllocDesc.AllocationOnHost.size();
    auto InstTrace = std::make_unique<Trace>();
    uint64_t TraceDeviceEndAddr{0};

    LUTHIER_CTX_EMIT_ON_ERROR(
        Ctx, disassembleTrace(EntryPointHostAddr, SegmentSize,
                              InitialEntryPointAddr, *DisAsm, MaxInstSize, MII,
                              *InstTrace, TraceDeviceEndAddr));

    /// Handle branch and call instructions
    for (const auto &[InstAddr, TraceInst] : *InstTrace) {
      const auto &MCInst = TraceInst.getMCInst();
      llvm::MCInstrDesc PseudoOpcodeDesc =
          MII.get(getPseudoOpcodeFromReal(MCInst.getOpcode()));
      bool IsIndirectBranch = PseudoOpcodeDesc.isIndirectBranch();
      bool IsDirectBranch = PseudoOpcodeDesc.isBranch() && !IsIndirectBranch;
      bool IsCall = PseudoOpcodeDesc.isCall();
      bool IsIndirectCall = IsCall && !MCInst.getOperand(0).isImm();
      bool IsDirectCall = IsCall && !IsIndirectCall;

      if (IsDirectBranch || IsDirectCall) {
        llvm::Expected<uint64_t> TargetOrErr =
            evaluateBranchOrCallIfDirect(MCInst, InstAddr);
        LUTHIER_CTX_EMIT_ON_ERROR(Ctx, TargetOrErr.takeError());
        Out.DirectBranchTargets.insert(*TargetOrErr);
        /// Find if we have already have the target of this branch in the
        /// current trace; If not, look into the previously discoverd traces; If
        /// it's also not there, add the address to the unvisited list
        if (!InstTrace->contains(*TargetOrErr)) {
          bool HaveVisitedDirectBranchTarget{false};
          for (const auto &[TraceInterval, Trace] : Out.Traces) {
            if (TraceInterval.first <= *TargetOrErr &&
                TraceInterval.second <= *TargetOrErr &&
                Trace->contains(*TargetOrErr)) {
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
    };

    /// Put the discovered trace in the map
    Out.Traces.insert({std::make_pair(CurrentDeviceAddr, TraceDeviceEndAddr),
                       std::move(InstTrace)});

    /// Remove the current entry point from the unvisited set
    UnvisitedTraceAddresses.erase(CurrentDeviceAddr);
  }
  return Out;
}
} // namespace luthier
