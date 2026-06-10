//===-- InstructionTracesAnalysis.cpp -------------------------------------===//
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
/// \file InstructionTracesAnalysis.cpp
/// Implements the \c InstructionTracesAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InstructionTracesAnalysis.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/PseudoOpcodeAndRegMapper.h"
#include <AMDGPUTargetMachine.h>
#include <algorithm>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/TargetRegistry.h>
#include <unordered_set>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-instr-traces"

namespace luthier {

llvm::AnalysisKey InstructionTracesAnalysis::Key;

llvm::Expected<uint64_t>
InstructionTracesAnalysis::evaluateDirectBranchOrCall(const llvm::MCInst &Inst,
                                                      uint64_t Addr) {
  const llvm::MCOperand &BrOrCallTargetOp = Inst.getOperand(
      Inst.getOpcode() == llvm::AMDGPU::S_CBRANCH_I_FORK ? 1 : 0);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      BrOrCallTargetOp.isImm(),
      "Direct branch or call's target operand is not immediate"));
  int64_t Imm = Inst.getOperand(0).getImm();
  return llvm::SignExtend64<16>(Imm) * 4 + Addr + 4;
}

uint64_t InstructionTracesAnalysis::evaluateDirectBranchOrCall(int64_t Imm,
                                                               uint64_t Addr) {
  return llvm::SignExtend64<16>(Imm) * 4 + Addr + 4;
}

static llvm::Error disassembleTrace(
    const MemoryAllocationAccessor &SegAccessor, uint64_t StartDeviceAddr,
    llvm::MCDisassembler &Disassembler, size_t MaxReadSize,
    const llvm::MCInstrInfo &MII, llvm::MCInstPrinter *IP,
    InstructionTraces::Trace &Instructions, uint64_t &LastInstructionAddr) {
  /// Indicates whether the last disassembled instruction was a trace terminator
  bool WasTraceTermInstrEncountered{false};

  /// Start adding instructions to the trace
  uint64_t CurrentDeviceAddress = StartDeviceAddr;

  /// This nested while loop will start disassemble instructions from the
  /// trace's start address. It will only terminate if:
  /// - An instruction ending the trace was encountered, i.e return
  /// instructions (S_ENGPGM), call instructions (short and long), unconditional
  /// and/or indirect branches
  /// - The end of the current allocation descriptor is reached, and the next
  /// adjacent address does not have a memory allocation descriptor (i.e. there
  /// is no memory allocation after the end of the already disassembled memory
  /// allocation)
  /// Note that a trace might not have any instructions associated with it if
  /// it is determinted that its starting address does not belong to a memory
  /// allocation
  while (!WasTraceTermInstrEncountered) {
    MemoryAllocationAccessor::AllocationDescriptor AllocDesc;

    LUTHIER_RETURN_ON_ERROR(
        SegAccessor.getAllocationDescriptor(CurrentDeviceAddress)
            .moveInto(AllocDesc));

    if (AllocDesc.empty()) {
      break;
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

      uint16_t PseudoOpcode = getPseudoOpcodeFromReal(Inst.getOpcode());
      llvm::MCInstrDesc PseudoOpcodeDesc = MII.get(PseudoOpcode);
      LLVM_DEBUG(
          luthier::dbgs() << llvm::formatv("[InstructionTraces] Disassembled "
                                           "instruction at {0:x}: ",
                                           CurrentDeviceAddress);
          Inst.dump_pretty(luthier::dbgs(), IP, " ",
                           &Disassembler.getContext());
          luthier::dbgs() << llvm::formatv(", Size: {0} bytes\n", InstSize);
          luthier::dbgs() << llvm::formatv(
              "  Flags: Return={0}, IndirectBranch={1}, Call={2}, "
              "UnconditionalBranch={3}\n",
              PseudoOpcodeDesc.isReturn(), PseudoOpcodeDesc.isIndirectBranch(),
              PseudoOpcodeDesc.isCall(),
              PseudoOpcodeDesc.isUnconditionalBranch()););

      WasTraceTermInstrEncountered =
          PseudoOpcodeDesc.isReturn() || PseudoOpcodeDesc.isCall() ||
          PseudoOpcodeDesc.isIndirectBranch() ||
          PseudoOpcodeDesc.isUnconditionalBranch() ||
          PseudoOpcode == llvm::AMDGPU::S_CBRANCH_G_FORK;
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
  size_t MaxInstSize = TM.getMCAsmInfo().getMaxInstLength();

  const llvm::MCInstrInfo &MII = *TM.getMCInstrInfo();

  uint64_t InitialEntryPointAddr = EP.getEntryPointAddress();

  LLVM_DEBUG(
      luthier::dbgs() << llvm::formatv(
          "[InstructionTraces] Discovering traces for EP from address {0:x}\n",
          InitialEntryPointAddr));

  InstructionAddrSet UnvisitedTraceAddresses{InitialEntryPointAddr};

  std::unique_ptr<llvm::MCDisassembler> DisAsm(
      TM.getTarget().createMCDisassembler(TM.getMCSubtargetInfo(), MCCtx));

  if (!DisAsm) {
    return LUTHIER_MAKE_GENERIC_ERROR("Failed to create an MC Disassembler");
  }

  std::unique_ptr<llvm::MCInstPrinter> IP{nullptr};

  LLVM_DEBUG(IP.reset(TM.getTarget().createMCInstPrinter(
      TM.getTargetTriple(), TM.getMCAsmInfo().getAssemblerDialect(),
      TM.getMCAsmInfo(), *TM.getMCInstrInfo(), TM.getMCRegisterInfo())););

  while (!UnvisitedTraceAddresses.empty()) {
    /// Process the lowest unvisited address first. \c InstructionAddrSet is an
    /// unordered set, but disassembly only moves forward, so taking the minimum
    /// guarantees an enclosing lower-address trace is built before a higher
    /// merge address it subsumes via fall-through is popped — making the
    /// coverage check below order-independent.
    uint64_t CurrentDeviceAddr = *std::min_element(
        UnvisitedTraceAddresses.begin(), UnvisitedTraceAddresses.end());

    /// A branch target can be enqueued before another trace reaches its address
    /// by fall-through: trace groups deliberately run over direct conditional
    /// branches (the fall-through path is reachable and must be part of the
    /// trace), so a trace started at an earlier block can subsume an address
    /// that was queued as a branch target. If this address has already been
    /// disassembled as part of a previously discovered trace, skip it so we
    /// don't create a second, overlapping trace that duplicates the same
    /// instructions. We scan the discovered intervals and confirm the address
    /// is an actual instruction boundary within the covering trace.
    bool AlreadyVisited = false;
    for (const auto &[TraceInterval, Trace] : Out->Traces) {
      if (CurrentDeviceAddr >= TraceInterval.first &&
          CurrentDeviceAddr <= TraceInterval.second &&
          Trace->contains(CurrentDeviceAddr)) {
        AlreadyVisited = true;
        break;
      }
    }
    if (AlreadyVisited) {
      LLVM_DEBUG(
          luthier::dbgs() << llvm::formatv(
              "[InstructionTraces] {0:x} already covered by a discovered "
              "trace, skipping\n",
              CurrentDeviceAddr));
      UnvisitedTraceAddresses.erase(CurrentDeviceAddr);
      continue;
    }

    auto InstTrace = std::make_unique<Trace>();
    uint64_t TraceDeviceEndAddr{0};

    LUTHIER_RETURN_ON_ERROR(
        disassembleTrace(MemAccessor, CurrentDeviceAddr, *DisAsm, MaxInstSize,
                         MII, IP.get(), *InstTrace, TraceDeviceEndAddr));

    /// Handle direct branch and call instructions' targets, check if we have
    /// any targets not covered by current discovered traces
    LLVM_DEBUG(luthier::dbgs()
               << "\n[InstructionTraces] Processing trace from "
               << llvm::formatv("{0:x}", CurrentDeviceAddr) << " - "
               << InstTrace->size() << " instructions\n");

    for (const auto &[InstAddr, TraceInst] : *InstTrace) {
      const auto &MCInst = TraceInst.getMCInst();
      uint16_t Opcode = MCInst.getOpcode();
      const llvm::MCInstrDesc &PseudoOpcodeDesc =
          MII.get(getPseudoOpcodeFromReal(Opcode));

      LLVM_DEBUG(luthier::dbgs()
                     << llvm::formatv("[InstructionTraces] {0:x}: ", InstAddr);
                 TraceInst.getMCInst().dump_pretty(luthier::dbgs(), IP.get(),
                                                   " ", &MCCtx);
                 luthier::dbgs() << "\n");

      bool IsDirectBranch =
          PseudoOpcodeDesc.isBranch() && !PseudoOpcodeDesc.isIndirectBranch() ||
          Opcode == llvm::AMDGPU::S_CBRANCH_I_FORK;

      if (IsDirectBranch) {
        LLVM_DEBUG(luthier::dbgs()
                       << "[InstructionTraces] Direct branch found at "
                       << llvm::formatv("{0:x}", InstAddr) << "\n";);
        llvm::Expected<uint64_t> TargetOrErr =
            InstructionTracesAnalysis::evaluateDirectBranchOrCall(MCInst,
                                                                  InstAddr);
        LUTHIER_RETURN_ON_ERROR(TargetOrErr.takeError());

        LLVM_DEBUG(luthier::dbgs()
                   << "[InstructionTraces] Branch target resolved: "
                   << llvm::formatv("{0:x}\n", *TargetOrErr));

        Out->DirectBranchTargets.insert(*TargetOrErr);
        /// Find if we have already have the target of this branch in the
        /// current trace; If not, look into the previously discoverd traces; If
        /// it's also not there, add the address to the unvisited list
        if (!InstTrace->contains(*TargetOrErr)) {
          bool HaveVisitedDirectBranchTarget{false};
          for (const auto &[TraceInterval, Trace] : Out->Traces) {
            LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                           "[InstructionTraces] Checking trace [{0:x}, "
                           "{1:x}] for branch target {2:x}: Contains={3}\n",
                           TraceInterval.first, TraceInterval.second,
                           *TargetOrErr, Trace->contains(*TargetOrErr)));
            if (Trace->contains(*TargetOrErr)) {
              HaveVisitedDirectBranchTarget = true;
              LLVM_DEBUG(
                  luthier::dbgs()
                  << "[InstructionTraces] Branch target "
                  << llvm::formatv("{0:x} already in current trace, skipping\n",
                                   *TargetOrErr));
              break;
            }
          }
          if (!HaveVisitedDirectBranchTarget) {
            LLVM_DEBUG(
                luthier::dbgs()
                << "[InstructionTraces] Adding branch target "
                << llvm::formatv("{0:x} to unvisited set\n", *TargetOrErr));
            UnvisitedTraceAddresses.insert(*TargetOrErr);
          }
        }
      }
    };

    /// Put the discovered trace in the map if it's not empty
    if (!InstTrace->empty()) {
      LLVM_DEBUG(luthier::dbgs()
                 << "[InstructionTraces] Added trace ["
                 << llvm::formatv("{0:x}, {1:x}]", CurrentDeviceAddr,
                                  TraceDeviceEndAddr)
                 << " with " << InstTrace->size() << " instructions\n");
      Out->Traces.insert({std::make_pair(CurrentDeviceAddr, TraceDeviceEndAddr),
                          std::move(InstTrace)});
    }

    /// Remove the current entry point from the unvisited set
    UnvisitedTraceAddresses.erase(CurrentDeviceAddr);
    LLVM_DEBUG(luthier::dbgs() << "[InstructionTraces] Removed "
                               << llvm::formatv("{0:x} from unvisited set\n",
                                                CurrentDeviceAddr));
  }
  return std::move(Out);
}

bool InstructionTracesAnalysis::Result::invalidate(
    llvm::MachineFunction &, const llvm::PreservedAnalyses &PA,
    llvm::MachineFunctionAnalysisManager::Invalidator &) {
  // Unless it is invalidated explicitly, it should remain preserved.
  auto PAC = PA.getChecker<InstructionTracesAnalysis>();
  return !PAC.preservedWhenStateless();
}

InstructionTracesAnalysis::Result InstructionTracesAnalysis::run(
    llvm::MachineFunction &TargetMF,
    llvm::MachineFunctionAnalysisManager &TargetMFAM) {
  /// We skip any functions that don't have an entry point associated with
  /// them (i.e. functions added manually by Luthier or the tool)
  LLVM_DEBUG(luthier::dbgs() << "[InstructionTraces] Running analysis for "
                             << TargetMF.getName() << "\n";);
  if (std::optional<EntryPoint> EP =
          getFunctionEntryPoint(TargetMF.getFunction());
      EP.has_value()) {

    LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                   "[InstructionTraces] function {0}'s entry point is at "
                   "{1:x}. Entry point is a kernel? {2}.",
                   TargetMF.getName(), EP->getRawAddress(), EP->isKernel()));

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
      Ctx.emitError(llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(
          "Memory Allocation Analysis result is not available")));
      return Result{nullptr};
    }

    const MemoryAllocationAccessor &SegAccessor = MAMRes->getAccessor();

    llvm::Expected<std::unique_ptr<InstructionTraces>> OutOrErr =
        InstructionTraces::discoverTraces(*EP, SegAccessor, TM, MCCtx);

    if (auto Err = OutOrErr.takeError()) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return Result{nullptr};
    }

    LLVM_DEBUG(luthier::dbgs()
                   << "[InstructionTraces] Analysis complete for "
                   << TargetMF.getName() << ": Found "
                   << OutOrErr->get()->traces_size() << " traces, "
                   << OutOrErr->get()->direct_branch_target_addrs_size()
                   << " direct branch targets\n";);

    return Result{std::move(*OutOrErr)};
  } else
    return Result{nullptr};
}
} // namespace luthier
