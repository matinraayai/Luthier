//===-- AMDGPUPopulateMachineFunctionsPass.h --------------------*- C++ -*-===//
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
/// Implements the <tt>AMDGPUPopulateMachineFunctionsPass</tt> pass in charge
/// of populating the machine functions inside the target module with
/// lifted machine instructions and basic blocks.
//===----------------------------------------------------------------------===//
#include "LuthierRealToPseudoOpcodeMap.hpp"
#include "LuthierRealToPseudoRegEnumMap.hpp"
#include <SIInstrInfo.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Instrumentation/AMDGPUPopulateMachineFunctionsPass.h>
#include <luthier/Instrumentation/BranchTargetOffsetsAnalysis.h>
#include <luthier/Instrumentation/DisassemblerAnalysis.h>
#include <luthier/Instrumentation/ELFRelocationResolverAnalysisPass.h>
#include <luthier/Instrumentation/GlobalObjectSymbolsAnalysis.h>
#include <luthier/Instrumentation/MIToMCAnalysis.h>
#include <luthier/Instrumentation/Metadata.h>
#include <luthier/Instrumentation/ObjectFileAnalysisPass.h>
#include <luthier/Object/AMDGCNObjectFile.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-populate-machine-functions"

namespace luthier {

static bool shouldReadExec(const llvm::MachineInstr &MI) {
  if (llvm::SIInstrInfo::isVALU(MI)) {
    switch (MI.getOpcode()) {
    case llvm::AMDGPU::V_READLANE_B32:
    case llvm::AMDGPU::SI_RESTORE_S32_FROM_VGPR:
    case llvm::AMDGPU::V_WRITELANE_B32:
    case llvm::AMDGPU::SI_SPILL_S32_TO_VGPR:
      return false;
    }

    return true;
  }

  if (MI.isPreISelOpcode() ||
      llvm::SIInstrInfo::isGenericOpcode(MI.getOpcode()) ||
      llvm::SIInstrInfo::isSALU(MI) || llvm::SIInstrInfo::isSMRD(MI))
    return false;

  return true;
}

static llvm::Error verifyInstruction(llvm::MachineInstrBuilder &Builder) {
  auto &MI = *Builder.getInstr();
  if (shouldReadExec(MI) &&
      !MI.hasRegisterImplicitUseOperand(llvm::AMDGPU::EXEC)) {
    MI.addOperand(
        llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
  }
  return llvm::Error::success();
}

static llvm::Error fixupBitsetInst(llvm::MachineInstr &MI) {
  unsigned int Opcode = MI.getOpcode();
  if (Opcode == llvm::AMDGPU::S_BITSET0_B32 ||
      Opcode == llvm::AMDGPU::S_BITSET0_B64 ||
      Opcode == llvm::AMDGPU::S_BITSET1_B32 ||
      Opcode == llvm::AMDGPU::S_BITSET1_B64) {
    // bitset instructions have a tied def/use that is not reflected in the
    // MC version
    if (MI.getNumOperands() < MI.getNumExplicitOperands()) {
      // Check if the first operand is a register
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          MI.getOperand(0).isReg(),
          "The first operand of a bitset instruction is not a register."));
      // Add the output reg also as the first input, and tie the first and
      // second operands together
      MI.addOperand(
          llvm::MachineOperand::CreateReg(MI.getOperand(0).getReg(), false));
      MI.tieOperands(0, 2);
    } else {
      return llvm::Error::success();
    }
  }
  return llvm::Error::success();
}

static llvm::Error liftFunction(
    llvm::MachineFunction &MF, llvm::Module &Module,
    llvm::MachineModuleInfo &MMI, llvm::ArrayRef<Instr> Instructions,
    const ELFRelocationResolverAnalysisPass::Result &RelocationResolver,
    const BranchTargetOffsetsAnalysis::Result &BranchTargets,
    MIToMCAnalysis::Result *MIToMCMap) {
  auto &F = MF.getFunction();

  LLVM_DEBUG(llvm::dbgs() << "================================================="
                             "=======================\n";
             llvm::dbgs() << "Populating contents of Machine Function "
                          << MF.getName() << "\n");

  auto &TM = MMI.getTarget();

  llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();

  MF.push_back(MBB);
  auto MBBEntry = MBB;

  llvm::MCContext &MCContext = MMI.getContext();

  auto MCInstInfo = TM.getMCInstrInfo();

  llvm::DenseMap<uint64_t,
                 llvm::SmallVector<llvm::MachineInstr *>>
      UnresolvedBranchMIs; // < Set of branch instructions located at a
                           // luthier_address_t waiting for their
                           // target to be resolved after MBBs and MIs
                           // are created
  llvm::DenseMap<uint64_t, llvm::MachineBasicBlock *>
      BranchTargetMBBs; // < Set of MBBs that will be the target of the
                        // UnresolvedBranchMIs
  auto MIA = std::unique_ptr<llvm::MCInstrAnalysis>(
      TM.getTarget().createMCInstrAnalysis(TM.getMCInstrInfo()));

  llvm::SmallVector<llvm::MachineBasicBlock *, 4> MBBs;
  MBBs.push_back(MBB);
  for (unsigned int InstIdx = 0; InstIdx < Instructions.size(); InstIdx++) {
    LLVM_DEBUG(llvm::dbgs() << "+++++++++++++++++++++++++++++++++++++++++++++++"
                               "+++++++++++++++++++++++++\n";);
    const auto &Inst = Instructions[InstIdx];
    auto MCInst = Inst.getMCInst();
    const unsigned Opcode = getPseudoOpcodeFromReal(MCInst.getOpcode());
    const llvm::MCInstrDesc &MCID = MCInstInfo->get(Opcode);
    bool IsDirectBranch = MCID.isBranch() && !MCID.isIndirectBranch();
    bool IsDirectBranchTarget = BranchTargets.contains(Inst.getOffset());
    LLVM_DEBUG(
        llvm::dbgs() << "Lifting and adding MC Inst: ";
        MCInst.dump_pretty(llvm::dbgs(), nullptr, " ", TM.getMCRegisterInfo());
        llvm::dbgs() << "\n";
        llvm::dbgs() << "Instruction idx: " << InstIdx << "\n";
        llvm::dbgs() << llvm::formatv(
            "Loaded address of the instruction: {0:x}\n", Inst.getOffset());
        llvm::dbgs() << llvm::formatv("Is branch? {0}\n", MCID.isBranch());
        llvm::dbgs() << llvm::formatv("Is indirect branch? {0}\n",
                                      MCID.isIndirectBranch()););

    if (IsDirectBranchTarget) {
      LLVM_DEBUG(llvm::dbgs() << "Instruction is a branch target.\n";);
      if (!MBB->empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Current MBB is not empty; Creating a new basic block\n");
        auto OldMBB = MBB;
        MBB = MF.CreateMachineBasicBlock();
        MF.push_back(MBB);
        MBBs.push_back(MBB);
        OldMBB->addSuccessor(MBB);
        // Branch targets mark the beginning of an MBB
        LLVM_DEBUG(llvm::dbgs()
                   << "*********************************************"
                      "***************************\n");
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Current MBB is empty; No new block created "
                                   "for the branch target.\n");
      }
      BranchTargetMBBs.insert({Inst.getOffset(), MBB});
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "Address {0:x} marks the beginning of MBB idx {1}.\n",
                     Inst.getOffset(), MBB->getNumber()););
    }
    llvm::MachineInstrBuilder Builder =
        llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
    if (MIToMCMap) {
      MIToMCMap->insert({*Builder.getInstr(), Inst});
    }

    LLVM_DEBUG(llvm::dbgs() << "Number of operands according to MCID: "
                            << MCID.operands().size() << "\n";
               llvm::dbgs() << "Populating operands\n";);
    for (unsigned OpIndex = 0, E = MCInst.getNumOperands(); OpIndex < E;
         ++OpIndex) {
      const llvm::MCOperand &Op = MCInst.getOperand(OpIndex);
      if (Op.isReg()) {
        LLVM_DEBUG(llvm::dbgs() << "Resolving reg operand.\n");
        unsigned RegNum = RealToPseudoRegisterMapTable(Op.getReg());
        const bool IsDef = OpIndex < MCID.getNumDefs();
        unsigned Flags = 0;
        const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
        if (IsDef && !OpInfo.isOptionalDef()) {
          Flags |= llvm::RegState::Define;
        }
        LLVM_DEBUG(llvm::dbgs()
                       << "Adding register "
                       << llvm::printReg(RegNum,
                                         MF.getSubtarget().getRegisterInfo())
                       << " with flags " << Flags << "\n";);
        (void)Builder.addReg(RegNum, Flags);
      } else if (Op.isImm()) {
        LLVM_DEBUG(llvm::dbgs() << "Resolving an immediate operand.\n");
        // TODO: Resolve immediate load/store operands if they don't have
        // relocations associated with them (e.g. when they happen in the
        // text section)
        uint64_t InstOffset = Inst.getOffset();
        size_t InstSize = Inst.getSize();
        // Check if at any point in the instruction we need to apply
        // relocations
        bool RelocationApplied{false};
        for (uint64_t I = InstOffset; I <= InstOffset + InstSize; ++I) {
          auto RelocationInfo = RelocationResolver.find(I);
          if (RelocationInfo != RelocationResolver.end()) {
            llvm::GlobalObject &TargetSymbol = RelocationInfo->second.first;
            llvm::Expected<long> Addend =
                RelocationInfo->second.second.getAddend();
            LUTHIER_RETURN_ON_ERROR(Addend.takeError());

            uint64_t RelocationType = RelocationInfo->second.second.getType();

            if (llvm::isa<llvm::GlobalVariable>(TargetSymbol)) {
              LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                             "Relocation is being resolved to the global "
                             "variables {0} at offset {1:x}.\n",
                             TargetSymbol.getName(), InstOffset));

              if (RelocationType == llvm::ELF::R_AMDGPU_REL32_LO)
                RelocationType = llvm::SIInstrInfo::MO_GOTPCREL32_LO;
              else if (RelocationType == llvm::ELF::R_AMDGPU_REL32_HI)
                RelocationType = llvm::SIInstrInfo::MO_GOTPCREL32_HI;
              Builder.addGlobalAddress(&TargetSymbol, *Addend, RelocationType);
            } else if (auto *TargetAsFunc =
                           llvm::dyn_cast<llvm::Function>(&TargetSymbol);
                       TargetAsFunc && TargetAsFunc->getCallingConv() !=
                                           llvm::CallingConv::AMDGPU_KERNEL) {

              // Add the function as the operand
              if (RelocationType == llvm::ELF::R_AMDGPU_REL32_LO)
                RelocationType = llvm::SIInstrInfo::MO_REL32_LO;
              if (RelocationType == llvm::ELF::R_AMDGPU_REL32_HI)
                RelocationType = llvm::SIInstrInfo::MO_REL32_HI;
              Builder.addGlobalAddress(TargetAsFunc, *Addend, RelocationType);
            } else {
              // For now, we don't handle calling kernels from kernels
              llvm_unreachable("not implemented");
            }
            RelocationApplied = true;
            break;
          }
        }
        if (!RelocationApplied && !IsDirectBranch) {
          LLVM_DEBUG(
              llvm::dbgs()
                  << "Relocation was not applied for the "
                     "immediate operand, and it is not a direct branch.\n";
              llvm::dbgs() << "Adding the immediate operand directly to the "
                              "instruction\n");
          if (llvm::SIInstrInfo::isSOPK(*Builder)) {
            LLVM_DEBUG(llvm::dbgs() << "Instruction is in SOPK format\n");
            if (llvm::SIInstrInfo::sopkIsZext(Opcode)) {
              auto Imm = static_cast<uint16_t>(Op.getImm());
              LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                             "Adding truncated imm value: {0}\n", Imm));
              (void)Builder.addImm(Imm);
            } else {
              auto Imm = static_cast<int16_t>(Op.getImm());
              LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                             "Adding truncated imm value: {0}\n", Imm));
              (void)Builder.addImm(Imm);
            }
          } else {
            LLVM_DEBUG(llvm::dbgs()
                       << llvm::formatv("Adding Imm: {0}\n", Op.getImm()));
            (void)Builder.addImm(Op.getImm());
          }
        }

      } else if (!Op.isValid()) {
        llvm_unreachable("Operand is not set");
      } else {
        // TODO: implement floating point operands
        llvm_unreachable("Not yet implemented");
      }
    }
    // Create a (fake) memory operand to keep the machine verifier happy
    // when encountering image instructions
    if (llvm::SIInstrInfo::isImage(*Builder)) {
      llvm::MachinePointerInfo PtrInfo =
          llvm::MachinePointerInfo::getConstantPool(MF);
      auto *MMO = MF.getMachineMemOperand(
          PtrInfo,
          MCInstInfo->get(Builder->getOpcode()).mayLoad()
              ? llvm::MachineMemOperand::MOLoad
              : llvm::MachineMemOperand::MOStore,
          16, llvm::Align(8));
      Builder->addMemOperand(MF, MMO);
    }

    if (MCInst.getNumOperands() < MCID.NumOperands) {
      LLVM_DEBUG(llvm::dbgs() << "Must fixup instruction ";
                 Builder->print(llvm::dbgs()); llvm::dbgs() << "\n";
                 llvm::dbgs() << "Num explicit operands added so far: "
                              << MCInst.getNumOperands() << "\n";
                 llvm::dbgs() << "Num explicit operands according to MCID: "
                              << MCID.NumOperands << "\n";);
      // Loop over missing explicit operands (if any) and fixup any missing
      for (unsigned int MissingExpOpIdx = MCInst.getNumOperands();
           MissingExpOpIdx < MCID.NumOperands; MissingExpOpIdx++) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Fixing up operand no " << MissingExpOpIdx << "\n";);
        auto OpType = MCID.operands()[MissingExpOpIdx].OperandType;
        if (OpType == llvm::MCOI::OPERAND_IMMEDIATE ||
            OpType == llvm::AMDGPU::OPERAND_KIMM32) {
          LLVM_DEBUG(llvm::dbgs() << "Added a 0-immediate operand.\n";);
          Builder.addImm(0);
        }
      }
    }

    LUTHIER_RETURN_ON_ERROR(fixupBitsetInst(*Builder));
    LUTHIER_RETURN_ON_ERROR(verifyInstruction(Builder));
    LLVM_DEBUG(llvm::dbgs() << "Final form of the instruction (not final if "
                               "it's a direct branch): ";
               Builder->print(llvm::dbgs()); llvm::dbgs() << "\n");
    // Basic Block resolving
    if (MCID.isTerminator()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Instruction is a terminator; Finishing basic block.\n");
      if (IsDirectBranch) {
        LLVM_DEBUG(llvm::dbgs() << "The terminator is a direct branch.\n");
        uint64_t BranchTarget;
        if (MIA->evaluateBranch(MCInst, Inst.getOffset(), 4, BranchTarget)) {
          LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                         "Address was resolved to {0:x}\n", BranchTarget));
          if (!UnresolvedBranchMIs.contains(BranchTarget)) {
            UnresolvedBranchMIs.insert({BranchTarget, {Builder.getInstr()}});
          } else {
            UnresolvedBranchMIs[BranchTarget].push_back(Builder.getInstr());
          }
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Error resolving the target address of the branch\n");
        }
      }
      // if this is the last instruction in the stream, no need for creating
      // a new basic block
      if (InstIdx != Instructions.size() - 1) {
        LLVM_DEBUG(llvm::dbgs() << "Creating a new basic block.\n");
        auto OldMBB = MBB;
        MBB = MF.CreateMachineBasicBlock();
        MBBs.push_back(MBB);
        MF.push_back(MBB);
        // Don't add the next block to the list of successors if the
        // terminator is an unconditional branch
        if (!MCID.isUnconditionalBranch())
          OldMBB->addSuccessor(MBB);
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "Address {0:x} marks the beginning of MBB idx {1}.\n",
                       Inst.getOffset(), MBB->getNumber()););
      }
      LLVM_DEBUG(llvm::dbgs() << "*********************************************"
                                 "***************************\n");
    }
  }

  // Resolve the branch and target MIs/MBBs
  LLVM_DEBUG(llvm::dbgs() << "Resolving direct branch MIs\n");
  for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
    LLVM_DEBUG(
        llvm::dbgs() << llvm::formatv(
            "Resolving MIs jumping to target address {0:x}.\n", TargetAddress));
    MBB = BranchTargetMBBs[TargetAddress];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MBB != nullptr,
        "Failed to find the MachineBasicBlock associated with "
        "the branch target address {0:x}.",
        TargetAddress));
    for (auto &MI : BranchMIs) {
      LLVM_DEBUG(llvm::dbgs() << "Resolving branch for the instruction ";
                 MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
      MI->addOperand(llvm::MachineOperand::CreateMBB(MBB));
      MI->getParent()->addSuccessor(MBB);
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "MBB {0:x} {1} was set as the target of the branch.\n",
                     MBB, MBB->getName()));
      LLVM_DEBUG(llvm::dbgs() << "Final branch instruction: ";
                 MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "*********************************************"
                             "***************************\n");

  MF.getRegInfo().freezeReservedRegs();

  // Populate the properties of MF
  llvm::MachineFunctionProperties &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Properties.set(llvm::MachineFunctionProperties::Property::TracksLiveness);
  Properties.set(llvm::MachineFunctionProperties::Property::Selected);

  LLVM_DEBUG(llvm::dbgs() << "Final form of the Machine function:\n";
             MF.print(llvm::dbgs());
             llvm::dbgs() << "\n"
                          << "*********************************************"
                             "***************************\n";);
  return llvm::Error::success();
}

llvm::PreservedAnalyses AMDGPUPopulateMachineFunctionsPass::run(
    llvm::MachineFunction &MF, llvm::MachineFunctionAnalysisManager &MFAM) {
  llvm::Function &Function = MF.getFunction();
  llvm::LLVMContext &Ctx = Function.getContext();
  llvm::Module &TargetModule = *Function.getParent();
  auto &MAMProxy =
      MFAM.getResult<llvm::ModuleAnalysisManagerMachineFunctionProxy>(MF);
  if (!MAMProxy.cachedResultExists<ObjectFileAnalysisPass>(TargetModule))
    Ctx.emitError("The module object file analysis doesn't exist");
  /// Get the AMDGCN object file
  const llvm::object::ObjectFile &ObjFile =
      MAMProxy.getCachedResult<ObjectFileAnalysisPass>(TargetModule)
          ->getObject();
  /// Get the MMI
  if (!MAMProxy.cachedResultExists<llvm::MachineModuleAnalysis>(TargetModule))
    Ctx.emitError("The machine module analysis doesn't exist");
  llvm::MachineModuleInfo &MMI =
      MAMProxy.getCachedResult<llvm::MachineModuleAnalysis>(TargetModule)
          ->getMMI();

  if (!llvm::isa<luthier::object::AMDGCNObjectFile>(ObjFile))
    Ctx.emitError("Object file is not an amdgcn object");
  auto &AMDGCNObjFile = llvm::cast<luthier::object::AMDGCNObjectFile>(ObjFile);

  if (!MAMProxy.cachedResultExists<GlobalObjectSymbolsAnalysis>(TargetModule))
    Ctx.emitError("Global value symbols analysis doesn't exist");
  /// Get the global value symbols
  auto &GlobalValueSymbols =
      *MAMProxy.getCachedResult<GlobalObjectSymbolsAnalysis>(TargetModule);

  /// Get the disassembler and disassemble its instructions
  llvm::ArrayRef<Instr> Instructions =
      MFAM.getResult<DisassemblerAnalysis>(MF).getInstructions();

  /// Get the relocation resolver
  LUTHIER_EMIT_ERROR_IN_CONTEXT(
      Ctx, LUTHIER_GENERIC_ERROR_CHECK(
               MAMProxy.cachedResultExists<ELFRelocationResolverAnalysisPass>(
                   TargetModule),
               "Relocation resolver is not available"));
  auto &RelocationResolver =
      *MAMProxy.getCachedResult<ELFRelocationResolverAnalysisPass>(
          TargetModule);

  std::optional<llvm::object::SymbolRef> FuncSymbol =
      GlobalValueSymbols.getSymbolRef(Function);
  if (!FuncSymbol.has_value())
    Ctx.emitError(llvm::formatv("Failed to obtain the symbol for function {0}",
                                Function.getName()));

  auto &BranchTargets = MFAM.getResult<BranchTargetOffsetsAnalysis>(MF);

  auto *MIToMCMap = MFAM.isPassRegistered<MIToMCAnalysis>()
                        ? MFAM.getResult<MIToMCAnalysis>(MF)
                        : nullptr;

  LUTHIER_EMIT_ERROR_IN_CONTEXT(
      Ctx, liftFunction(TargetModule, MF, Instructions, RelocationResolver,
                        BranchTargets, MIToMCMap));

  if (Function.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
    // Manually set the stack frame size if the function is a kernel
    auto &KernelMD =
        MAMProxy
            .getCachedResult<AMDGCNMetadataParserAnalysisPass>(TargetModule)
            ->getKernelMetadata();
    // TODO: dynamic stack kernels
    LLVM_DEBUG(llvm::dbgs() << "Stack size according to the metadata: "
                            << KernelMD.PrivateSegmentFixedSize << "\n");
    if (KernelMD.PrivateSegmentFixedSize != 0) {
      MF.getFrameInfo().CreateFixedObject(KernelMD.PrivateSegmentFixedSize, 0,
                                          true);
      MF.getFrameInfo().setStackSize(KernelMD.PrivateSegmentFixedSize);
      LLVM_DEBUG(llvm::dbgs()
                 << "Stack size: " << MF.getFrameInfo().getStackSize() << "\n");
    }
  }
  return llvm::PreservedAnalyses::all();
}
} // namespace luthier