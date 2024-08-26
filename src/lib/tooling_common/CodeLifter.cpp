//===-- CodeLifter.cpp - Luthier's Code Lifter  ---------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// This file implements Luthier's Code Lifter.
//===----------------------------------------------------------------------===//
#include "tooling_common/CodeLifter.hpp"

#include "LuthierRealToPseudoOpcodeMap.hpp"

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/BinaryFormat/MsgPackDocument.h>
#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCFixupKindInfo.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSymbolELF.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/RelocationResolver.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>

#include <SIRegisterInfo.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <memory>

#include "common/Error.hpp"
#include "common/ObjectUtils.hpp"
#include "hsa/Executable.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/ISA.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "hsa/hsa.hpp"
#include "luthier/hsa/Instr.h"
#include "luthier/hsa/KernelDescriptor.h"
#include "luthier/types.h"
#include "tooling_common/TargetManager.hpp"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-lifter"

namespace luthier {

// TODO: Merge this fix to upstream LLVM
bool evaluateBranch(const llvm::MCInst &Inst, uint64_t Addr, uint64_t Size,
                    uint64_t &Target) {
  if (!Inst.getOperand(0).isImm())
    return false;
  int64_t Imm = Inst.getOperand(0).getImm();
  // Our branches take a simm16, but we need two extra bits to account for
  // the factor of 4.
  llvm::APInt SignedOffset(18, Imm * 4, true);
  Target = (SignedOffset.sext(64) + Addr + 4).getZExtValue();
  return true;
}

template <> CodeLifter *Singleton<CodeLifter>::Instance{nullptr};

llvm::Error CodeLifter::invalidateCachedExecutableItems(hsa::Executable &Exec) {
  // Remove its lifted representation
  LiftedExecutables.erase(Exec);
  llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));

  for (const auto &LCO : LCOs) {
    // Remove the branch and branch target locations
    if (DirectBranchTargetLocations.contains(LCO))
      DirectBranchTargetLocations.erase(LCO);
    // Remove relocation info
    if (Relocations.contains(LCO)) {
      Relocations.erase(LCO);
    }

    llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *> Symbols;
    LUTHIER_RETURN_ON_ERROR(LCO.getLoadedCodeObjectSymbols(Symbols));
    for (const auto &Symbol : Symbols) {
      // Remove the disassembled hsa::Instr of each hsa::ExecutableSymbol
      if (MCDisassembledSymbols.contains(Symbol))
        MCDisassembledSymbols.erase(Symbol);
      // Remove its lifted representation if this is a kernel
      const auto *SymbolAsKernel =
          llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Symbol);
      if (LiftedKernelSymbols.contains(SymbolAsKernel))
        LiftedKernelSymbols.erase(SymbolAsKernel);
    }
  }
  return llvm::Error::success();
}

llvm::Expected<CodeLifter::DisassemblyInfo &>
luthier::CodeLifter::getDisassemblyInfo(const hsa::ISA &ISA) {
  if (!DisassemblyInfoMap.contains(ISA)) {
    auto TargetInfo = TargetManager::instance().getTargetInfo(ISA);
    LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

    auto TT = ISA.getTargetTriple();
    LUTHIER_RETURN_ON_ERROR(TT.takeError());

    std::unique_ptr<llvm::MCContext> CTX(new (std::nothrow) llvm::MCContext(
        llvm::Triple(*TT), TargetInfo->getMCAsmInfo(),
        TargetInfo->getMCRegisterInfo(), TargetInfo->getMCSubTargetInfo()));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CTX != nullptr));

    std::unique_ptr<llvm::MCDisassembler> DisAsm(
        TargetInfo->getTarget()->createMCDisassembler(
            *(TargetInfo->getMCSubTargetInfo()), *CTX));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(DisAsm != nullptr));

    DisassemblyInfoMap.insert(
        {ISA, DisassemblyInfo{std::move(CTX), std::move(DisAsm)}});
  }
  return DisassemblyInfoMap[ISA];
}

bool CodeLifter::isAddressDirectBranchTarget(const hsa::LoadedCodeObject &LCO,
                                             address_t Address) {
  if (!DirectBranchTargetLocations.contains(LCO)) {
    return false;
  }
  auto &AddressInfo = DirectBranchTargetLocations[LCO];
  return AddressInfo.contains(Address);
}

void luthier::CodeLifter::addDirectBranchTargetAddress(
    const hsa::LoadedCodeObject &LCO, address_t Address) {
  if (!DirectBranchTargetLocations.contains(LCO)) {
    DirectBranchTargetLocations.insert(
        {LCO, llvm::DenseSet<luthier::address_t>{}});
  }
  DirectBranchTargetLocations[LCO].insert(Address);
}

llvm::Expected<
    std::pair<std::vector<llvm::MCInst>, std::vector<luthier::address_t>>>
CodeLifter::disassemble(const hsa::ISA &ISA, llvm::ArrayRef<uint8_t> Code) {
  auto DisassemblyInfo = getDisassemblyInfo(ISA);
  LUTHIER_RETURN_ON_ERROR(DisassemblyInfo.takeError());

  auto TargetInfo = TargetManager::instance().getTargetInfo(ISA);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());
  const auto &DisAsm = DisassemblyInfo->DisAsm;

  size_t MaxReadSize = TargetInfo->getMCAsmInfo()->getMaxInstLength();
  size_t Idx = 0;
  luthier::address_t CurrentAddress = 0;
  std::vector<llvm::MCInst> Instructions;
  std::vector<luthier::address_t> Addresses;

  while (Idx < Code.size()) {
    size_t ReadSize =
        (Idx + MaxReadSize) < Code.size() ? MaxReadSize : Code.size() - Idx;
    size_t InstSize{};
    llvm::MCInst Inst;
    auto ReadBytes =
        arrayRefFromStringRef(toStringRef(Code).substr(Idx, ReadSize));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        DisAsm->getInstruction(Inst, InstSize, ReadBytes, CurrentAddress,
                               llvm::nulls()) ==
        llvm::MCDisassembler::Success));

    Addresses.push_back(CurrentAddress);
    Idx += InstSize;
    CurrentAddress += InstSize;
    Instructions.push_back(Inst);
  }

  return std::make_pair(Instructions, Addresses);
}

llvm::Expected<const std::vector<hsa::Instr> &>
luthier::CodeLifter::disassemble(const hsa::LoadedCodeObjectSymbol &Symbol) {
  if (!MCDisassembledSymbols.contains(&Symbol)) {
    auto SymbolType = Symbol.getType();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(
        SymbolType == hsa::LoadedCodeObjectSymbol::SK_DEVICE_FUNCTION ||
        SymbolType == hsa::LoadedCodeObjectSymbol::SK_KERNEL));

    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());

    // The ISA associated with the Symbol is
    auto LCO = hsa::LoadedCodeObject(Symbol.getLoadedCodeObject());

    auto Agent = Symbol.getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());

    auto ISA = LCO.getISA();
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());

    auto MachineCodeOnDevice = Symbol.getLoadedSymbolContents();
    LUTHIER_RETURN_ON_ERROR(MachineCodeOnDevice.takeError());
    auto MachineCodeOnHost = hsa::convertToHostEquivalent(*MachineCodeOnDevice);
    LUTHIER_RETURN_ON_ERROR(MachineCodeOnHost.takeError());

    auto InstructionsAndAddresses = disassemble(*ISA, *MachineCodeOnHost);
    LUTHIER_RETURN_ON_ERROR(InstructionsAndAddresses.takeError());
    auto [Instructions, Addresses] = *InstructionsAndAddresses;

    auto &Out =
        MCDisassembledSymbols
            .insert({&Symbol, std::make_unique<std::vector<hsa::Instr>>()})
            .first->getSecond();
    Out->reserve(Instructions.size());

    auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
    LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

    auto MII = TargetInfo->getMCInstrInfo();
    auto MIA = TargetInfo->getMCInstrAnalysis();

    auto BaseLoadedAddress =
        reinterpret_cast<luthier::address_t>(MachineCodeOnDevice->data());

    luthier::address_t PrevInstAddress = BaseLoadedAddress;

    auto Executable = Symbol.getExecutable();
    LUTHIER_RETURN_ON_ERROR(Executable.takeError());

    for (unsigned int I = 0; I < Instructions.size(); ++I) {
      auto &Inst = Instructions[I];
      auto Address = Addresses[I] + BaseLoadedAddress;
      auto Size = Address - PrevInstAddress;
      if (MII->get(Inst.getOpcode()).isBranch()) {
        LLVM_DEBUG(llvm::dbgs() << "Instruction "; Inst.dump_pretty(
            llvm::dbgs(), TargetInfo->getMCInstPrinter(), " ",
            TargetInfo->getMCRegisterInfo());
                   llvm::dbgs() << llvm::formatv(
                       " at idx {0}, address {1:x}, size {2} is a branch.\n", I,
                       Address, Size););
        //        if (!isAddressBranchOrBranchTarget(*LCO, Address)) {
        LLVM_DEBUG(llvm::dbgs() << "Evaluating its target.\n");
        luthier::address_t Target;
        if (evaluateBranch(Inst, Address, Size, Target)) {
          LLVM_DEBUG(
              llvm::dbgs() << llvm::formatv(
                  "Evaluated address {0:x} as the branch target\n", Target););
          addDirectBranchTargetAddress(LCO, Target);
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Failed to evaluate the branch target.\n");
        }
      }
      PrevInstAddress = Address;
      Out->push_back(hsa::Instr(Inst, Symbol, Address, Size));
    }
  }
  return *MCDisassembledSymbols.at(&Symbol);
}

llvm::Expected<std::optional<CodeLifter::LCORelocationInfo>>
CodeLifter::resolveRelocation(const hsa::LoadedCodeObject &LCO,
                              luthier::address_t Address) {
  if (!Relocations.contains(LCO)) {
    // If the LCO doesn't have its relocation info cached, calculate it
    auto LoadedMemory = LCO.getLoadedMemory();
    LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

    auto LoadedMemoryBase = reinterpret_cast<address_t>(LoadedMemory->data());

    auto StorageELF = LCO.getStorageELF();
    LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

    // Create an entry for the LCO in the relocations map
    auto &LCORelocationsMap =
        Relocations
            .insert({LCO, llvm::DenseMap<address_t, LCORelocationInfo>{}})
            .first->getSecond();

    for (const auto &Section : StorageELF->sections()) {
      for (const llvm::object::ELFRelocationRef Reloc : Section.relocations()) {
        // Only rely on the loaded address of the symbol instead of its name
        // The name will be stripped from the relocation section
        // if the symbol has a private linkage (i.e. device functions)
        auto RelocSymbolLoadedAddress = Reloc.getSymbol()->getAddress();
        LUTHIER_RETURN_ON_ERROR(RelocSymbolLoadedAddress.takeError());
        LLVM_DEBUG(LUTHIER_RETURN_ON_MOVE_INTO_FAIL(
                       llvm::StringRef, SymName, Reloc.getSymbol()->getName());
                   llvm::dbgs() << llvm::formatv(
                       "Found relocation for symbol {0} at address {1:x}.\n",
                       SymName, LoadedMemoryBase + *RelocSymbolLoadedAddress));
        // Check with the hsa::Platform which HSA executable Symbol this
        // address is associated with
        auto RelocSymbol = hsa::LoadedCodeObjectSymbol::fromLoadedAddress(
            LoadedMemoryBase + *RelocSymbolLoadedAddress);
        LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(RelocSymbol != nullptr));
        // The target address will be the base of the loaded
        luthier::address_t TargetAddress = LoadedMemoryBase + Reloc.getOffset();
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "Relocation found for symbol {0} at address {1:x} for "
                       "LCO {2:x}.\n",
                       llvm::cantFail(RelocSymbol->getName()), TargetAddress,
                       LCO.hsaHandle()));
        LCORelocationsMap.insert({TargetAddress, {*RelocSymbol, Reloc}});
      }
    }
  }
  // Querying actually begins here
  const auto &LCORelocationsMap = Relocations.at(LCO);
  LLVM_DEBUG(
      llvm::dbgs() << llvm::formatv("Querying address {0:x} for LCO {1:x}\n",
                                    Address, LCO.hsaHandle()));
  if (LCORelocationsMap.contains(Address)) {
    auto &Out = LCORelocationsMap.at(Address);
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("Relocation information found for loaded "
                                "address: {0:x}, targeting symbol {1}.\n",
                                Address, llvm::cantFail(Out.Symbol.getName())));

    return Out;
  }
  return std::nullopt;
}

llvm::Error CodeLifter::initLiftedLCOEntry(const hsa::LoadedCodeObject &LCO,
                                           LiftedRepresentation &LR) {
  auto ISA = LCO.getISA();
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());

  auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  LUTHIER_RETURN_ON_ERROR(
      TargetManager::instance().createTargetMachine(*ISA).moveInto(LR.TM));
  LR.TM->Options.MCOptions.AsmVerbose = true;
  // TODO: If debug information is available, the module's name must be
  // set to its source file
  llvm::orc::ThreadSafeModule TSModule{
      std::make_unique<llvm::Module>(llvm::to_string(LCO.hsaHandle()),
                                     *LR.Context.getContext()),
      LR.Context};

  auto Module = TSModule.getModuleUnlocked();
  // Set the data layout (very important)
  Module->setDataLayout(LR.TM->createDataLayout());

  auto MMIWP =
      std::make_unique<llvm::MachineModuleInfoWrapperPass>(LR.TM.get());

  auto &ModuleEntry =
      LR.Modules
          .insert({LCO.asHsaType(), std::move(std::make_pair(
                                        std::move(TSModule), MMIWP.release()))})
          .first->getSecond();
  LR.RelatedLCOs.insert(
      {LCO.asHsaType(),
       {ModuleEntry.first.getModuleUnlocked(), &ModuleEntry.second->getMMI()}});
  return llvm::Error::success();
}

llvm::Error
CodeLifter::initLiftedGlobalVariableEntry(const hsa::LoadedCodeObject &LCO,
                                          const hsa::LoadedCodeObjectSymbol &GV,
                                          LiftedRepresentation &LR) {
  auto &LLVMContext = *LR.getContext().getContext();
  auto &Module = *LR.Modules[LCO.asHsaType()].first.getModuleUnlocked();
  auto GVName = GV.getName();
  LUTHIER_RETURN_ON_ERROR(GVName.takeError());
  size_t GVSize = GV.getSize();
  // Lift each variable as an array of bytes, with a length of GVSize
  // We remove any initializers present in the LCO
  LR.RelatedGlobalVariables[&GV] = new llvm::GlobalVariable(
      Module, llvm::ArrayType::get(llvm::Type::getInt8Ty(LLVMContext), GVSize),
      false, llvm::GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
      *GVName);
  return llvm::Error::success();
}

llvm::Error
CodeLifter::initLiftedKernelEntry(const hsa::LoadedCodeObject &LCO,
                                  const hsa::LoadedCodeObjectKernel &Kernel,
                                  LiftedRepresentation &LR) {
  auto &LLVMContext = *LR.Context.getContext();
  auto &Module = *LR.Modules[LCO.asHsaType()].first.getModuleUnlocked();
  // Populate the Arguments ==================================================
  auto SymbolName = Kernel.getName();
  LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
  auto KernelMD = Kernel.getKernelMetadata();

  // Kernel's return type is always void
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);

  // Create the Kernel's FunctionType with appropriate kernel Arguments
  // (if any)
  llvm::SmallVector<llvm::Type *> Params;
  if (KernelMD.Args.has_value()) {
    // Reserve the number of arguments in the Params vector
    Params.reserve(KernelMD.Args->size());
    // For now, we only rely on required argument metadata
    // This should be updated as new cases are encountered
    for (const auto &ArgMD : *KernelMD.Args) {
      LLVM_DEBUG(llvm::dbgs() << "Argument size: " << ArgMD.Size << "\n");
      llvm::Type *ParamType =
          llvm::Type::getIntNTy(Module.getContext(), ArgMD.Size);
      // if argument is not passed by value, then it's probably a pointer
      if (ArgMD.ValueKind != hsa::md::ValueKind::ByValue) {
        // AddressSpace is most likely global, but we check it anyway if
        // it's given
        unsigned int AddressSpace = ArgMD.AddressSpace.has_value()
                                        ? *ArgMD.AddressSpace
                                        : llvm::AMDGPUAS::GLOBAL_ADDRESS;
        // Convert the argument to a pointer
        ParamType = llvm::PointerType::get(ParamType, AddressSpace);
      }
      Params.push_back(ParamType);
    }
  }

  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, Params, false);

  auto *F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::ExternalLinkage,
      SymbolName->substr(0, SymbolName->rfind(".kd")), Module);

  // Populate the Attributes =================================================

  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

  F->addFnAttr("uniform-work-group-size",
               KernelMD.UniformWorkgroupSize ? "true" : "false");

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated
  auto KDOnDevice = Kernel.getKernelDescriptor();
  LUTHIER_RETURN_ON_ERROR(KDOnDevice.takeError());

  auto KDOnHost = hsa::queryHostAddress(*KDOnDevice);
  LUTHIER_RETURN_ON_ERROR(KDOnHost.takeError());

  F->addFnAttr("amdgpu-lds-size",
               llvm::to_string((*KDOnHost)->GroupSegmentFixedSize));
  // Kern Arg is determined via analysis usage + args set earlier
  auto Rsrc1 = (*KDOnHost)->getRsrc1();
  auto Rsrc2 = (*KDOnHost)->getRsrc2();
  auto KCP = (*KDOnHost)->getKernelCodeProperties();
  if (KCP.EnableSgprDispatchId == 0) {
    F->addFnAttr("amdgpu-no-dispatch-id");
  }
  if (KCP.EnableSgprDispatchPtr == 0) {
    F->addFnAttr("amdgpu-no-dispatch-ptr");
  }
  if (KCP.EnableSgprQueuePtr == 0) {
    F->addFnAttr("amdgpu-no-queue-ptr");
  }

  F->addFnAttr("amdgpu-ieee", Rsrc1.EnableIeeeMode ? "true" : "false");
  F->addFnAttr("amdgpu-dx10-clamp", Rsrc1.EnableDx10Clamp ? "true" : "false");
  if (Rsrc2.EnableSgprWorkgroupIdX == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-x");
  }
  if (Rsrc2.EnableSgprWorkgroupIdY == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-y");
  }
  if (Rsrc2.EnableSgprWorkgroupIdZ == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-z");
  }
  switch (Rsrc2.EnableVgprWorkitemId) {
  case 0:
    F->addFnAttr("amdgpu-no-workitem-id-y");
  case 1:
    F->addFnAttr("amdgpu-no-workitem-id-z");
  case 2:
    break;
  default:
    llvm_unreachable("KD's VGPR workitem ID is not valid");
  }

  // TODO: Check the args metadata to set this correctly
  F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");

  // TODO: Set the rest of the attributes
  //    llvm::outs() << "Preloaded Args: " << (*KDOnHost)->KernArgPreload <<
  //    "\n";
  //  F->addFnAttr("amdgpu-calls");
  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module.getContext(), "", F);
  new llvm::UnreachableInst(Module.getContext(), BB);

  // Populate the MFI ========================================================

  llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs[LCO.asHsaType()].second;

  auto &MF = MMI.getOrCreateMachineFunction(*F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));
  auto &TM = MMI.getTarget();

  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(*F)->getRegisterInfo());
  auto MFI = MF.template getInfo<llvm::SIMachineFunctionInfo>();

  if (KCP.EnableSgprPrivateSegmentBuffer == 1) {
    MFI->addPrivateSegmentBuffer(*TRI);
  }
  if (KCP.EnableSgprKernArgSegmentPtr == 1) {
    MFI->addKernargSegmentPtr(*TRI);
  }
  if (KCP.EnableSgprFlatScratchInit == 1) {
    MFI->addFlatScratchInit(*TRI);
  }
  if (Rsrc2.EnableSgprPrivateSegmentWaveByteOffset == 1) {
    MFI->addPrivateSegmentWaveByteOffset();
  }

  LR.RelatedFunctions.insert({&Kernel, &MF});

  return llvm::Error::success();
}

llvm::Error CodeLifter::initLiftedDeviceFunctionEntry(
    const hsa::LoadedCodeObject &LCO,
    const hsa::LoadedCodeObjectDeviceFunction &Func, LiftedRepresentation &LR) {
  auto &LLVMContext = *LR.Context.getContext();
  auto &Module = *LR.Modules[LCO.asHsaType()].first.getModuleUnlocked();
  llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs.at(LCO.asHsaType()).second;

  auto FuncName = Func.getName();
  LUTHIER_RETURN_ON_ERROR(FuncName.takeError());
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(Module.getContext());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ReturnType != nullptr));
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {}, false);

  auto *F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::PrivateLinkage, *FuncName, Module);
  F->setCallingConv(llvm::CallingConv::C);

  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module.getContext(), "", F);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(BB != nullptr));
  new llvm::UnreachableInst(Module.getContext(), BB);
  auto &MF = MMI.getOrCreateMachineFunction(*F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));
  LR.RelatedFunctions.insert({&Func, &MF});
  return llvm::Error::success();
}

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

llvm::Error verifyInstruction(llvm::MachineInstrBuilder &Builder) {
  auto &MI = *Builder.getInstr();
  if (shouldReadExec(MI) &&
      !MI.hasRegisterImplicitUseOperand(llvm::AMDGPU::EXEC)) {
    MI.addOperand(
        llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
  }
  return llvm::Error::success();
}

void computeLiveIns(
    llvm::LivePhysRegs &LiveRegs, const llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  const llvm::MachineFunction &MF = *MBB.getParent();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  const llvm::TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
  LiveRegs.init(TRI);
  LiveRegs.addLiveOutsNoPristines(MBB);
  LLVM_DEBUG(llvm::dbgs() << "Live out registers of MBB " << MBB.getFullName()
                          << ": " << LiveRegs << "\n");
  for (const llvm::MachineInstr &MI : llvm::reverse(MBB)) {
    LiveRegs.stepBackward(MI);
    // Update the MI LiveIns map
    if (!PerMILiveIns.contains(&MI))
      (void)PerMILiveIns.insert(
          {const_cast<llvm::MachineInstr *>(&MI),
           std::move(std::make_unique<llvm::LivePhysRegs>())});
    auto &MILivePhysRegs = PerMILiveIns[const_cast<llvm::MachineInstr *>(&MI)];
    MILivePhysRegs->init(TRI);
    for (const auto &LivePhysReg : LiveRegs) {
      MILivePhysRegs->addReg(LivePhysReg);
    }
  }
}

std::vector<llvm::MachineBasicBlock::RegisterMaskPair> computeAndAddLiveIns(
    llvm::LivePhysRegs &LiveRegs, llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  luthier::computeLiveIns(LiveRegs, MBB, PerMILiveIns);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> OldLiveIns;
  // Clear out the live-ins before adding the new ones
  // This ensures correct live-out information calculations in loops i.e.
  // where the MBB is a successor/predecessor of itself
  MBB.clearLiveIns(OldLiveIns);
  llvm::addLiveIns(MBB, LiveRegs);
  return OldLiveIns;
}

/// Convenience function for recomputing live-in's for a MBB. Returns true if
/// any changes were made.
static bool recomputeLiveIns(
    llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  llvm::LivePhysRegs LPR;
  LLVM_DEBUG(auto TRI = MBB.getParent()->getSubtarget().getRegisterInfo();
             llvm::dbgs() << "Old live-in registers for MBB "
                          << MBB.getFullName() << "\n";
             for (auto &LiveInPhysReg
                  : MBB.getLiveIns()) {
               llvm::dbgs()
                   << printReg(llvm::Register(LiveInPhysReg.PhysReg), TRI)
                   << "\n";
             }

  );

  auto OldLiveIns = luthier::computeAndAddLiveIns(LPR, MBB, PerMILiveIns);
  MBB.sortUniqueLiveIns();

  const std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &NewLiveIns =
      MBB.getLiveIns();
  return OldLiveIns != NewLiveIns;
}

/// Re-implementation of \c llvm::fullyRecomputeLiveIns to save
/// the Live-ins at each MI as well
/// \param MF the lifted \c llvm::MachineFunction
/// \param PerMILiveIns a map which keeps track of the \c llvm::LivePhysRegs at
/// each \c llvm::MachineInstr
static void fullyComputeLiveInsOfLiftedMF(
    llvm::MachineFunction &MF,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  while (true) {
    bool AnyChange = false;
    for (auto &MBB : MF)
      if (luthier::recomputeLiveIns(MBB, PerMILiveIns))
        AnyChange = true;
    if (!AnyChange)
      return;
  }
}

/// Given a \c llvm::MachineInstr of call type, finds
/// all the <tt>llvm::MachineFunction</tt>s that it calls.
/// \param CallMI the call machine instruction to be analyzed
/// \return the machine function called by the \p CallMI or \c nullptr
/// if the callee is not deterministic or \c CallMI is not a call instruction
/// TODO: Make this analysis more thorough instead of just checking the
/// values inside its own basic block
static llvm::MachineFunction *
findCalleeMFOfCallInst(const llvm::MachineInstr &CallMI,
                       llvm::MachineModuleInfo &MMI) {
  if (!CallMI.isCall())
    return nullptr;
  auto Callee = CallMI.getOperand(0);

  // Since we require relocation info, callee should not have an immediate
  // type
  if (Callee.isImm())
    llvm_unreachable("Callee of type immediate should not show up here");
  // If the callee is a global value, then we have found the machine function
  // this instruction calls
  if (Callee.isGlobal()) {
    auto *CalleeFunc = llvm::dyn_cast<llvm::Function>(Callee.getGlobal());
    if (CalleeFunc == nullptr)
      llvm_unreachable("Callee is a global value of not type function");
    return MMI.getMachineFunction(*CalleeFunc);
  }
  if (Callee.isReg()) {
    auto CalleeSGPRPair = Callee.getReg();
    // Called target is an SGPR pair; We first try to find the instruction
    // that defines it inside the current basic block
    auto &MRI = CallMI.getParent()->getParent()->getRegInfo();
    llvm::SmallDenseSet<llvm::MachineInstr *, 1> UsesInCurrentMBB;
    const llvm::MachineInstr *SelectedUseInCurrentMBB{nullptr};
    for (auto &Def : MRI.def_instructions(CalleeSGPRPair)) {
      if (Def.getParent() == CallMI.getParent())
        UsesInCurrentMBB.insert(&Def);
    }
    // if the def list is zero, return for now
    // TODO: make this more thorough
    if (UsesInCurrentMBB.empty()) {
      return nullptr;
    } else if (UsesInCurrentMBB.size() > 1) {
      // Limit our analysis to the instruction closest to the CallInst
      for (auto It = CallMI.getReverseIterator();
           It != CallMI.getParent()->rend(); It++) {
        if (UsesInCurrentMBB.contains(&(*It))) {
          SelectedUseInCurrentMBB = &(*It);
          break;
        }
      }
      if (SelectedUseInCurrentMBB == nullptr)
        llvm_unreachable("should not reach here");
    } else {
      SelectedUseInCurrentMBB = *UsesInCurrentMBB.begin();
    }

    llvm::SmallDenseSet<llvm::MachineFunction *, 1> DiscoveredFunctions;
    // Now that we found the defining instruction, we can find the
    // global value operands of type function involved in construction of
    // the value until now
    auto *TRI =
        CallMI.getParent()->getParent()->getSubtarget().getRegisterInfo();
    bool DeterministicCall{true};
    for (auto It = SelectedUseInCurrentMBB->getIterator();
         It != CallMI.getIterator(); It++) {
      auto NumDefs = It->getNumExplicitDefs();
      bool ModifiesCallReg{false};
      for (int I = 0; I < NumDefs; I++) {
        if (It->getOperand(I).isReg() &&
            TRI->regsOverlap(CalleeSGPRPair, It->getOperand(I).getReg())) {
          ModifiesCallReg = true;
        }
      }
      if (ModifiesCallReg) {
        for (int I = NumDefs; I < It->getNumExplicitOperands(); I++) {
          auto Operand = It->getOperand(I);
          if (Operand.isGlobal()) {
            auto *GV = Operand.getGlobal();
            if (auto *F = llvm::dyn_cast<llvm::Function>(GV)) {
              DiscoveredFunctions.insert(MMI.getMachineFunction(*F));
            } else {
              // If a GV is not a function, then probably it has a list of
              // function pointers defined somewhere and using them; Hence
              // we can't be certain about what is being called.
              DeterministicCall = false;
            }
          }
        }
      }
    }
    if (!DeterministicCall)
      return nullptr;
    else if (DiscoveredFunctions.size() == 1)
      return *DiscoveredFunctions.begin();
    else
      return nullptr;
  } else
    return nullptr;
}

void CodeLifter::constructCallGraph(
    LiftedRepresentation &LR,
    llvm::DenseMap<llvm::MachineInstr *, llvm::MachineFunction *>
        &CalledInstrToCalleeFuncMap) {
  // Populate the CallGraph
  for (const auto &[MI, CalleeMF] : CalledInstrToCalleeFuncMap) {
    auto CallerMF = MI->getParent()->getParent();
    LiftedRepresentation::CallGraphNode *CallerNode;
    LiftedRepresentation::CallGraphNode *CalleeNode;

    if (!LR.CallGraph.contains(CallerMF)) {
      LR.CallGraph.insert(
          {CallerMF,
           std::move(std::make_unique<LiftedRepresentation::CallGraphNode>())});
      LR.CallGraph.at(CallerMF)->Node = CallerMF;
    }
    if (!LR.CallGraph.contains(CalleeMF)) {
      LR.CallGraph.insert(
          {CalleeMF,
           std::move(std::make_unique<LiftedRepresentation::CallGraphNode>())});
      LR.CallGraph.at(CalleeMF)->Node = CalleeMF;
    }

    CallerNode = LR.CallGraph.at(CallerMF).get();
    CallerNode->CalledFunctions.push_back(CalleeMF);
    CalleeNode = LR.CallGraph.at(CalleeMF).get();
    CalleeNode->CalleeFunctions.push_back(CallerMF);
  }
}

llvm::Error CodeLifter::liftFunction(
    const hsa::LoadedCodeObjectSymbol &Symbol, LiftedRepresentation &LR,
    llvm::DenseMap<const hsa::LoadedCodeObjectSymbol *, bool> &SymbolUsageMap,
    llvm::DenseMap<llvm::GlobalValue *, llvm::SmallVector<llvm::MachineInstr *>>
        &GlobalValueUses,
    llvm::DenseMap<llvm::MachineInstr *, llvm::MachineFunction *>
        &CallInstrToCalleeMap,
    bool &HasUnknownCallTarget) {
  auto LCO = hsa::LoadedCodeObject(Symbol.getLoadedCodeObject());

  llvm::Module &Module = *LR.Modules[LCO.asHsaType()].first.getModuleUnlocked();
  llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs.at(LCO.asHsaType()).second;
  llvm::MachineFunction &MF = *LR.RelatedFunctions.at(&Symbol);
  auto &F = MF.getFunction();

  LLVM_DEBUG(llvm::dbgs() << "================================================="
                             "=======================\n";
             llvm::dbgs() << "Populating contents of Machine Function "
                          << MF.getName() << "\n");

  auto &TM = MMI.getTarget();

  auto ISA = LCO.getISA();
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());

  auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();

  MF.push_back(MBB);
  auto MBBEntry = MBB;

  llvm::MCContext &MCContext = MMI.getContext();

  auto MCInstInfo = TargetInfo->getMCInstrInfo();

  llvm::DenseMap<luthier::address_t,
                 llvm::SmallVector<llvm::MachineInstr *>>
      UnresolvedBranchMIs; // < Set of branch instructions located at a
                           // luthier_address_t waiting for their
                           // target to be resolved after MBBs and MIs
                           // are created
  llvm::DenseMap<luthier::address_t, llvm::MachineBasicBlock *>
      BranchTargetMBBs; // < Set of MBBs that will be the target of the
                        // UnresolvedBranchMIs
  auto MIA = TargetInfo->getMCInstrAnalysis();

  auto TargetFunction = CodeLifter::instance().disassemble(Symbol);
  LUTHIER_RETURN_ON_ERROR(TargetFunction.takeError());

  llvm::SmallDenseSet<unsigned>
      LiveIns; // < Set of registers that are not explicitly defined by
               // instructions (AKA instruction output operand), and
               // have their value populated by the Driver using the
               // Kernel Descriptor
  llvm::SmallDenseSet<unsigned> Defines; // < Set of registers defined by
                                         // instructions (output operands)
  llvm::SmallVector<llvm::MachineBasicBlock *, 4> MBBs;
  MBBs.push_back(MBB);
  for (unsigned int InstIdx = 0; InstIdx < TargetFunction->size(); InstIdx++) {
    LLVM_DEBUG(llvm::dbgs() << "+++++++++++++++++++++++++++++++++++++++++++++++"
                               "+++++++++++++++++++++++++\n";);
    const auto &Inst = (*TargetFunction)[InstIdx];
    auto MCInst = Inst.getMCInst();
    const unsigned Opcode = getPseudoOpcodeFromReal(MCInst.getOpcode());
    const llvm::MCInstrDesc &MCID = MCInstInfo->get(Opcode);
    bool IsDirectBranch = MCID.isBranch() && !MCID.isIndirectBranch();
    bool IsDirectBranchTarget =
        isAddressDirectBranchTarget(LCO, Inst.getLoadedDeviceAddress());
    LLVM_DEBUG(llvm::dbgs() << "Lifting and adding MC Inst: ";
               MCInst.dump_pretty(llvm::dbgs(), TargetInfo->getMCInstPrinter(),
                                  " ", TargetInfo->getMCRegisterInfo());
               llvm::dbgs() << "\n";
               llvm::dbgs() << "Instruction idx: " << InstIdx << "\n";
               llvm::dbgs()
               << llvm::formatv("Loaded address of the instruction: {0:x}\n",
                                Inst.getLoadedDeviceAddress());
               llvm::dbgs()
               << llvm::formatv("Is branch? {0}\n", MCID.isBranch());
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
      BranchTargetMBBs.insert({Inst.getLoadedDeviceAddress(), MBB});
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "Address {0:x} marks the beginning of MBB idx {1}.\n",
                     Inst.getLoadedDeviceAddress(), MBB->getNumber()););
    }
    llvm::MachineInstrBuilder Builder =
        llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
    LR.MachineInstrToMCMap.insert(
        {Builder.getInstr(), const_cast<hsa::Instr *>(&Inst)});

    LLVM_DEBUG(llvm::dbgs() << "Number of operands according to MCID: "
                            << MCID.operands().size() << "\n";
               llvm::dbgs() << "Populating operands\n";);
    for (unsigned OpIndex = 0, E = MCInst.getNumOperands(); OpIndex < E;
         ++OpIndex) {
      const llvm::MCOperand &Op = MCInst.getOperand(OpIndex);
      if (Op.isReg()) {
        LLVM_DEBUG(llvm::dbgs() << "Resolving reg operand.\n");
        unsigned RegNum = Op.getReg();
        const bool IsDef = OpIndex < MCID.getNumDefs();
        unsigned Flags = 0;
        const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
        if (IsDef && !OpInfo.isOptionalDef()) {
          Flags |= llvm::RegState::Define;
          Defines.insert(RegNum);
        } else if (!Defines.contains(RegNum)) {
          LiveIns.insert(RegNum);
        }
        LLVM_DEBUG(llvm::dbgs() << "Adding register ";
                   Op.print(llvm::dbgs(), TargetInfo->getMCRegisterInfo());
                   llvm::dbgs() << " with flags " << Flags << "\n";);
        Builder.addReg(Op.getReg(), Flags);
      } else if (Op.isImm()) {
        LLVM_DEBUG(llvm::dbgs() << "Resolving an immediate operand.\n");
        // TODO: Resolve immediate load/store operands if they don't have
        // relocations associated with them (e.g. when they happen in the
        // text section)
        luthier::address_t InstAddr = Inst.getLoadedDeviceAddress();
        size_t InstSize = Inst.getSize();
        // Check if at any point in the instruction we need to apply
        // relocations
        bool RelocationApplied{false};
        for (luthier::address_t I = InstAddr; I <= InstAddr + InstSize; ++I) {
          auto RelocationInfo = resolveRelocation(LCO, I);
          LUTHIER_RETURN_ON_ERROR(RelocationInfo.takeError());
          if (RelocationInfo->has_value()) {
            auto &TargetSymbol = RelocationInfo.get()->Symbol;

            auto TargetSymbolType = TargetSymbol.getType();

            auto Addend = RelocationInfo.get()->Relocation.getAddend();
            LUTHIER_RETURN_ON_ERROR(Addend.takeError());

            uint64_t Type = RelocationInfo.get()->Relocation.getType();

            SymbolUsageMap[&TargetSymbol] = true;
            if (TargetSymbolType == hsa::LoadedCodeObjectSymbol::SK_VARIABLE) {
              LLVM_DEBUG(
                  auto TargetSymbolAddress =
                      TargetSymbol.getLoadedSymbolAddress();
                  LUTHIER_RETURN_ON_ERROR(TargetSymbolAddress.takeError());
                  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(llvm::StringRef,
                                                   TargetSymbolName,
                                                   TargetSymbol.getName());
                  llvm::dbgs()
                  << llvm::formatv("Relocation is being resolved to the global "
                                   "variables {0} at address {1:x}.\n",
                                   TargetSymbolName,
                                   reinterpret_cast<luthier::address_t>(
                                       *TargetSymbolAddress)));
              auto *GV = LR.RelatedGlobalVariables.at(&TargetSymbol);

              if (!GlobalValueUses.contains(GV))
                GlobalValueUses.insert({GV, {Builder.getInstr()}});
              else
                GlobalValueUses[GV].push_back(Builder.getInstr());

              if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                Type = llvm::SIInstrInfo::MO_GOTPCREL32_LO;
              else if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                Type = llvm::SIInstrInfo::MO_GOTPCREL32_HI;
              Builder.addGlobalAddress(GV, *Addend, Type);
            } else if (TargetSymbolType ==
                       hsa::LoadedCodeObjectSymbol::SK_DEVICE_FUNCTION) {
              auto *UsedMF = LR.RelatedFunctions.at(&TargetSymbol);
              auto *UsedF = &UsedMF->getFunction();
              if (!GlobalValueUses.contains(UsedF))
                GlobalValueUses.insert({UsedF, {Builder.getInstr()}});
              else
                GlobalValueUses[UsedF].push_back(Builder.getInstr());

              // Add the function as the operand
              if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                Type = llvm::SIInstrInfo::MO_REL32_LO;
              if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                Type = llvm::SIInstrInfo::MO_REL32_HI;
              Builder.addGlobalAddress(UsedF, *Addend, Type);
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
              Builder.addImm(Imm);
            } else {
              auto Imm = static_cast<int16_t>(Op.getImm());
              LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                             "Adding truncated imm value: {0}\n", Imm));
              Builder.addImm(Imm);
            }
          } else {
            LLVM_DEBUG(llvm::dbgs()
                       << llvm::formatv("Adding Imm: {0}\n", Op.getImm()));
            Builder.addImm(Op.getImm());
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
      auto *MMO =
          MF.getMachineMemOperand(PtrInfo,
                                  MCInstInfo->get(Builder->getOpcode()).mayLoad()
                                      ? llvm::MachineMemOperand::MOLoad
                                      : llvm::MachineMemOperand::MOStore,
                                  0, llvm::Align());
      Builder->addMemOperand(MF, MMO);
    }

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
        luthier::address_t BranchTarget;
        if (evaluateBranch(MCInst, Inst.getLoadedDeviceAddress(),
                           Inst.getSize(), BranchTarget)) {
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
      if (InstIdx != TargetFunction->size() - 1) {
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
                       Inst.getLoadedDeviceAddress(), MBB->getNumber()););
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
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MBB != nullptr));
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

  luthier::fullyComputeLiveInsOfLiftedMF(MF, LR.MachineInstrLivenessMap);

  // Experimental callgraph analysis
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.isCall()) {
        llvm::MachineFunction *CalleeMF = findCalleeMFOfCallInst(MI, MMI);
        CallInstrToCalleeMap.insert({&MI, CalleeMF});
        LLVM_DEBUG(if (CalleeMF != nullptr) llvm::dbgs()
                       << "Found MI at: " << &MI << "to be a call to function: "
                       << CalleeMF->getName() << "\n";
                   else llvm::dbgs()
                   << "MI at : " << &MI << "has an unknown call target.\n";);
        if (CalleeMF == nullptr)
          HasUnknownCallTarget = true;
      }
    }
  }

  if (MF.getFunction().getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
    // Manually set the stack frame size if the function is a kernel
    // TODO: dynamic stack kernels
    const auto *KernelSymbol =
        llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(&Symbol);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(KernelSymbol != nullptr));
    auto KernelMD = KernelSymbol->getKernelMetadata();
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

llvm::Expected<const LiftedRepresentation &>
luthier::CodeLifter::lift(const hsa::LoadedCodeObjectKernel &KernelSymbol) {
  if (!LiftedKernelSymbols.contains(&KernelSymbol)) {
    // Lift the executable if not already lifted
    // Start by creating an entry in the LiftedExecutables Map
    auto &LRPointer =
        LiftedKernelSymbols
            .insert({&KernelSymbol, std::unique_ptr<LiftedRepresentation>()})
            .first->getSecond();
    LRPointer.reset(new LiftedRepresentation());
    auto &LR = *LRPointer;
    // Create a thread-safe LLVMContext
    LR.Context =
        llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());
    auto &LLVMContext = *LR.Context.getContext();
    auto Exec = KernelSymbol.getExecutable();
    LUTHIER_RETURN_ON_ERROR(Exec.takeError());
    llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
    LUTHIER_RETURN_ON_ERROR(hsa::Executable(*Exec).getLoadedCodeObjects(LCOs));

    llvm::DenseMap<const hsa::LoadedCodeObjectSymbol *, bool> UsageMap;

    for (const auto &LCO : LCOs) {
      // create a single Module/MMI for each LCO
      LUTHIER_RETURN_ON_ERROR(initLiftedLCOEntry(LCO, LR));

      // Create Global Variables associated with this LCO
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> GlobalVariables;
      LUTHIER_RETURN_ON_ERROR(LCO.getVariableSymbols(GlobalVariables));
      LUTHIER_RETURN_ON_ERROR(LCO.getExternalSymbols(GlobalVariables));
      for (auto GV : GlobalVariables) {
        LUTHIER_RETURN_ON_ERROR(initLiftedGlobalVariableEntry(LCO, *GV, LR));
        UsageMap.insert({GV, false});
      }
      // Create Kernel entries for this LCO
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> Kernels;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
      for (const auto &Kernel : Kernels) {
        LUTHIER_RETURN_ON_ERROR(initLiftedKernelEntry(
            LCO, *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Kernel), LR));
        UsageMap.insert({Kernel, false});
      }
      // Create device function entries for this LCO
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> DeviceFuncs;
      LUTHIER_RETURN_ON_ERROR(LCO.getDeviceFunctionSymbols(DeviceFuncs));
      for (const auto &Func : DeviceFuncs) {
        LUTHIER_RETURN_ON_ERROR(initLiftedDeviceFunctionEntry(
            LCO, *llvm::dyn_cast<hsa::LoadedCodeObjectDeviceFunction>(Func),
            LR));
        UsageMap.insert({Func, false});
      }
    }
    // Now that all global objects are initialized, we can now populate
    // the target kernel's instructions
    // As we encounter global objects used by the kernel, we populate them
    llvm::DenseSet<const hsa::LoadedCodeObjectSymbol *> Populated;

    auto KernelLCO = hsa::LoadedCodeObject(KernelSymbol.getLoadedCodeObject());

    llvm::DenseMap<llvm::MachineInstr *, llvm::MachineFunction *>
        CallInstToCalleeMap;

    LUTHIER_RETURN_ON_ERROR(
        liftFunction(KernelSymbol, LR, UsageMap, LR.GlobalValueMIUses,
                     CallInstToCalleeMap, LR.HasNonDeterministicCallGraph));

    Populated.insert(&KernelSymbol);
    bool WereAnySymbolsPopulatedDuringLoop{true};
    while (WereAnySymbolsPopulatedDuringLoop) {
      WereAnySymbolsPopulatedDuringLoop = false;
      for (auto &[S, WasUsed] : UsageMap) {
        if (WasUsed || LR.HasNonDeterministicCallGraph) {
          auto Type = S->getType();
          if (Type == hsa::LoadedCodeObjectSymbol::SK_KERNEL ||
              Type == hsa::LoadedCodeObjectSymbol::SK_DEVICE_FUNCTION) {
            if (!Populated.contains(S)) {
              LUTHIER_RETURN_ON_ERROR(liftFunction(
                  *S, LR, UsageMap, LR.GlobalValueMIUses, CallInstToCalleeMap,
                  LR.HasNonDeterministicCallGraph));
              Populated.insert(S);
              WereAnySymbolsPopulatedDuringLoop = true;
            }
          }
        }
      }
    }

    constructCallGraph(LR, CallInstToCalleeMap);

    // Get the list of used LCOs
    llvm::DenseSet<hsa::LoadedCodeObject> UsedLCOs;
    for (auto &[S, WasUsed] : UsageMap) {
      if (WasUsed) {
        UsedLCOs.insert(hsa::LoadedCodeObject(S->getLoadedCodeObject()));
      }
    }

    // Cleanup any unused LCOs or variables
    for (const auto &LCO : UsedLCOs) {
      LR.RelatedLCOs.erase(LCO.asHsaType());
    }
    for (const auto &[S, WasUsed] : UsageMap) {
      if (!WasUsed) {
        if (S->getType() == hsa::LoadedCodeObjectSymbol::SK_VARIABLE ||
            S->getType() == hsa::LoadedCodeObjectSymbol::SK_EXTERNAL) {
          LR.RelatedGlobalVariables.erase(S);
        } else {
          LR.RelatedFunctions.erase(S);
        }
      }
    }
  }
  return *LiftedKernelSymbols.at(&KernelSymbol);
}

llvm::Expected<const LiftedRepresentation &>
CodeLifter::lift(const hsa::Executable &Exec) {
  if (!LiftedExecutables.contains(Exec)) {
    // Lift the executable if not already lifted
    // Start by creating an entry in the LiftedExecutables Map
    auto &LRPointer =
        LiftedExecutables
            .insert({Exec, std::unique_ptr<LiftedRepresentation>()})
            .first->getSecond();
    LRPointer.reset(new LiftedRepresentation());
    auto &LR = *LRPointer;
    // Create a thread-safe LLVMContext
    LR.Context =
        llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());
    auto &LLVMContext = *LR.Context.getContext();
    // Get all the LCOs in the executable
    llvm::SmallVector<hsa::LoadedCodeObject, 1> LCOs;
    LUTHIER_RETURN_ON_ERROR(Exec.getLoadedCodeObjects(LCOs));
    llvm::DenseMap<const hsa::LoadedCodeObjectSymbol *, bool> UsageMap;
    for (const auto &LCO : LCOs) {
      // create a single Module/MMI for each LCO
      LUTHIER_RETURN_ON_ERROR(initLiftedLCOEntry(LCO, LR));
      // Create Global Variables associated with this LCO
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> GlobalVariables;
      LUTHIER_RETURN_ON_ERROR(LCO.getVariableSymbols(GlobalVariables));
      LUTHIER_RETURN_ON_ERROR(LCO.getExternalSymbols(GlobalVariables));
      for (const auto &GV : GlobalVariables) {
        LUTHIER_RETURN_ON_ERROR(initLiftedGlobalVariableEntry(LCO, *GV, LR));
        UsageMap.insert({GV, false});
      }
      // Create Kernel entries for this LCO
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> Kernels;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
      for (const auto &Kernel : Kernels) {
        LUTHIER_RETURN_ON_ERROR(initLiftedKernelEntry(
            LCO, *llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(Kernel), LR));
        UsageMap.insert({Kernel, false});
      }
      // Create device function entries for this LCO
      llvm::SmallVector<const hsa::LoadedCodeObjectSymbol *, 4> DeviceFuncs;
      LUTHIER_RETURN_ON_ERROR(LCO.getDeviceFunctionSymbols(DeviceFuncs));
      for (const auto &Func : DeviceFuncs) {
        LUTHIER_RETURN_ON_ERROR(initLiftedDeviceFunctionEntry(
            LCO, *llvm::dyn_cast<hsa::LoadedCodeObjectDeviceFunction>(Func),
            LR));
        UsageMap.insert({Func, false});
      }
      // Now that all global objects are initialized, we can now populate
      // individual functions' instructions

      llvm::DenseMap<llvm::MachineInstr *, llvm::MachineFunction *>
          CallInstToCalleeMF;

      for (const auto &Kernel : Kernels) {
        LUTHIER_RETURN_ON_ERROR(
            liftFunction(*Kernel, LR, UsageMap, LR.GlobalValueMIUses,
                         CallInstToCalleeMF, LR.HasNonDeterministicCallGraph));
      }
      for (const auto &Func : DeviceFuncs) {
        LUTHIER_RETURN_ON_ERROR(
            liftFunction(*Func, LR, UsageMap, LR.GlobalValueMIUses,
                         CallInstToCalleeMF, LR.HasNonDeterministicCallGraph));
      }

      constructCallGraph(LR, CallInstToCalleeMF);
    }
  }
  return *LiftedExecutables.at(Exec);
}

llvm::Expected<std::unique_ptr<LiftedRepresentation>>
CodeLifter::cloneRepresentation(const LiftedRepresentation &SrcLR) {
  // Construct the output
  std::unique_ptr<LiftedRepresentation> DestLR;
  DestLR.reset(new LiftedRepresentation());
  // The cloned LiftedRepresentation will share the context and the
  // lifted primitive
  DestLR->Context = SrcLR.Context;
  DestLR->TM = SrcLR.TM;
  // This VMap will be populated by a mapping between the original global
  // objects and their cloned version. This will be useful when populating
  // the related functions and related global variable maps of the cloned
  // LiftedRepresentation
  llvm::ValueToValueMapTy VMap;
  // This map helps us populate the MachineInstr to hsa::Instr map
  llvm::DenseMap<llvm::MachineInstr *, llvm::MachineInstr *> SrcToDstInstrMap;
  for (const auto &[LCOHandle, SrcModuleAndMMI] : SrcLR.Modules) {
    const llvm::orc::ThreadSafeModule &SrcModule = SrcModuleAndMMI.first;
    const llvm::MachineModuleInfo &SrcMMI = SrcModuleAndMMI.second->getMMI();

    auto ClonedModuleAndMMI = SrcModule.withModuleDo(
        [&](const llvm::Module &M)
            -> llvm::Expected<std::tuple<
                std::unique_ptr<llvm::Module>,
                std::unique_ptr<llvm::MachineModuleInfoWrapperPass>>> {
          auto ClonedModule = llvm::CloneModule(M, VMap);
          auto ClonedMMI = std::make_unique<llvm::MachineModuleInfoWrapperPass>(
              &SrcMMI.getTarget());
          LUTHIER_RETURN_ON_ERROR(cloneMMI(SrcMMI, M, VMap, ClonedMMI->getMMI(),
                                           &SrcToDstInstrMap));
          return std::make_tuple(std::move(ClonedModule), std::move(ClonedMMI));
        });
    LUTHIER_RETURN_ON_ERROR(ClonedModuleAndMMI.takeError());

    auto &[DestModule, DestMMIWP] = *ClonedModuleAndMMI;
    // Now that the module and the MMI are cloned, create a thread-safe module
    // and put it in the Output's Module list
    llvm::orc::ThreadSafeModule DestTSModule{std::move(DestModule),
                                             DestLR->Context};
    auto &DestModuleEntry =
        DestLR->Modules
            .insert(
                {LCOHandle, std::move(std::make_pair(std::move(DestTSModule),
                                                     DestMMIWP.release()))})
            .first->getSecond();
    DestLR->RelatedLCOs.insert({LCOHandle,
                                {DestModuleEntry.first.getModuleUnlocked(),
                                 &DestModuleEntry.second->getMMI()}});
  }
  // With all Modules and MMIs cloned, we need to populate the related
  // functions and related global variables. We use the VMap to do this
  for (const auto &[GVHandle, GV] : SrcLR.RelatedGlobalVariables) {
    auto GVDestEntry = VMap.find(GV);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(GVDestEntry != VMap.end()));
    auto *DestGV = cast<llvm::GlobalVariable>(GVDestEntry->second);
    DestLR->RelatedGlobalVariables.insert({GVHandle, DestGV});
  }
  for (const auto &[FuncSymbol, SrcMF] : SrcLR.RelatedFunctions) {
    auto FDestEntry = VMap.find(&SrcMF->getFunction());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(FDestEntry != VMap.end()));
    auto *DestF = cast<llvm::Function>(FDestEntry->second);
    // Get the MMI of the dest function
    auto FuncLCO = FuncSymbol->getLoadedCodeObject();
    auto &MMI = *DestLR->RelatedLCOs.at(FuncLCO).second;

    auto DestMF = MMI.getMachineFunction(*DestF);
    DestLR->RelatedFunctions.insert({FuncSymbol, DestMF});
  }
  // Finally, populate the instruction map, the Live-ins map, and the global
  // value uses
  for (const auto &[SrcMI, HSAInst] : SrcLR.MachineInstrToMCMap) {
    DestLR->MachineInstrToMCMap.insert({SrcToDstInstrMap[SrcMI], HSAInst});
  }

  for (const auto &[SrcMI, SrcMILivePhysReg] : SrcLR.MachineInstrLivenessMap) {
    auto &[DestMI, DestMILivePhysRegs] =
        *(DestLR->MachineInstrLivenessMap
              .insert({SrcToDstInstrMap[SrcMI],
                       std::make_unique<llvm::LivePhysRegs>()})
              .first);
    auto *DestMF = DestMI->getParent()->getParent();
    auto *TRI = DestMF->getRegInfo().getTargetRegisterInfo();
    DestMILivePhysRegs->init(*TRI);
    for (const auto &LivePhysReg : *SrcMILivePhysReg) {
      DestMILivePhysRegs->addReg(LivePhysReg);
    }
  }

  for (const auto &[SrcGV, SrcMIUses] : SrcLR.GlobalValueMIUses) {
    auto DestGVIterator = VMap.find(SrcGV);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(DestGVIterator != VMap.end()));
    auto *DestGV = llvm::dyn_cast<llvm::GlobalValue>(DestGVIterator->second);
    auto &DestGVUses =
        DestLR->GlobalValueMIUses.insert({DestGV, {}}).first->getSecond();
    DestGVUses.reserve(SrcMIUses.size());
    for (const auto SrcMIUse : SrcMIUses) {
      DestGVUses.push_back(SrcToDstInstrMap[SrcMIUse]);
    }
  }

  llvm::DenseMap<llvm::MachineFunction*, llvm::MachineFunction*> SrcToDestMFMap;

  for (const auto &[FuncSymbol, SrcMF] : SrcLR.functions()) {
    auto *DestF = llvm::cast<llvm::Function>(
        VMap.find(&SrcMF->getFunction())->second);

    auto &MMI = *DestLR->RelatedLCOs.at(FuncSymbol->getLoadedCodeObject()).second;
    auto DestMF = MMI.getMachineFunction(*DestF);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(DestF != nullptr));
    SrcToDestMFMap.insert({SrcMF, DestMF});
  }
  for (const auto &[SrcMF, SrcCallGraphNode]: SrcLR.CallGraph) {
    auto * DestMF = SrcToDestMFMap.at(SrcMF);
    DestLR->CallGraph.insert({DestMF, std::move(std::make_unique<LiftedRepresentation::CallGraphNode>())});
    auto &CallGraphNode = DestLR->CallGraph.at(DestMF);
    CallGraphNode->Node = DestMF;
    for (const auto &SrcCalledFunction: SrcCallGraphNode->CalledFunctions) {
      CallGraphNode->CalledFunctions.push_back(SrcToDestMFMap.at(SrcCalledFunction));
    }
    for (const auto &SrcCalleeFunction: SrcCallGraphNode->CalleeFunctions) {
      CallGraphNode->CalleeFunctions.push_back(SrcToDestMFMap.at(SrcCalleeFunction));
    }
  }

  DestLR->HasNonDeterministicCallGraph = SrcLR.HasNonDeterministicCallGraph;

  return DestLR;
}

} // namespace luthier