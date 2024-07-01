#include "code_lifter.hpp"

#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/BinaryFormat/MsgPackDocument.h>
#include <llvm/CodeGen/AsmPrinter.h>
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

#include <memory>

#include "error.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "luthier/instr.h"
#include "luthier/types.h"
#include "object_utils.hpp"
#include "target_manager.hpp"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-lifter"

namespace luthier {

template <> CodeLifter *Singleton<CodeLifter>::Instance{nullptr};

llvm::Error CodeLifter::invalidateCachedExecutableItems(hsa::Executable &Exec) {
  // Remove its lifted representation
  LiftedExecutables.erase(Exec);

  auto LCOs = Exec.getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCOs.takeError());

  for (const auto &LCO : *LCOs) {
    // Remove the branch and branch target locations
    if (BranchAndTargetLocations.contains(LCO))
      BranchAndTargetLocations.erase(LCO);
    // Remove relocation info
    if (Relocations.contains(LCO)) {
      Relocations.erase(LCO);
    }

    llvm::SmallVector<hsa::ExecutableSymbol> Symbols;
    LUTHIER_RETURN_ON_ERROR(LCO.getExecutableSymbols(Symbols));
    for (const auto &Symbol : Symbols) {
      // Remove the disassembled hsa::Instr of each hsa::ExecutableSymbol
      if (MCDisassembledSymbols.contains(Symbol))
        MCDisassembledSymbols.erase(Symbol);
      // Remove its lifted representation if this is a kernel
      if (LiftedKernelSymbols.contains(Symbol))
        LiftedKernelSymbols.erase(Symbol);
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

bool CodeLifter::isAddressBranchOrBranchTarget(const hsa::LoadedCodeObject &LCO,
                                               luthier::address_t Address) {
  if (!BranchAndTargetLocations.contains(LCO)) {
    return false;
  }
  auto &AddressInfo = BranchAndTargetLocations[LCO];
  return AddressInfo.contains(Address);
}

void luthier::CodeLifter::addBranchOrBranchTargetAddress(
    const hsa::LoadedCodeObject &LCO, luthier::address_t Address) {
  if (!BranchAndTargetLocations.contains(LCO)) {
    BranchAndTargetLocations.insert(
        {LCO, llvm::DenseSet<luthier::address_t>{}});
  }
  BranchAndTargetLocations[LCO].insert(Address);
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
luthier::CodeLifter::disassemble(const hsa::ExecutableSymbol &Symbol) {
  if (!MCDisassembledSymbols.contains(Symbol)) {
    auto SymbolType = Symbol.getType();
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ARGUMENT_ERROR_CHECK(SymbolType != hsa::VARIABLE));

    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());

    // The ISA associated with the Symbol is
    auto LCO = Symbol.getDefiningLoadedCodeObject();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LCO.has_value()));

    auto Agent = Symbol.getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());

    auto ISA = LCO->getISA();
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());

    auto MachineCodeOnDevice = Symbol.getMachineCode();
    LUTHIER_RETURN_ON_ERROR(MachineCodeOnDevice.takeError());
    auto MachineCodeOnHost = hsa::convertToHostEquivalent(*MachineCodeOnDevice);
    LUTHIER_RETURN_ON_ERROR(MachineCodeOnHost.takeError());

    auto InstructionsAndAddresses = disassemble(*ISA, *MachineCodeOnHost);
    LUTHIER_RETURN_ON_ERROR(InstructionsAndAddresses.takeError());
    auto [Instructions, Addresses] = *InstructionsAndAddresses;

    MCDisassembledSymbols.insert(
        {Symbol, std::make_unique<std::vector<hsa::Instr>>()});
    auto &Out = MCDisassembledSymbols.at(Symbol);
    Out->reserve(Instructions.size());

    auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
    LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

    auto MII = TargetInfo->getMCInstrInfo();
    auto MIA = TargetInfo->getMCInstrAnalysis();

    luthier::address_t PrevInstAddress = 0;

    auto Executable = Symbol.getExecutable();
    LUTHIER_RETURN_ON_ERROR(Executable.takeError());

    for (unsigned int I = 0; I < Instructions.size(); ++I) {
      auto &Inst = Instructions[I];
      auto &Address = Addresses[I];
      auto Size = Address - PrevInstAddress;
      if (MII->get(Inst.getOpcode()).isBranch()) {
        if (!isAddressBranchOrBranchTarget(*LCO, Address)) {
          luthier::address_t Target;
          //          TargetInfo->getMCInstPrinter()->printInst(
          //              &Inst, Address, "", *TargetInfo->getMCSubTargetInfo(),
          //              llvm::outs());
          //          Inst.dump_pretty(llvm::outs(), nullptr, " ",
          //                           TargetInfo->getMCRegisterInfo());
          if (MIA->evaluateBranch(Inst, Address, Size, Target)) {
            //            llvm::outs() << llvm::formatv(
            //                "Resolved branches: Address: {0:x}, Target:
            //                {1:x}\n", Address, Target);

            addBranchOrBranchTargetAddress(*LCO, Target);
          }
          addBranchOrBranchTargetAddress(*LCO, Address);
        }
      }
      PrevInstAddress = Address;
      Out->push_back(hsa::Instr(Inst, Symbol.asHsaType(),
                                Address + reinterpret_cast<luthier::address_t>(
                                              MachineCodeOnDevice->data()),
                                Size));
    }
  }
  return *MCDisassembledSymbols.at(Symbol);
}

llvm::Expected<std::optional<CodeLifter::LCORelocationInfo>>
CodeLifter::resolveRelocation(const hsa::LoadedCodeObject &LCO,
                              luthier::address_t Address) {
  if (!Relocations.contains(LCO)) {
    // If the LCO doesn't have its relocation info cached, cache it
    auto LoadedMemory = LCO.getLoadedMemory();
    LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

    auto LoadedMemoryBegin = reinterpret_cast<address_t>(LoadedMemory->data());

    auto StorageELF = LCO.getStorageELF();
    LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

    // Create an entry for the LCO in the relocations map
    auto LCORelocationsMap =
        Relocations
            .insert({LCO, llvm::DenseMap<address_t, LCORelocationInfo>{}})
            .first->getSecond();

    for (const auto &Section : StorageELF->sections()) {
      for (const llvm::object::ELFRelocationRef Reloc : Section.relocations()) {
        // Only rely on the loaded address of the symbol instead of its name
        // The name will be stripped from the relocation section
        // if the symbol has a private linkage
        auto RelocSymbolLoadedAddress = Reloc.getSymbol()->getAddress();
        llvm::outs() << llvm::cantFail(Reloc.getSymbol()->getName()) << "\n";
        LUTHIER_RETURN_ON_ERROR(RelocSymbolLoadedAddress.takeError());
        // Check with the hsa::Platform which HSA executable Symbol this address
        // is associated with
        auto RelocSymbol = hsa::Platform::instance().getSymbolFromLoadedAddress(
            LoadedMemoryBegin + *RelocSymbolLoadedAddress);
        LUTHIER_RETURN_ON_ERROR(RelocSymbol.takeError());
        LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(RelocSymbol->has_value()));
        luthier::address_t TargetAddress =
            reinterpret_cast<luthier::address_t>(LoadedMemory->data()) +
            Reloc.getOffset();
        LCORelocationsMap.insert({TargetAddress, {**RelocSymbol, Reloc}});
      }
    }
  }
  const auto &LCORelocationsMap = Relocations.at(LCO);

  if (LCORelocationsMap.contains(Address)) {
    auto &Out = LCORelocationsMap.at(Address);
    LLVM_DEBUG(
        llvm::dbgs() << llvm::formatv(
            "Relocation information found for loaded address: {0:x}, with data "
            "ref: {1}.\n",
            Address, Out.Relocation.getRawDataRefImpl()));

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

  llvm::GCNTargetMachine *TM = TargetInfo->getTargetMachine();

  // TODO: If debug information is available, the module's name must be
  // set to its source file
  llvm::orc::ThreadSafeModule TSModule{
      std::make_unique<llvm::Module>(llvm::to_string(LCO.hsaHandle()),
                                     *LR.Context.getContext()),
      LR.Context};

  auto Module = TSModule.getModuleUnlocked();
  // Set the data layout (very important)
  Module->setDataLayout(TM->createDataLayout());

  auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TM);
  MMIWP->doInitialization(*Module);
  auto &MMI = MMIWP->getMMI();

  auto &ModuleEntry = LR.Modules
                          .insert({LCO.hsaHandle(),
                                   std::move(std::make_pair(std::move(TSModule),
                                                            std::move(MMIWP)))})
                          .first->getSecond();
  LR.RelatedLCOs.insert({LCO.hsaHandle(), {&ModuleEntry.first, &MMI}});
  return llvm::Error::success();
}

llvm::Error
CodeLifter::initLiftedGlobalVariableEntry(const hsa::LoadedCodeObject &LCO,
                                          const hsa::ExecutableSymbol &GV,
                                          LiftedRepresentation &LR) {
  auto &LLVMContext = *LR.getContext().getContext();
  auto &Module = *LR.RelatedLCOs.at(LCO.hsaHandle()).first->getModuleUnlocked();
  auto GVName = GV.getName();
  LUTHIER_RETURN_ON_ERROR(GVName.takeError());
  size_t GVSize = GV.getSize();
  // Lift each variable as an array of bytes, with a length of GVSize
  // We remove any initializers present in the LCO
  new llvm::GlobalVariable(
      Module, llvm::ArrayType::get(llvm::Type::getInt8Ty(LLVMContext), GVSize),
      false, llvm::GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
      *GVName);
  return llvm::Error::success();
}

llvm::Error
CodeLifter::initLiftedKernelEntry(const hsa::LoadedCodeObject &LCO,
                                  const hsa::ExecutableSymbol &Kernel,
                                  LiftedRepresentation &LR) {
  auto &LLVMContext = *LR.Context.getContext();
  auto &Module = *LR.RelatedLCOs.at(LCO.hsaHandle()).first->getModuleUnlocked();
  // Populate the Arguments ==================================================
  auto SymbolName = Kernel.getName();
  LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
  auto KernelMD = Kernel.getKernelMetadata();
  LUTHIER_RETURN_ON_ERROR(KernelMD.takeError());

  // Kernel's return type is always void
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);

  // Create the Kernel's FunctionType with appropriate kernel Arguments
  // (if any)
  llvm::SmallVector<llvm::Type *> Params;
  if (KernelMD->Args.has_value()) {
    // Reserve the number of arguments in the Params vector
    Params.reserve(KernelMD->Args->size());
    // For now, we only rely on required argument metadata
    // This should be updated as new cases are encountered
    for (const auto &ArgMD : *KernelMD->Args) {
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
        ParamType =
            llvm::PointerType::get(ParamType, AddressSpace);
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
               KernelMD->UniformWorkgroupSize ? "true" : "false");

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated
  auto KD = Kernel.getKernelDescriptor();
  LUTHIER_RETURN_ON_ERROR(KD.takeError());

  auto KDOnHost = hsa::queryHostAddress(*KD);
  LUTHIER_RETURN_ON_ERROR(KDOnHost.takeError());

  F->addFnAttr(
      "amdgpu-lds-size",
      llvm::formatv("0, {0}", (*KDOnHost)->GroupSegmentFixedSize).str());
  // Private (scratch) segment size is determined by Analysis Usage pass
  // Kern Arg is determined via analysis usage + args set earlier
  if ((*KDOnHost)->getKernelCodePropertiesEnableSgprDispatchId() == 0) {
    F->addFnAttr("amdgpu-no-dispatch-id");
  }
  if ((*KDOnHost)->getKernelCodePropertiesEnableSgprDispatchPtr() == 0) {
    F->addFnAttr("amdgpu-no-dispatch-ptr");
  }
  if ((*KDOnHost)->getKernelCodePropertiesEnableSgprQueuePtr() == 0) {
    F->addFnAttr("amdgpu-no-queue-ptr");
  }
  F->addFnAttr("amdgpu-ieee",
               (*KDOnHost)->getRsrc1EnableIeeeMode() ? "true" : "false");
  F->addFnAttr("amdgpu-dx10-clamp",
               (*KDOnHost)->getRsrc1EnableDx10Clamp() ? "true" : "false");
  if ((*KDOnHost)->getRsrc2EnableSgprWorkgroupIdX() == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-x");
  }
  if ((*KDOnHost)->getRsrc2EnableSgprWorkgroupIdY() == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-y");
  }
  if ((*KDOnHost)->getRsrc2EnableSgprWorkgroupIdZ() == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-z");
  }
  switch ((*KDOnHost)->getRsrc2EnableVgprWorkitemId()) {
  case 0:
    F->addFnAttr("amdgpu-no-workitem-id-y");
  case 1:
    F->addFnAttr("amdgpu-no-workitem-id-z");
    break;
  default:
    llvm_unreachable("KD's VGPR workitem ID is not valid");
  }

  // TODO: Check the args metadata to set this correctly
  F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");

  // TODO: Set the rest of the attributes
  //    llvm::outs() << "Preloaded Args: " << (*KDOnHost)->KernArgPreload <<
  //    "\n";
  F->addFnAttr("amdgpu-calls");
  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module.getContext(), "", F);
  LUTHIER_CHECK(BB);
  new llvm::UnreachableInst(Module.getContext(), BB);

  // Populate the MFI ========================================================

  llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs[LCO.hsaHandle()].second;

  auto &MF = MMI.getOrCreateMachineFunction(*F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));
  auto &TM = MMI.getTarget();

  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(*F)->getRegisterInfo());
  auto MFI = MF.template getInfo<llvm::SIMachineFunctionInfo>();

  if ((*KDOnHost)->getKernelCodePropertiesEnableSgprPrivateSegmentBuffer() ==
      1) {
    MFI->addPrivateSegmentBuffer(*TRI);
  }
  if ((*KDOnHost)->getKernelCodePropertiesEnableSgprKernArgSegmentPtr() == 1) {
    MFI->addKernargSegmentPtr(*TRI);
  }
  if ((*KDOnHost)->getKernelCodePropertiesEnableSgprFlatScratchInit() == 1) {
    MFI->addFlatScratchInit(*TRI);
  }
  if ((*KDOnHost)->getRsrc2EnableSgprPrivateSegmentWaveByteOffset() == 1) {
    //    llvm::outs() << "Private segment Wave offset\n";
    MFI->addPrivateSegmentWaveByteOffset();
  }

  LR.RelatedFunctions.insert({Kernel.hsaHandle(), &MF});

  return llvm::Error::success();
}

llvm::Error
CodeLifter::initLiftedDeviceFunctionEntry(const hsa::LoadedCodeObject &LCO,
                                          const hsa::ExecutableSymbol &Func,
                                          LiftedRepresentation &LR) {
  auto &LLVMContext = *LR.Context.getContext();
  auto &Module = *LR.RelatedLCOs.at(LCO.hsaHandle()).first->getModuleUnlocked();
  llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs.at(LCO.hsaHandle()).second;

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
  LUTHIER_CHECK(BB);
  new llvm::UnreachableInst(Module.getContext(), BB);
  auto &MF = MMI.getOrCreateMachineFunction(*F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));
  LR.RelatedFunctions.insert({Func.hsaHandle(), &MF});
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

llvm::Error CodeLifter::liftFunction(
    const hsa::ExecutableSymbol &Symbol, LiftedRepresentation &LR,
    llvm::DenseMap<hsa::ExecutableSymbol, bool> &SymbolUsageMap) {
  auto LCO = Symbol.getDefiningLoadedCodeObject();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LCO.has_value()));

  llvm::Module &Module =
      *LR.RelatedLCOs.at(LCO->hsaHandle()).first->getModuleUnlocked();
  llvm::MachineModuleInfo &MMI = *LR.RelatedLCOs.at(LCO->hsaHandle()).second;
  llvm::MachineFunction &MF = *LR.RelatedFunctions.at(Symbol.hsaHandle());
  auto &F = MF.getFunction();
  auto &TM = MMI.getTarget();

  auto ISA = LCO->getISA();
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

  for (const auto &Inst : *TargetFunction) {
    auto MCInst = Inst.getMCInst();
    const unsigned Opcode = MCInst.getOpcode();
    const llvm::MCInstrDesc &MCID = MCInstInfo->get(Opcode);
    bool IsDirectBranch = MCID.isBranch() && !MCID.isIndirectBranch();
    bool IsDirectBranchTarget =
        isAddressBranchOrBranchTarget(*LCO, Inst.getLoadedDeviceAddress()) &&
        !IsDirectBranch;

    if (IsDirectBranchTarget) {
      // Branch targets mark the beginning of an MBB
      auto OldMBB = MBB;
      MBB = MF.CreateMachineBasicBlock();
      MF.push_back(MBB);
      OldMBB->addSuccessor(MBB);
      BranchTargetMBBs.insert({Inst.getLoadedDeviceAddress(), MBB});
    }
    llvm::MachineInstrBuilder Builder =
        llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
    LR.MachineInstrToMCMap.insert(
        {Builder.getInstr(), const_cast<hsa::Instr *>(&Inst)});

    for (unsigned OpIndex = 0, E = MCInst.getNumOperands(); OpIndex < E;
         ++OpIndex) {
      //      llvm::outs() << "Number of operands in MCID: " <<
      //      MCID.operands().size()
      //                   << "\n";
      const llvm::MCOperand &Op = MCInst.getOperand(OpIndex);
      if (Op.isReg()) {
        //        llvm::outs() << "Reg Op detected \n";
        unsigned RegNum = Op.getReg();
        const bool IsDef = OpIndex < MCID.getNumDefs();
        unsigned Flags = 0;
        const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
        if (IsDef && !OpInfo.isOptionalDef()) {
          Flags |= llvm::RegState::Define;
          Defines.insert(RegNum);
        } else if (!Defines.contains(RegNum)) {
          LiveIns.insert(RegNum);
          //          llvm::outs() << "Live in detected: \n";
          //          llvm::outs() << "Register: ";
          //          Op.print(llvm::outs(), TargetInfo->getMCRegisterInfo());
          //          llvm::outs() << "\n";
          //          llvm::outs() << "Flags: " << Flags << "\n";
        }
        Builder.addReg(Op.getReg(), Flags);
      } else if (Op.isImm()) {
        // TODO: Resolve immediate load/store operands if they don't have
        // relocations associated with them (e.g. when they happen in the
        // text section)
        luthier::address_t InstAddr = Inst.getLoadedDeviceAddress();
        size_t InstSize = Inst.getSize();
        // Check if at any point in the instruction we need to apply
        // relocations
        bool RelocationApplied{false};
        for (luthier::address_t I = InstAddr; I <= InstAddr + InstSize; ++I) {
          auto RelocationInfo = resolveRelocation(*LCO, I);
          LUTHIER_RETURN_ON_ERROR(RelocationInfo.takeError());
          if (RelocationInfo->has_value()) {
            hsa::ExecutableSymbol TargetSymbol = RelocationInfo.get()->Symbol;

            auto TargetSymbolType = TargetSymbol.getType();

            auto Addend = RelocationInfo.get()->Relocation.getAddend();
            LUTHIER_RETURN_ON_ERROR(Addend.takeError());

            uint64_t Type = RelocationInfo.get()->Relocation.getType();

            SymbolUsageMap[TargetSymbol] = true;
            if (TargetSymbolType == hsa::VARIABLE) {
              auto *GV = LR.RelatedGlobalVariables.at(TargetSymbol.hsaHandle());
              if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                Type = llvm::SIInstrInfo::MO_GOTPCREL32_LO;
              else if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                Type = llvm::SIInstrInfo::MO_GOTPCREL32_HI;
              Builder.addGlobalAddress(GV, *Addend, Type);
            } else if (TargetSymbolType == hsa::DEVICE_FUNCTION) {
              auto *UsedMF = LR.RelatedFunctions.at(TargetSymbol.hsaHandle());
              // Add the function as the operand
              if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                Type = llvm::SIInstrInfo::MO_REL32_LO;
              if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                Type = llvm::SIInstrInfo::MO_REL32_HI;
              Builder.addGlobalAddress(&UsedMF->getFunction(), *Addend, Type);
            } else {
              // For now, we don't handle calling kernels from kernels
              llvm_unreachable("not implemented");
            }
            RelocationApplied = true;
            break;
          }
        }
        if (!RelocationApplied) {
          Builder.addImm(Op.getImm());
        }

      } else if (!Op.isValid()) {
        llvm_unreachable("Operand is not set");
      } else {
        llvm_unreachable("Not yet implemented");
      }
    }
    LUTHIER_RETURN_ON_ERROR(verifyInstruction(Builder));
    // Basic Block resolving

    if (IsDirectBranch) {
      luthier::address_t BranchTarget;
      if (MIA->evaluateBranch(MCInst, Inst.getLoadedDeviceAddress(),
                              Inst.getSize(), BranchTarget)) {
        if (!UnresolvedBranchMIs.contains(BranchTarget)) {
          UnresolvedBranchMIs.insert({BranchTarget, {Builder.getInstr()}});
        } else {
          UnresolvedBranchMIs[BranchTarget].push_back(Builder.getInstr());
        }
      }
      //      MCInst.dump_pretty(llvm::outs(), TargetInfo->getMCInstPrinter(),
      //      "
      //      ",
      //                         TargetInfo->getMCRegisterInfo());
      auto OldMBB = MBB;
      MBB = MF.CreateMachineBasicBlock();
      MF.push_back(MBB);
      OldMBB->addSuccessor(MBB);
    }
  }

  // Resolve the branch and target MIs/MBBs
  for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
    MBB = BranchTargetMBBs[TargetAddress];
    for (auto &MI : BranchMIs) {
      MI->addOperand(llvm::MachineOperand::CreateMBB(MBB));
      MI->getParent()->addSuccessor(MBB);
      //      MI->print(llvm::outs());
      //      llvm::outs() << "\n";
    }
  }

  // Add the Live-ins to the first MBB
  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(F)->getRegisterInfo());
  for (auto &LiveIn : LiveIns) {
    MF.getRegInfo().addLiveIn(LiveIn);
    MBBEntry->addLiveIn(LiveIn);
  }

  // Populate the properties of MF
  llvm::MachineFunctionProperties &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Properties.set(llvm::MachineFunctionProperties::Property::TracksLiveness);
  Properties.set(llvm::MachineFunctionProperties::Property::Selected);
  return llvm::Error::success();
}

llvm::Expected<const LiftedRepresentation &>
luthier::CodeLifter::lift(const hsa::ExecutableSymbol &Symbol) {
  if (!LiftedKernelSymbols.contains(Symbol)) {
    // Lift the executable if not already lifted
    // Start by creating an entry in the LiftedExecutables Map
    auto &LRPointer =
        LiftedKernelSymbols
            .insert({Symbol, std::unique_ptr<LiftedRepresentation>()})
            .first->getSecond();
    LRPointer.reset(new LiftedRepresentation());
    auto &LR = *LRPointer;
    // Create a thread-safe LLVMContext
    LR.Context =
        llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());
    auto &LLVMContext = *LR.Context.getContext();
    auto Exec = Symbol.getExecutable();
    LUTHIER_RETURN_ON_ERROR(Exec.takeError());
    auto LCOs = Exec->getLoadedCodeObjects();
    LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
    llvm::DenseMap<hsa::ExecutableSymbol, bool> UsageMap;

    for (const auto &LCO : *LCOs) {
      // create a single Module/MMI for each LCO
      LUTHIER_RETURN_ON_ERROR(initLiftedLCOEntry(LCO, LR));

      // Create Global Variables associated with this LCO
      llvm::SmallVector<hsa::ExecutableSymbol, 4> GlobalVariables;
      LUTHIER_RETURN_ON_ERROR(LCO.getVariableSymbols(GlobalVariables));
      for (const auto &GV : GlobalVariables) {
        LUTHIER_RETURN_ON_ERROR(initLiftedGlobalVariableEntry(LCO, GV, LR));
        UsageMap.insert({GV, false});
      }
      // Create Kernel entries for this LCO
      llvm::SmallVector<hsa::ExecutableSymbol, 4> Kernels;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
      for (const auto &Kernel : Kernels) {
        LUTHIER_RETURN_ON_ERROR(initLiftedKernelEntry(LCO, Kernel, LR));
        UsageMap.insert({Kernel, false});
      }
      // Create device function entries for this LCO
      llvm::SmallVector<hsa::ExecutableSymbol, 4> DeviceFuncs;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(DeviceFuncs));
      for (const auto &Func : DeviceFuncs) {
        LUTHIER_RETURN_ON_ERROR(initLiftedDeviceFunctionEntry(LCO, Func, LR));
        UsageMap.insert({Func, false});
      }
    }
    // Now that all global objects are initialized, we can now populate
    // the target kernel's instructions
    // As we encounter global objects used by the kernel, we populate them
    llvm::DenseSet<hsa::ExecutableSymbol> Populated;

    auto KernelLCO = Symbol.getDefiningLoadedCodeObject();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(KernelLCO.has_value()));

    LUTHIER_RETURN_ON_ERROR(liftFunction(Symbol, LR, UsageMap));
    Populated.insert(Symbol);
    bool WereAnySymbolsPopulatedDuringLoop{true};
    while (WereAnySymbolsPopulatedDuringLoop) {
      WereAnySymbolsPopulatedDuringLoop = false;
      for (auto &[S, WasUsed] : UsageMap) {
        if (WasUsed) {
          auto Type = S.getType();
          if (Type == hsa::KERNEL || Type == hsa::DEVICE_FUNCTION) {
            if (!Populated.contains(S)) {
              LUTHIER_RETURN_ON_ERROR(liftFunction(S, LR, UsageMap));
              Populated.insert(S);
              WereAnySymbolsPopulatedDuringLoop = true;
            }
          }
        }
      }
    }

    // Get the list of used LCOs
    llvm::DenseSet<hsa::LoadedCodeObject> UsedLCOs;
    for (auto &[S, WasUsed] : UsageMap) {
      if (WasUsed) {
        auto SymbolLCO = S.getDefiningLoadedCodeObject();
        if (SymbolLCO.has_value())
          UsedLCOs.insert(*SymbolLCO);
      }
    }

    // Cleanup any unused LCOs or variables
    for (const auto &LCO : UsedLCOs) {
      LR.Modules.erase(LCO.hsaHandle());
      LR.RelatedLCOs.erase(LCO.hsaHandle());
    }
    for (const auto &[S, WasUsed] : UsageMap) {
      if (!WasUsed) {
        if (S.getType() == hsa::VARIABLE) {
          LR.RelatedGlobalVariables.erase(S.hsaHandle());
        } else {
          LR.RelatedFunctions.erase(S.hsaHandle());
        }
      }
    }
  }
  return *LiftedKernelSymbols.at(Symbol);
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
    auto LCOs = Exec.getLoadedCodeObjects();
    LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
    llvm::DenseMap<hsa::ExecutableSymbol, bool> UsageMap;
    for (const auto &LCO : *LCOs) {
      // create a single Module/MMI for each LCO
      LUTHIER_RETURN_ON_ERROR(initLiftedLCOEntry(LCO, LR));
      // Create Global Variables associated with this LCO
      llvm::SmallVector<hsa::ExecutableSymbol, 4> GlobalVariables;
      LUTHIER_RETURN_ON_ERROR(LCO.getVariableSymbols(GlobalVariables));
      for (const auto &GV : GlobalVariables) {
        LUTHIER_RETURN_ON_ERROR(initLiftedGlobalVariableEntry(LCO, GV, LR));
        UsageMap.insert({GV, false});
      }
      // Create Kernel entries for this LCO
      llvm::SmallVector<hsa::ExecutableSymbol, 4> Kernels;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(Kernels));
      for (const auto &Kernel : Kernels) {
        LUTHIER_RETURN_ON_ERROR(initLiftedKernelEntry(LCO, Kernel, LR));
        UsageMap.insert({Kernel, false});
      }
      // Create device function entries for this LCO
      llvm::SmallVector<hsa::ExecutableSymbol, 4> DeviceFuncs;
      LUTHIER_RETURN_ON_ERROR(LCO.getKernelSymbols(DeviceFuncs));
      for (const auto &Func : DeviceFuncs) {
        LUTHIER_RETURN_ON_ERROR(initLiftedDeviceFunctionEntry(LCO, Func, LR));
        UsageMap.insert({Func, false});
      }
      // Now that all global objects are initialized, we can now populate
      // individual functions' instructions
      for (const auto &Kernel : Kernels) {
        LUTHIER_RETURN_ON_ERROR(liftFunction(Kernel, LR, UsageMap));
      }
      for (const auto &Func : DeviceFuncs) {
        LUTHIER_RETURN_ON_ERROR(liftFunction(Func, LR, UsageMap));
      }
    }
  }
  return *LiftedExecutables.at(Exec);
}

llvm::Expected<std::unique_ptr<LiftedRepresentation>>
CodeLifter::cloneRepresentation(const LiftedRepresentation &LR) {
  // Construct the output
  std::unique_ptr<LiftedRepresentation> Out;
  Out.reset(new LiftedRepresentation());
  // The cloned LiftedRepresentation will share the context and the
  // lifted primitive
  Out->Context = LR.Context;
  // This VMap will be populated by a mapping between the original global
  // objects and their cloned version. This will be useful when populating
  // the related functions and related global variable maps of the cloned
  // LiftedRepresentation
  llvm::ValueToValueMapTy VMap;
  // This map helps us populate the MachineInstr to hsa::Instr map
  llvm::DenseMap<llvm::MachineInstr *, llvm::MachineInstr *> SrcToDstInstrMap;
  for (const auto &[LCOHandle, ModuleAndMMIWP] : LR.Modules) {
    const llvm::orc::ThreadSafeModule &TSModule = ModuleAndMMIWP.first;
    const std::unique_ptr<llvm::MachineModuleInfoWrapperPass> &MMIWP =
        ModuleAndMMIWP.second;

    auto ClonedModuleAndMMIWP = TSModule.withModuleDo(
        [&](const llvm::Module &M)
            -> llvm::Expected<std::tuple<
                std::unique_ptr<llvm::Module>,
                std::unique_ptr<llvm::MachineModuleInfoWrapperPass>>> {
          auto ClonedModule = llvm::CloneModule(M, VMap);
          auto ClonedMMIWP =
              std::make_unique<llvm::MachineModuleInfoWrapperPass>(
                  &MMIWP->getMMI().getTarget());
          ClonedMMIWP->doInitialization(*ClonedModule);
          LUTHIER_RETURN_ON_ERROR(luthier::cloneMMI(
              MMIWP->getMMI(), VMap, ClonedMMIWP->getMMI(), &SrcToDstInstrMap));
          return std::make_tuple(std::move(ClonedModule),
                                 std::move(ClonedMMIWP));
        });
    LUTHIER_RETURN_ON_ERROR(ClonedModuleAndMMIWP.takeError());

    auto &[ClonedModule, ClonedMMIWP] = *ClonedModuleAndMMIWP;
    // Now that the module and the MMI are cloned, create a thread-safe module
    // and put it in the Output's Module list
    llvm::orc::ThreadSafeModule ClonedTSModule{std::move(ClonedModule),
                                               Out->Context};

    auto &ClonedModuleEntry =
        Out->Modules
            .insert(
                {LCOHandle, std::move(std::make_pair(std::move(ClonedTSModule),
                                                     std::move(ClonedMMIWP)))})
            .first->getSecond();
    Out->RelatedLCOs.insert(
        {LCOHandle,
         {&ClonedModuleEntry.first, &ClonedModuleEntry.second->getMMI()}});
  }
  // With all Modules and MMIs cloned, we need to populate the related
  // functions and related global variables. We use the VMap to do this
  for (const auto &[GVHandle, GV] : LR.RelatedGlobalVariables) {
    auto GVDestEntry = VMap.find(GV);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(GVDestEntry != VMap.end()));
    auto *DestGV = cast<llvm::GlobalVariable>(GVDestEntry->second);
    Out->RelatedGlobalVariables.insert({GVHandle, DestGV});
  }

  for (const auto &[FuncHandle, SrcMF] : LR.RelatedFunctions) {
    auto FDestEntry = VMap.find(&SrcMF->getFunction());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(FDestEntry != VMap.end()));
    auto *DestF = cast<llvm::Function>(FDestEntry->second);
    // Get the MMI of the dest function
    auto FuncSymbol = hsa::ExecutableSymbol::fromHandle({FuncHandle});
    LUTHIER_RETURN_ON_ERROR(FuncSymbol.takeError());
    auto FuncLCO = FuncSymbol->getDefiningLoadedCodeObject()->hsaHandle();
    auto &MMI = *Out->RelatedLCOs.at(FuncLCO).second;

    auto DestMF = MMI.getMachineFunction(*DestF);

    Out->RelatedFunctions.insert({FuncHandle, DestMF});
  }
  // Finally, populate the instruction map
  for (const auto &[SrcMI, HSAInst] : LR.MachineInstrToMCMap) {
    Out->MachineInstrToMCMap.insert({SrcToDstInstrMap[SrcMI], HSAInst});
  }

  return Out;
}

} // namespace luthier