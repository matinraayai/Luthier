#include "disassembler.hpp"

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

namespace luthier {

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

bool CodeLifter::isAddressBranchOrBranchTarget(
    const hsa::Executable &Executable, const hsa::GpuAgent &Agent,
    luthier::address_t Address) {
  if (!BranchAndTargetLocations.contains({Executable, Agent})) {
    return false;
  }
  auto &AddressInfo = BranchAndTargetLocations[{Executable, Agent}];
  return AddressInfo.contains(Address);
}

llvm::Expected<std::optional<hsa::ExecutableSymbol>>
luthier::CodeLifter::resolveAddressToExecutableSymbol(
    const hsa::Executable &Executable, const hsa::GpuAgent &Agent,
    luthier::address_t Address) {
  if (!ExecutableSymbolAddressInfoMap.contains({Executable, Agent})) {
    auto [It, _] = ExecutableSymbolAddressInfoMap.insert(
        {{Executable, Agent},
         llvm::DenseMap<luthier::address_t, hsa::ExecutableSymbol>{}});
    auto Symbols = Executable.getAgentSymbols(Agent);
    LUTHIER_RETURN_ON_ERROR(Symbols.takeError());
    for (const auto &S : *Symbols) {
      auto SType = S.getType();
      LUTHIER_RETURN_ON_ERROR(SType.takeError());
      luthier::address_t SAddress;
      if (*SType == HSA_SYMBOL_KIND_VARIABLE) {
        auto VarAddress = S.getVariableAddress();
        LUTHIER_RETURN_ON_ERROR(VarAddress.takeError());
        It->getSecond().insert({*VarAddress, S});
      } else {
        auto Code = S.getMachineCode();
        LUTHIER_RETURN_ON_ERROR(Code.takeError());
        It->getSecond().insert(
            {reinterpret_cast<luthier::address_t>(Code->data()), S});
      }
    }
  }
  auto &SymbolAddressInfo = ExecutableSymbolAddressInfoMap[{Executable, Agent}];
  if (SymbolAddressInfo.contains(Address))
    return SymbolAddressInfo.at(Address);
  return std::nullopt;
}

llvm::Expected<const HSAMD::Kernel::Metadata &>
CodeLifter::getKernelMetaData(const hsa::ExecutableSymbol &Symbol) {
  if (!KernelsMetaData.contains(Symbol)) {
    auto Executable = Symbol.getExecutable();
    LUTHIER_RETURN_ON_ERROR(Executable.takeError());
    auto LoadedCodeObjects = Executable->getLoadedCodeObjects();
    LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
    for (const auto &LCO : *LoadedCodeObjects)
      LUTHIER_RETURN_ON_ERROR(getLoadedCodeObjectMetaData(LCO).takeError());
  }
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(KernelsMetaData.contains(Symbol)));
  return KernelsMetaData[Symbol];
}

llvm::Expected<const HSAMD::Metadata &>
CodeLifter::getLoadedCodeObjectMetaData(const hsa::LoadedCodeObject &LCO) {
  if (!LoadedCodeObjectsMetaData.contains(LCO)) {
    auto StorageMemory = LCO.getStorageMemory();
    LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());

    auto Agent = LCO.getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());

    auto Exec = LCO.getExecutable();
    LUTHIER_RETURN_ON_ERROR(Exec.takeError());

    auto StorageElf = getAMDGCNObjectFile(*StorageMemory);
    LUTHIER_RETURN_ON_ERROR(StorageElf.takeError());

    auto MetaData = parseNoteMetaData(StorageElf->get());
    LUTHIER_RETURN_ON_ERROR(MetaData.takeError());

    llvm::outs() << *MetaData << "\n";
    for (auto &KernelMD : MetaData->Kernels) {
      LUTHIER_RETURN_ON_MOVE_INTO_FAIL(
          std::optional<hsa::ExecutableSymbol>, KernelSymbol,
          Exec->getAgentSymbolByName(*Agent, KernelMD.Symbol));
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(KernelSymbol.has_value()));
      KernelsMetaData.insert({*KernelSymbol, KernelMD});
    }
    MetaData->Kernels.clear();
    LoadedCodeObjectsMetaData.insert({LCO, *MetaData});
  }
  return LoadedCodeObjectsMetaData[LCO];
}

void luthier::CodeLifter::addBranchOrBranchTargetAddress(
    const hsa::Executable &Executable, const hsa::GpuAgent &Agent,
    luthier::address_t Address) {
  if (!BranchAndTargetLocations.contains({Executable, Agent})) {
    BranchAndTargetLocations.insert(
        {{Executable, Agent}, llvm::DenseSet<luthier::address_t>{}});
  }
  BranchAndTargetLocations[{Executable, Agent}].insert(Address);
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

llvm::Expected<const std::vector<luthier::Instr> &>
luthier::CodeLifter::disassemble(const hsa::ExecutableSymbol &Symbol) {
  if (!DisassembledSymbolsRaw.contains(Symbol)) {
    auto SymbolType = Symbol.getType();
    LUTHIER_RETURN_ON_ERROR(SymbolType.takeError());
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ARGUMENT_ERROR_CHECK(*SymbolType != HSA_SYMBOL_KIND_VARIABLE));

    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());

    // The ISA associated with the Symbol is
    auto LCO = Symbol.getLoadedCodeObject();
    LUTHIER_RETURN_ON_ERROR(LCO.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LCO->has_value()));

    auto StorageMemory = LCO.get()->getStorageMemory();
    LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
    auto StorageELF = getAMDGCNObjectFile(*StorageMemory);
    LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

    llvm::Triple TT = StorageELF.get()->makeTriple();
    std::optional<llvm::StringRef> CPU = StorageELF.get()->tryGetCPUName();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CPU.has_value()));
    llvm::SubtargetFeatures Features;
    LUTHIER_RETURN_ON_ERROR(StorageELF.get()->getFeatures().moveInto(Features));

    auto Agent = Symbol.getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());

    auto ISA = hsa::ISA::fromLLVM(TT, *CPU, Features);
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());

    auto CodeOrErrorOnDevice = Symbol.getMachineCode();
    LUTHIER_RETURN_ON_ERROR(CodeOrErrorOnDevice.takeError());
    auto CodeOrErrorOnHost = hsa::convertToHostEquivalent(*CodeOrErrorOnDevice);
    LUTHIER_RETURN_ON_ERROR(CodeOrErrorOnHost.takeError());

    auto InstructionsOrError = disassemble(*ISA, *CodeOrErrorOnHost);
    LUTHIER_RETURN_ON_ERROR(InstructionsOrError.takeError());
    auto [Instructions, Addresses] = *InstructionsOrError;

    DisassembledSymbolsRaw.insert(
        {Symbol, std::make_unique<std::vector<Instr>>()});
    auto &Out = DisassembledSymbolsRaw.at(Symbol);
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
        if (!isAddressBranchOrBranchTarget(*Executable, *Agent, Address)) {
          luthier::address_t Target;
          TargetInfo->getMCInstPrinter()->printInst(
              &Inst, Address, "", *TargetInfo->getMCSubTargetInfo(),
              llvm::outs());
          //          Inst.dump_pretty(llvm::outs(), nullptr, " ",
          //                           TargetInfo->getMCRegisterInfo());
          if (MIA->evaluateBranch(Inst, Address, Size, Target)) {
            llvm::outs() << llvm::formatv(
                "Resolved branches: Address: {0:x}, Target: {1:x}\n", Address,
                Target);

            addBranchOrBranchTargetAddress(*Executable, *Agent, Target);
          }
          addBranchOrBranchTargetAddress(*Executable, *Agent, Address);
          //          else {
          //            // TODO: properly handle this error instead of
          //            fatal_error llvm::report_fatal_error("Was not able to
          //            resolve the branch!!!");
          //          }
        }
      }
      PrevInstAddress = Address;
      Out->push_back(Instr(Inst, (**LCO).asHsaType(), Symbol.asHsaType(),
                           Address + reinterpret_cast<luthier::address_t>(
                                         CodeOrErrorOnDevice->data()),
                           Size));
    }
  }
  return *DisassembledSymbolsRaw.at(Symbol);
}

llvm::Expected<
    std::pair<std::vector<llvm::MCInst>, std::vector<luthier::address_t>>>
luthier::CodeLifter::disassemble(const llvm::object::ELFSymbolRef &Symbol,
                                 std::optional<size_t> Size) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(Symbol.getELFType() != llvm::ELF::STT_FUNC));

  auto Triple = Symbol.getObject()->makeTriple();
  auto ISA = hsa::ISA::fromName(Triple.normalize().c_str());
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());

  auto Address = Symbol.getAddress();
  LUTHIER_RETURN_ON_ERROR(Address.takeError());

  llvm::StringRef Code(reinterpret_cast<const char *>(*Address),
                       Symbol.getSize());

  if (Size.has_value())
    Code = Code.substr(0, *Size > Code.size() ? Code.size() : *Size);

  return disassemble(*ISA, llvm::arrayRefFromStringRef(Code));
}

luthier::CodeLifter::~CodeLifter() {
  DisassemblyInfoMap.clear();
  //  LiftedExecutables.clear();
  DisassembledSymbolsRaw.clear();
}

llvm::Expected<llvm::Function *>
luthier::CodeLifter::createLLVMFunctionFromSymbol(
    const hsa::ExecutableSymbol &Symbol, llvm::Module &Module) {
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(hsa_symbol_kind_t, SymbolType,
                                   Symbol.getType());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(
      SymbolType == HSA_SYMBOL_KIND_KERNEL ||
      SymbolType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION));

  bool IsKernel = SymbolType == HSA_SYMBOL_KIND_KERNEL;
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(std::string, SymbolName, Symbol.getName());

  llvm::Function *F;
  if (IsKernel) {
    // Populate the Arguments
    auto KernelMD = getKernelMetaData(Symbol);
    LUTHIER_RETURN_ON_ERROR(KernelMD.takeError());

    // Kernel's return type is always void
    llvm::Type *const ReturnType = llvm::Type::getVoidTy(Module.getContext());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ReturnType != nullptr));

    // Populate the Kernel Arguments (if any)
    llvm::SmallVector<llvm::Type *> Params;
    if (KernelMD->Args.has_value()) {
      Params.reserve(KernelMD->Args->size());
      for (const auto &ArgMD : *KernelMD->Args) {
        llvm::Type *MemParamType;
        // TODO: Resolve the other argument kinds
        if (ArgMD.ValueKind == HSAMD::ValueKind::GlobalBuffer ||
            ArgMD.ValueKind == HSAMD::ValueKind::DynamicSharedPointer) {
          auto AddressSpace = ArgMD.AddressSpace;
          LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(AddressSpace.has_value()));
          // TODO: Special provision for when the TypeName is present??
          MemParamType = llvm::PointerType::get(
              llvm::Type::getIntNTy(Module.getContext(), ArgMD.Size),
              llvm::AMDGPUAS::GLOBAL_ADDRESS);
        } else if (ArgMD.ValueKind == HSAMD::ValueKind::ByValue) {
          MemParamType = llvm::Type::getIntNTy(Module.getContext(), ArgMD.Size);
        }
        Params.push_back(MemParamType);
      }
    }

    llvm::FunctionType *FunctionType =
        llvm::FunctionType::get(ReturnType, Params, false);

    F = llvm::Function::Create(FunctionType, llvm::GlobalValue::ExternalLinkage,
                               SymbolName.substr(0, SymbolName.rfind(".kd")),
                               Module);

    F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

    if (KernelMD->UniformWorkgroupSize.has_value() &&
        *KernelMD->UniformWorkgroupSize) {
      F->addFnAttr("uniform-work-group-size", "true");
    } else {
      F->addFnAttr("uniform-work-group-size", "false");
    }

    // Construct the attributes of the Function, which will result in the MF
    // attributes getting populated
    auto KD = Symbol.getKernelDescriptor();
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
    llvm::outs() << "Preloaded Args: " << (*KDOnHost)->KernArgPreload << "\n";
    F->addFnAttr("amdgpu-calls");

  } else {
    llvm::Type *const ReturnType = llvm::Type::getVoidTy(Module.getContext());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ReturnType != nullptr));
    llvm::FunctionType *FunctionType =
        llvm::FunctionType::get(ReturnType, {}, false);

    F = llvm::Function::Create(FunctionType, llvm::GlobalValue::PrivateLinkage,
                               SymbolName, Module);
    F->setCallingConv(llvm::CallingConv::C);
  }

  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses won't
  // run
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module.getContext(), "", F);
  LUTHIER_CHECK(BB);
  new llvm::UnreachableInst(Module.getContext(), BB);
  return F;
};

llvm::Expected<llvm::MachineFunction &>
CodeLifter::createLLVMMachineFunctionFromSymbol(
    const hsa::ExecutableSymbol &Symbol, llvm::MachineModuleInfo &MMI,
    llvm::LLVMTargetMachine &TM, llvm::Function &F) {
  auto &MF = MMI.getOrCreateMachineFunction(F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));
  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(F)->getRegisterInfo());
  auto MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
  if (F.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
    LUTHIER_RETURN_ON_MOVE_INTO_FAIL(const KernelDescriptor *, KDOnDevice,
                                     Symbol.getKernelDescriptor());
    LUTHIER_RETURN_ON_MOVE_INTO_FAIL(const KernelDescriptor *, KDOnHost,
                                     hsa::queryHostAddress(KDOnDevice));
    if (KDOnHost->getKernelCodePropertiesEnableSgprPrivateSegmentBuffer() ==
        1) {
      llvm::outs() << "Private segment buffer\n";
      MFI->addPrivateSegmentBuffer(*TRI);
    }
    if (KDOnHost->getKernelCodePropertiesEnableSgprKernArgSegmentPtr() == 1) {
      llvm::outs() << "KernArg\n";
      MFI->addKernargSegmentPtr(*TRI);
    }
    //    if (KDOnHost->getKernelCodePropertiesEnableSgprFlatScratchInit() == 1)
    //    {
    llvm::outs() << "Flat scracth\n";
    MFI->addFlatScratchInit(*TRI);
    //    }
    //    if (KDOnHost->getRsrc2EnableSgprPrivateSegmentWaveByteOffset() == 1) {
    llvm::outs() << "Private segment Wave offset\n";
    MFI->addPrivateSegmentWaveByteOffset();
    //    }
  }
  //  MF.getRegInfo().freezeReservedRegs();
  return MF;
}

llvm::Expected<std::optional<CodeLifter::LCORelocationInfo>>
CodeLifter::resolveRelocation(const hsa::LoadedCodeObject &LCO,
                              luthier::address_t Address) {
  if (!Relocations.contains(LCO)) {
    auto StorageMemory = LCO.getStorageMemory();
    LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());

    auto LoadedMemory = LCO.getLoadedMemory();
    LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

    auto ELF = getAMDGCNObjectFile(*StorageMemory);
    LUTHIER_RETURN_ON_ERROR(ELF.takeError());

    auto Exec = LCO.getExecutable();
    LUTHIER_RETURN_ON_ERROR(Exec.takeError());
    auto Agent = LCO.getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());

    auto Sections = ELF.get()->sections();

    auto [LCORelocationsMapIt, MapInsertionStatus] = Relocations.insert(
        {LCO, llvm::DenseMap<luthier::address_t, LCORelocationInfo>{}});
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MapInsertionStatus));

    for (const auto &Section : Sections) {
      for (const auto &Reloc : Section.relocations()) {
        auto SymbolName = Reloc.getSymbol()->getName();
        LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());

        auto Symbol = Exec->getAgentSymbolByName(*Agent, *SymbolName);
        LUTHIER_RETURN_ON_ERROR(Symbol.takeError());

        if (Symbol->has_value()) {
          auto ELFReloc = llvm::object::ELFRelocationRef(Reloc);
          auto Addend = ELFReloc.getAddend();
          LUTHIER_RETURN_ON_ERROR(Addend.takeError());
          auto Type = ELFReloc.getType();
          //          LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Symbol->has_value()));
          llvm::outs() << "Relocation: "
                       << reinterpret_cast<luthier::address_t>(
                              LoadedMemory->data()) +
                              Reloc.getOffset()
                       << ", " << *SymbolName << " Addened " << *Addend
                       << "Type :" << Type << "\n";
          LCORelocationsMapIt->second.insert(
              {reinterpret_cast<luthier::address_t>(LoadedMemory->data()) +
                   Reloc.getOffset(),
               {**Symbol, *Addend, Type}});
        }
      }
    }
  }
  const auto &LCORelocationsMap = Relocations.at(LCO);
  llvm::outs() << "Looking for relocation at address " << Address << "\n";
  if (LCORelocationsMap.contains(Address)) {
    llvm::outs() << "Relocation was found for address " << Address << "\n";
    return LCORelocationsMap.at(Address);
  }
  return std::nullopt;
}

llvm::Expected<llvm::Function *>
cloneLLVMFunction(const llvm::Function &SourceF, llvm::Module &Module,
                  llvm::MachineModuleInfo &MMI) {
  auto DestF =
      llvm::Function::Create(SourceF.getFunctionType(), SourceF.getLinkage(),
                             SourceF.getName(), Module);
  DestF->setCallingConv(SourceF.getCallingConv());
  DestF->setAttributes(SourceF.getAttributes());
  // Don't forget to add the dummy IR Inst
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(Module.getContext(), "", DestF);
  LUTHIER_CHECK(BB);
  new llvm::UnreachableInst(Module.getContext(), BB);
  return DestF;
}

llvm::Error cloneMachineFunctionContent(
    const llvm::MachineFunction &SrcMF, llvm::MachineFunction &DestMF,
    const llvm::DenseMap<llvm::MachineFunction *, llvm::MachineFunction *>
        &MFMap,
    const llvm::DenseMap<llvm::GlobalVariable *, llvm::GlobalVariable *> &VMap,
    llvm::Module &Module, llvm::MachineModuleInfo &MMI,
    llvm::DenseMap<Instr *, llvm::MachineInstr *> &MCToMIRMap,
    llvm::DenseMap<llvm::MachineInstr *, Instr *> &MIRToMCMap) {
  auto &TM = MMI.getTarget();
  llvm::DenseMap<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *> MBBMap{
      SrcMF.getNumBlockIDs()};
  // Initializing the MBBs first will make it easier to create the branch
  // instructions
  for (const auto &SrcMBB : SrcMF) {
    MBBMap.insert({const_cast<llvm::MachineBasicBlock *>(&SrcMBB),
                   DestMF.CreateMachineBasicBlock()});
  }
  // Add the Live-ins to the MF
  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(DestMF.getFunction())->getRegisterInfo());
  for (auto &LiveIn : SrcMF.getRegInfo().liveins()) {
    DestMF.addLiveIn(LiveIn.first, TRI->getPhysRegBaseClass(LiveIn.first));
  }

  for (const auto &[SrcMBB, DestMBB] : MBBMap) {
    // Insert the successors of the Src MBB into Dest's MBB
    for (const auto SrcSuccessor : SrcMBB->successors()) {
      DestMBB->addSuccessor(MBBMap[SrcSuccessor]);
    }
    // Insert the instructions
    for (const auto &SrcMI : *SrcMBB) {
      llvm::MachineInstrBuilder DestBuilder =
          llvm::BuildMI(DestMBB, SrcMI.getDebugLoc(), SrcMI.getDesc());
      for (const auto &SrcOp : SrcMI.operands()) {
        if (SrcOp.isReg()) {
          DestBuilder->addOperand(llvm::MachineOperand::CreateReg(
              SrcOp.getReg(), SrcOp.isDef(), SrcOp.isImplicit(), SrcOp.isKill(),
              SrcOp.isDead(), SrcOp.isUndef(), SrcOp.isEarlyClobber(), 0,
              SrcOp.isDebug(), SrcOp.isInternalRead(), SrcOp.isRenamable()));
        } else if (SrcOp.isMBB()) {
          DestBuilder.addMBB(MBBMap[SrcOp.getMBB()]);
        } else if (SrcOp.isGlobal()) {
          auto GlobalValue = SrcOp.getGlobal();
          if (llvm::dyn_cast<llvm::Function>(GlobalValue)) {
            auto SrcOpMF = MMI.getMachineFunction(
                *llvm::dyn_cast<llvm::Function>(GlobalValue));
            auto DestOpMF = MFMap.at(SrcOpMF);
            DestBuilder.addGlobalAddress(&DestOpMF->getFunction(),
                                         SrcOp.getOffset(),
                                         SrcOp.getTargetFlags());
          } else if (llvm::dyn_cast<llvm::GlobalVariable>(GlobalValue)) {
            auto SrcOpGV = llvm::dyn_cast<llvm::GlobalVariable>(GlobalValue);
            auto DestOpGV = VMap.at(SrcOpGV);
            DestBuilder.addGlobalAddress(DestOpGV, SrcOp.getOffset(),
                                         SrcOp.getTargetFlags());
          }

        } else if (SrcOp.isImm()) {
          DestBuilder.addImm(SrcOp.getImm());
        } else {
          llvm_unreachable(
              "The operand's cloning logic hasn't been implemented yet");
        }
      }
    }
  }
  // Clone the properties
  DestMF.getProperties() = SrcMF.getProperties();
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

llvm::Error verifyInstruction(llvm::MachineInstrBuilder &Builder,
                              llvm::MachineFunction &MF,
                              llvm::GCNTargetMachine &TM) {
  auto &MI = *Builder.getInstr();
  if (shouldReadExec(MI) &&
      !MI.hasRegisterImplicitUseOperand(llvm::AMDGPU::EXEC)) {
    MI.addOperand(
        llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
  }
  return llvm::Error::success();
}

llvm::Expected<LiftedSymbolInfo>
CodeLifter::liftSymbolAndAddToModule(const hsa::ExecutableSymbol &Symbol,
                                     llvm::Module &Module,
                                     llvm::MachineModuleInfo &MMI) {
  auto Agent = Symbol.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  auto Exec = Symbol.getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());

  // TODO: ISA of the Symbol is determined from the LCO, not the Agent
  auto ISA = Agent->getIsa();
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());

  auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  llvm::GCNTargetMachine *TM = TargetInfo->getTargetMachine();

  auto SymbolName = Symbol.getName();
  LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());

  LiftedSymbolInfo Out;

  auto F = createLLVMFunctionFromSymbol(Symbol, Module);

  LUTHIER_RETURN_ON_ERROR(F.takeError());

  auto MF = createLLVMMachineFunctionFromSymbol(Symbol, MMI, *TM, **F);
  LUTHIER_RETURN_ON_ERROR(MF.takeError());

  // Cache the lifted Function
  llvm::MachineFunction *MFPtr = &(MF.get());

  Out.Symbol = Symbol.asHsaType();
  Out.SymbolMF = MFPtr;

  llvm::MachineBasicBlock *MBB = MF->CreateMachineBasicBlock();
  MF->push_back(MBB);
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
        isAddressBranchOrBranchTarget(*Exec, *Agent,
                                      Inst.getLoadedDeviceAddress()) &&
        !IsDirectBranch;

    if (IsDirectBranchTarget) {
      // Branch targets mark the beginning of an MBB
      auto OldMBB = MBB;
      MBB = MF->CreateMachineBasicBlock();
      MF->push_back(MBB);
      OldMBB->addSuccessor(MBB);
      BranchTargetMBBs.insert({Inst.getLoadedDeviceAddress(), MBB});
    }
    llvm::MachineInstrBuilder Builder =
        llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
    Out.MCToMachineInstrMap.insert(
        {const_cast<Instr *>(&Inst), Builder.getInstr()});
    Out.MachineInstrToMCMap.insert(
        {Builder.getInstr(), const_cast<Instr *>(&Inst)});

    for (unsigned OpIndex = 0, E = MCInst.getNumOperands(); OpIndex < E;
         ++OpIndex) {
      llvm::outs() << "Number of operands in MCID: " << MCID.operands().size()
                   << "\n";
      const llvm::MCOperand &Op = MCInst.getOperand(OpIndex);
      if (Op.isReg()) {
        llvm::outs() << "Reg Op detected \n";
        unsigned RegNum = Op.getReg();
        const bool IsDef = OpIndex < MCID.getNumDefs();
        unsigned Flags = 0;
        const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
        if (IsDef && !OpInfo.isOptionalDef()) {
          Flags |= llvm::RegState::Define;
          Defines.insert(RegNum);
        } else if (!Defines.contains(RegNum)) {
          LiveIns.insert(RegNum);
          llvm::outs() << "Live in detected: \n";
          llvm::outs() << "Register: ";
          Op.print(llvm::outs(), TargetInfo->getMCRegisterInfo());
          llvm::outs() << "\n";
          llvm::outs() << "Flags: " << Flags << "\n";
        }
        Builder.addReg(Op.getReg(), Flags);
      } else if (Op.isImm()) {
        llvm::outs() << "Imm Op detected \n";
        // TODO: Resolve immediate load/store operands if they don't have
        // relocations associated with them (e.g. when they happen in the
        // text section)
        luthier::address_t InstAddr = Inst.getLoadedDeviceAddress();
        size_t InstSize = Inst.getSize();
        // Check if at any point in the instruction we need to apply
        // relocations
        auto LCO = hsa::LoadedCodeObject(Inst.getLoadedCodeObject());
        bool RelocationApplied{false};
        for (luthier::address_t I = InstAddr; I <= InstAddr + InstSize; ++I) {
          auto RelocationInfo = resolveRelocation(LCO, I);
          LUTHIER_RETURN_ON_ERROR(RelocationInfo.takeError());
          if (RelocationInfo->has_value()) {
            llvm::outs() << "Relocation found at " << InstAddr << "for "
                         << llvm::cantFail(
                                RelocationInfo.get()->Symbol.getName())
                         << "\n";
            hsa::ExecutableSymbol TargetSymbol = RelocationInfo.get()->Symbol;

            auto TargetSymbolType = TargetSymbol.getType();
            LUTHIER_RETURN_ON_ERROR(TargetSymbolType.takeError());

            auto TargetSymbolName = TargetSymbol.getName();
            LUTHIER_RETURN_ON_ERROR(TargetSymbolName.takeError());

            auto Addend = RelocationInfo.get()->Addend;
            uint64_t Type = RelocationInfo.get()->Type;
            if (*TargetSymbolType == HSA_SYMBOL_KIND_VARIABLE) {
              // Add the Global Variable to the Executable Module if it
              // hasn't been already
              //              if (!Out.RelatedGlobalVariables.contains(
              //                      TargetSymbol.hsaHandle())) {
              //                // TODO: Detect the size of the variable, and
              //                other relevant
              //                // parameters. Right now we assume the variable
              //                fits in an
              //                // int32
              //                Out.RelatedGlobalVariables.insert(
              //                    {Symbol.hsaHandle(),
              //                      llvm::dyn_cast<llvm::GlobalVariable>(
              //                          Module.getOrInsertGlobal(
              //                              *TargetSymbolName,
              //                              llvm::Type::getInt32Ty(
              //                                  *TargetInfo->getLLVMContext())))
              //                  //                     new
              //                  llvm::GlobalVariable(
              ////                         Module,
              //// llvm::Type::getInt32Ty(*TargetInfo->getLLVMContext()), /
              /// false, llvm::GlobalValue::ExternalLinkage, nullptr, /
              ///*TargetSymbolName)
              //                    });
              //              }
              auto GV =
                  llvm::dyn_cast<llvm::GlobalVariable>(Module.getOrInsertGlobal(
                      *TargetSymbolName,
                      llvm::Type::getInt32Ty(*TargetInfo->getLLVMContext())));
              //                            GV->hasAttribute(llvm::Attribute::ZExt)
              //                            GV->setVisibility(llvm::GlobalVariable::ProtectedVisibility);
              //              llvm::outs() << "Name of global variable: " <<
              //              GV->getName() << "\n"; llvm::outs() << "Has name:
              //              " << GV->hasName() << "\n";
              if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                Type = llvm::SIInstrInfo::MO_GOTPCREL32_LO;
              else if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                Type = llvm::SIInstrInfo::MO_GOTPCREL32_HI;
              Builder.addGlobalAddress(GV, Addend, Type);
              Out.RelatedGlobalVariables.insert({TargetSymbol.hsaHandle(), GV});

              //              llvm::outs() << "Type of flag : " <<
              //              RelocationInfo.get()->Type << "\n";
              //              Builder.addExternalSymbol("globalCounter",
              //                                        llvm::SIInstrInfo::MO_REL32_LO
              //                                        );
            } else if (*TargetSymbolType == HSA_SYMBOL_KIND_INDIRECT_FUNCTION) {
              // Add this Symbol to the related functions of the current
              // function
              // Lift the function and cache it
              auto IndirectFunctionInfo =
                  liftSymbolAndAddToModule(TargetSymbol, Module, MMI);
              LUTHIER_RETURN_ON_ERROR(IndirectFunctionInfo.takeError());

              // Add the child's related functions to the parent
              for (const auto &RF : IndirectFunctionInfo->RelatedFunctions) {
                Out.RelatedFunctions.insert(RF);
              }
              // Add the child's related variables to the parent
              for (const auto &V :
                   IndirectFunctionInfo->RelatedGlobalVariables) {
                Out.RelatedGlobalVariables.insert(V);
              }
              // Add the function as the operand
              if (Type == llvm::ELF::R_AMDGPU_REL32_LO)
                Type = llvm::SIInstrInfo::MO_REL32_LO;
              if (Type == llvm::ELF::R_AMDGPU_REL32_HI)
                Type = llvm::SIInstrInfo::MO_REL32_HI;
              Builder.addGlobalAddress(
                  &IndirectFunctionInfo->SymbolMF->getFunction(), Addend, Type);
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
    LUTHIER_RETURN_ON_ERROR(verifyInstruction(Builder, *MF, *TM));
    // Basic Block resolving

    if (IsDirectBranch) {
      // Branches signal the end of the current Machine Basic Block
      llvm::outs() << llvm::formatv("Address: {0:x}\n",
                                    Inst.getLoadedDeviceAddress());
      llvm::outs() << "Found a branch!\n";
      luthier::address_t BranchTarget;
      if (MIA->evaluateBranch(MCInst, Inst.getLoadedDeviceAddress(),
                              Inst.getSize(), BranchTarget)) {
        if (!UnresolvedBranchMIs.contains(BranchTarget)) {
          UnresolvedBranchMIs.insert({BranchTarget, {Builder.getInstr()}});
        } else {
          UnresolvedBranchMIs[BranchTarget].push_back(Builder.getInstr());
        }
      }
      MCInst.dump_pretty(llvm::outs(), TargetInfo->getMCInstPrinter(), " ",
                         TargetInfo->getMCRegisterInfo());
      auto OldMBB = MBB;
      MBB = MF->CreateMachineBasicBlock();
      MF->push_back(MBB);
      OldMBB->addSuccessor(MBB);
    }
  }

  // Resolve the branch and target MIs/MBBs
  for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
    MBB = BranchTargetMBBs[TargetAddress];
    for (auto &MI : BranchMIs) {
      MI->addOperand(llvm::MachineOperand::CreateMBB(MBB));
      MI->getParent()->addSuccessor(MBB);
      MI->print(llvm::outs());
      llvm::outs() << "\n";
    }
  }

  // Add the Live-ins to the first MBB
  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM->getSubtargetImpl(**F)->getRegisterInfo());
  for (auto &LiveIn : LiveIns) {
    MF->getRegInfo().addLiveIn(LiveIn);
    MBBEntry->addLiveIn(LiveIn);
  }

  // Populate the properties of MF
  llvm::MachineFunctionProperties &Properties = MF->getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Properties.set(llvm::MachineFunctionProperties::Property::TracksLiveness);
  Properties.set(llvm::MachineFunctionProperties::Property::Selected);
  return Out;
}

llvm::Expected<std::tuple<std::unique_ptr<llvm::Module>,
                          std::unique_ptr<llvm::MachineModuleInfoWrapperPass>,
                          luthier::LiftedSymbolInfo>>
luthier::CodeLifter::liftSymbol(const hsa::ExecutableSymbol &Symbol) {
  auto Agent = Symbol.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  auto Exec = Symbol.getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());

  // TODO: ISA of the Symbol is determined from the LCO, not the Agent
  auto ISA = Agent->getIsa();
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());

  LUTHIER_RETURN_ON_ERROR(ISA.takeError());

  auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  auto SymbolName = Symbol.getName();
  LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
  auto Module = std::make_unique<llvm::Module>(*SymbolName,
                                               *TargetInfo->getLLVMContext());

  llvm::GCNTargetMachine *TM = TargetInfo->getTargetMachine();
  auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TM);
  Module->setDataLayout(TM->createDataLayout());
  Module->setPICLevel(llvm::PICLevel::BigPIC);
  auto LiftedSymbolInfo =
      liftSymbolAndAddToModule(Symbol, *Module, MMIWP->getMMI());
  LUTHIER_RETURN_ON_ERROR(LiftedSymbolInfo.takeError());
  auto Out = *LiftedSymbolInfo;
  return std::make_tuple<std::unique_ptr<llvm::Module>,
                         std::unique_ptr<llvm::MachineModuleInfoWrapperPass>,
                         luthier::LiftedSymbolInfo>(
      std::move(Module), std::move(MMIWP), std::move(Out));
}

} // namespace luthier