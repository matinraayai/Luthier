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
      Out->push_back(hsa::Instr(Inst, LCO->asHsaType(), Symbol.asHsaType(),
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
        LUTHIER_RETURN_ON_ERROR(RelocSymbolLoadedAddress.takeError());
        // Check with the hsa::Platform which HSA executable Symbol this address
        // is associated with
        auto RelocSymbol = hsa::Platform::instance().getSymbolFromLoadedAddress(
            *RelocSymbolLoadedAddress);
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

llvm::Error CodeLifter::verifyInstruction(llvm::MachineInstrBuilder &Builder) {
  auto &MI = *Builder.getInstr();
  if (shouldReadExec(MI) &&
      !MI.hasRegisterImplicitUseOperand(llvm::AMDGPU::EXEC)) {
    MI.addOperand(
        llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
  }
  return llvm::Error::success();
}

llvm::Expected<const LiftedRepresentation<hsa_executable_symbol_t> &>
luthier::CodeLifter::liftKernelSymbol(const hsa::ExecutableSymbol &Symbol) {
  if (!LiftedKernelSymbols.contains(Symbol)) {
    // Lift the executable if not already lifted
    // Start by creating an entry in the LiftedExecutables Map
    auto &LR = LiftedKernelSymbols.insert({Symbol, {}}).first->getSecond();
    // Assign the executable as the lifted primitive
    LR.LiftedPrimitive = Symbol.asHsaType();
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
    for (const auto & LCO: UsedLCOs) {
      LR.Modules.erase(LCO.hsaHandle());
      LR.RelatedLCOs.erase(LCO.hsaHandle());
    }
    for (const auto &[S, WasUsed]: UsageMap) {
      if (!WasUsed) {
        if (S.getType() == hsa::VARIABLE) {
          LR.RelatedGlobalVariables.erase(S.hsaHandle());
        } else {
          LR.RelatedFunctions.erase(S.hsaHandle());
        }
      }
    }
  }
  return LiftedKernelSymbols.at(Symbol);
}

llvm::Expected<const LiftedRepresentation<hsa_executable_t> &>
CodeLifter::liftExecutable(const hsa::Executable &Exec) {
  if (!LiftedExecutables.contains(Exec)) {
    // Lift the executable if not already lifted
    // Start by creating an entry in the LiftedExecutables Map
    auto &LR = LiftedExecutables.insert({Exec, {}}).first->getSecond();
    // Assign the executable as the lifted primitive
    LR.LiftedPrimitive = Exec.asHsaType();
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
  return LiftedExecutables.at(Exec);
}

} // namespace luthier