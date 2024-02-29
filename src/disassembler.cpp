#include "disassembler.hpp"

#include <GCNSubtarget.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/AsmPrinter.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSymbolELF.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>

#include <memory>

#include "error.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"
#include "hsa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "target_manager.hpp"

namespace luthier {

llvm::Expected<CodeLifter::DisassemblyInfo &>
luthier::CodeLifter::getDisassemblyInfo(const hsa::ISA &Isa) {
  if (!DisassemblyInfoMap.contains(Isa)) {
    auto TargetInfo = TargetManager::instance().getTargetInfo(Isa);
    LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

    auto TT = Isa.getLLVMTargetTriple();
    LUTHIER_RETURN_ON_ERROR(TT.takeError());

    auto Target = TargetInfo->getTarget();

    std::unique_ptr<llvm::MCContext> CTX(new (std::nothrow) llvm::MCContext(
        llvm::Triple(*TT), TargetInfo->getMCAsmInfo(),
        TargetInfo->getMCRegisterInfo(), TargetInfo->getMCSubTargetInfo()));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CTX != nullptr));

    std::unique_ptr<llvm::MCDisassembler> DisAsm(
        TargetInfo->getTarget()->createMCDisassembler(
            *(TargetInfo->getMCSubTargetInfo()), *CTX));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(DisAsm != nullptr));

    std::unique_ptr<llvm::MCRelocationInfo> RelInfo(
        Target->createMCRelocationInfo(*TT, *CTX));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(RelInfo != nullptr));

    auto [It, _] = DisassemblyInfoMap.insert(
        {Isa, DisassemblyInfo{
                  std::move(CTX), std::move(DisAsm), {}}});

    std::unique_ptr<llvm::MCSymbolizer> Symbolizer(Target->createMCSymbolizer(
        *TT, nullptr, nullptr, &It->second.Symbols, &(*(It->second.Context)),
        std::move(RelInfo)));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Symbolizer != nullptr));
    It->second.Symbolizer = Symbolizer.get();
    It->second.DisAsm->setSymbolizer(std::move(Symbolizer));
  }
  return DisassemblyInfoMap.at(Isa);
}

llvm::Expected<std::vector<llvm::MCInst>>
CodeLifter::disassemble(const hsa::ISA &Isa, llvm::ArrayRef<uint8_t> code) {
  auto DisassemblyInfo = getDisassemblyInfo(Isa);

  LUTHIER_RETURN_ON_ERROR(DisassemblyInfo.takeError());
  auto TargetInfo = TargetManager::instance().getTargetInfo(Isa);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());
  const auto &DisAsm = DisassemblyInfo->DisAsm;

  size_t maxReadSize = TargetInfo->getMCAsmInfo()->getMaxInstLength();
  size_t idx = 0;
  auto currentAddress = reinterpret_cast<luthier_address_t>(code.data());
  std::vector<llvm::MCInst> instructions;

  while (idx < code.size()) {
    size_t readSize =
        (idx + maxReadSize) < code.size() ? maxReadSize : code.size() - idx;
    size_t instSize{};
    llvm::MCInst inst;
    auto readBytes =
        arrayRefFromStringRef(toStringRef(code).substr(idx, readSize));
    if (DisAsm->getInstruction(inst, instSize, readBytes, currentAddress,
                               llvm::nulls()) !=
        llvm::MCDisassembler::Success) {

      llvm::report_fatal_error("Failed to disassemble instructions");
    }
    inst.setLoc(llvm::SMLoc::getFromPointer(
        reinterpret_cast<const char *>(currentAddress)));

    idx += instSize;
    currentAddress += instSize;
    instructions.push_back(inst);
  }
  llvm::outs() << "number of instructions: " << instructions.size() << "\n";

  return instructions;
}

llvm::Expected<const std::vector<hsa::Instr> *>
luthier::CodeLifter::disassemble(const hsa::ExecutableSymbol &Symbol,
                                 std::optional<hsa::ISA> Isa) {
  if (!DisassembledSymbolsRaw.contains(Symbol)) {
    LUTHIER_RETURN_ON_MOVE_INTO_FAIL(hsa_symbol_kind_t, SymbolType,
                                     Symbol.getType());
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ARGUMENT_ERROR_CHECK(SymbolType != HSA_SYMBOL_KIND_VARIABLE));

    if (!Isa.has_value()) {
      auto IsaOrError = Symbol.getAgent().getIsa();
      if (auto Err = IsaOrError.takeError()) {
        return Err;
      }
      //      LUTHIER_RETURN_ON_ERROR(IsaOrError.takeError());
      Isa.emplace(*IsaOrError);
    }

    auto CodeOrErrorOnDevice = Symbol.getMachineCode();
    LUTHIER_RETURN_ON_ERROR(CodeOrErrorOnDevice.takeError());
    auto CodeOrErrorOnHost = hsa::convertToHostEquivalent(*CodeOrErrorOnDevice);
    LUTHIER_RETURN_ON_ERROR(CodeOrErrorOnHost.takeError());

    llvm::Expected<std::vector<llvm::MCInst>> InstructionsOrError =
        disassemble(*Isa, *CodeOrErrorOnHost);
    LUTHIER_RETURN_ON_ERROR(InstructionsOrError.takeError());

    DisassembledSymbolsRaw.insert(
        {Symbol, std::make_unique<std::vector<hsa::Instr>>()});
    auto &Out = DisassembledSymbolsRaw.at(Symbol);
    Out->reserve(InstructionsOrError->size());
    for (auto &inst : *InstructionsOrError) {
      auto address =
          reinterpret_cast<luthier_address_t>(inst.getLoc().getPointer());
      inst.setLoc({});
      Out->push_back(hsa::Instr(inst, Symbol, address));
    }
  }
  return DisassembledSymbolsRaw.at(Symbol).get();
}

llvm::Expected<std::vector<llvm::MCInst>>
luthier::CodeLifter::disassemble(const llvm::object::ELFSymbolRef &Symbol,
                                 std::optional<size_t> Size) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(Symbol.getELFType() != llvm::ELF::STT_FUNC));

  auto triple = Symbol.getObject()->makeTriple();
  auto isa = hsa::ISA::fromName(triple.normalize().c_str());
  LUTHIER_RETURN_ON_ERROR(isa.takeError());

  auto addressOrError = Symbol.getAddress();
  LUTHIER_RETURN_ON_ERROR(addressOrError.takeError());

  llvm::StringRef code(reinterpret_cast<const char *>(*addressOrError),
                       Symbol.getSize());

  if (Size.has_value())
    code = code.substr(0, *Size > code.size() ? code.size() : *Size);

  return disassemble(*isa, llvm::arrayRefFromStringRef(code));
}

luthier::CodeLifter::~CodeLifter() {
  DisassemblyInfoMap.clear();
  KernelModules.clear();
  DisassembledSymbolsRaw.clear();
}

llvm::Expected<std::tuple<std::unique_ptr<llvm::Module>,
                          std::unique_ptr<llvm::MachineModuleInfoWrapperPass>>>
luthier::CodeLifter::liftKernelModule(const hsa::ExecutableSymbol &Symbol) {
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(hsa_symbol_kind_t, SymbolType,
                                   Symbol.getType());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(SymbolType == HSA_SYMBOL_KIND_KERNEL));

  //  if (!KernelModules.contains(Symbol)) {
  auto Agent = Symbol.getAgent();

  auto ISA = Agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());

  auto TargetInfo = luthier::TargetManager::instance().getTargetInfo(*ISA);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  auto TM = TargetInfo->getTargetMachine();

  auto MCInstInfo = TM->getMCInstrInfo();

  auto SymbolName = Symbol.getName();
  LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());

  auto Module = std::make_unique<llvm::Module>(*SymbolName,
                                               *TargetInfo->getLLVMContext());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Module != nullptr));

  Module->setDataLayout(TM->createDataLayout());

  auto LCO = Symbol.getExecutable().getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LCO.takeError());

  auto StorageMemory = (*LCO)[0].getStorageMemory();
  LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());

  auto storageElfOrError = getELFObjectFileBase(*StorageMemory);

  LUTHIER_RETURN_ON_ERROR(storageElfOrError.takeError());

  auto noteSectionOrError = getElfNoteMetadataRoot(storageElfOrError->get());
  LUTHIER_RETURN_ON_ERROR(noteSectionOrError.takeError());

  llvm::Type *const returnType = llvm::Type::getVoidTy(Module->getContext());
  LUTHIER_CHECK(returnType);

  llvm::Type *const memParamType =
      llvm::PointerType::get(llvm::Type::getInt32Ty(Module->getContext()),
                             llvm::AMDGPUAS::GLOBAL_ADDRESS);

  LUTHIER_CHECK(memParamType);
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(returnType, {memParamType}, false);
  LUTHIER_CHECK(FunctionType);
  llvm::Function *F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::ExternalLinkage,
      SymbolName->substr(0, SymbolName->size() - 3), *Module);
  LUTHIER_CHECK(F);
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

  llvm::outs() << "Number of arguments: " << F->arg_size() << "\n";

  // Very important to have a dummy IR BasicBlock; Otherwise Passes won't
  // function
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(Module->getContext(), "", F);
  LUTHIER_CHECK(BB);
  new llvm::UnreachableInst(Module->getContext(), BB);

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated
  auto kd = Symbol.getKernelDescriptor();
  LUTHIER_RETURN_ON_ERROR(kd.takeError());

  F->addFnAttr("amdgpu-lds-size",
               llvm::formatv("0, {0}", (*kd)->groupSegmentFixedSize).str());
  // Private (scratch) segment size is determined by Analysis Usage pass
  // Kern Arg is determined via analysis usage + args set earlier
  if ((*kd)->getKernelCodePropertiesEnableSgprDispatchId() == 0) {
    F->addFnAttr("amdgpu-no-dispatch-id");
  }
  if ((*kd)->getKernelCodePropertiesEnableSgprDispatchPtr() == 0) {
    F->addFnAttr("amdgpu-no-dispatch-ptr");
  }
  if ((*kd)->getKernelCodePropertiesEnableSgprQueuePtr() == 0) {
    F->addFnAttr("amdgpu-no-queue-ptr");
  }
  F->addFnAttr("amdgpu-no-workgroup-id-y");
  F->addFnAttr("amdgpu-no-workitem-id-y");
  F->addFnAttr("amdgpu-no-workgroup-id-z");
  F->addFnAttr("amdgpu-no-workitem-id-z");
  F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");
  F->addFnAttr("uniform-work-group-size", "true");

  llvm::outs() << "Preloaded Args: " << (*kd)->kernArgPreload << "\n";

  auto mmiwp = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TM);

  LUTHIER_CHECK(mmiwp);

  auto &MF = mmiwp->getMMI().getOrCreateMachineFunction(*F);

  MF.setAlignment(llvm::Align(4096));

  auto targetFunctionInstsOrError = CodeLifter::instance().disassemble(Symbol);

  LUTHIER_RETURN_ON_ERROR(targetFunctionInstsOrError.takeError());

  llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);
  MBB->setLabelMustBeEmitted();
  llvm::MCContext &MCContext = mmiwp->getMMI().getContext();

  for (unsigned int i = 0; i < (*targetFunctionInstsOrError)->size(); ++i) {
    auto inst = (*targetFunctionInstsOrError)->at(i);
    auto mcInst = inst.getInstr();
    const unsigned Opcode = mcInst.getOpcode();

    const llvm::MCInstrDesc &MCID = MCInstInfo->get(Opcode);
    llvm::MachineInstrBuilder Builder =
        llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
    Builder.getInstr();
    for (unsigned OpIndex = 0, E = mcInst.getNumOperands(); OpIndex < E;
         ++OpIndex) {
      const llvm::MCOperand &Op = mcInst.getOperand(OpIndex);
      if (Op.isReg()) {
        const bool IsDef = OpIndex < MCID.getNumDefs();
        unsigned Flags = 0;
        const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
        if (IsDef && !OpInfo.isOptionalDef())
          Flags |= llvm::RegState::Define;
        Builder.addReg(Op.getReg(), Flags);
      } else if (Op.isImm()) {

        //          if (MCID.isBranch()) {
        //
        //            Builder.addSym(MCContext.getOrCreateSymbol(".L123"),
        //                           llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET);
        //          } else {
        Builder.addImm(Op.getImm());
        //          }
        //                    Op.isExpr()
        //                    Builder.addBlockAddress
        //

      } else if (!Op.isValid()) {
        llvm_unreachable("Operand is not set");
      } else {
        llvm_unreachable("Not yet implemented");
      }
    }
    if (MCID.isBranch()) {

      mcInst.dump_pretty(llvm::outs(), TargetInfo->getMCInstPrinter(), " ",
                         TargetInfo->getMCRegisterInfo());
      llvm::outs() << llvm::formatv("Address: {0:x}\n", inst.getAddress());
      llvm::outs() << "Found a branch!\n";
      MBB = MF.CreateMachineBasicBlock();
      MF.push_back(MBB);
      MBB->setLabelMustBeEmitted();
    }
  }
  //        MBB->dump();

  llvm::outs() << "Number of blocks : " << MF.getNumBlockIDs() << "\n";
  //        MF.dump();
  auto TII = reinterpret_cast<const llvm::SIInstrInfo *>(
      TM->getSubtargetImpl(*F)->getInstrInfo());
  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM->getSubtargetImpl(*F)->getRegisterInfo());
  auto MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
  MFI->addPrivateSegmentBuffer(*TRI);
  MFI->addKernargSegmentPtr(*TRI);

  for (auto &BB : MF) {
    for (auto &Inst : BB) {
      //                llvm::outs() << "MIR: ";
      //
      std::string error;
      llvm::StringRef errorRef(error);
      bool isInstCorrect = TII->verifyInstruction(Inst, errorRef);
      //                llvm::outs() << "Is instruction correct: " <<
      //                isInstCorrect << "\n";
      if (!isInstCorrect) {

        //                    llvm::outs() << "May read Exec : " <<
        //                    TII->mayReadEXEC(MF.getRegInfo(), Inst) << "\n";
        Inst.addOperand(
            llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
        TII->fixImplicitOperands(Inst);
        Inst.addImplicitDefUseOperands(MF);
        //                    llvm::outs() << "After correction: ";
        //                    Inst.print(llvm::outs(), true, false, false,
        //                    true, TII);

        //                    llvm::outs() << "Is correct now: " <<
        //                    TII->verifyInstruction(Inst, errorRef) << "\n";
      }
      //                llvm::outs() << "Error: " << errorRef << "\n";
      /*                for (auto &op: Inst.operands()) {
                          if (op.isReg()) {
                              llvm::outs() << "Reg: ";
                              op.print(llvm::outs(), TRI);
                              llvm::outs() << "\n";
                              llvm::outs() << "is implicit: " <<
         op.isImplicit() << "\n";
                              //                        if (op.isImplicit() &&
         op.readsReg() && op.isUse() && op.getReg().id() ==
         llvm::AMDGPU::EXEC) {
                              // op.setImplicit(true);
                              //
                              //                        }
                          }
                      }*/
      //                llvm::outs() <<
      //                "==============================================================\n";
      Inst.print(llvm::outs(), true, false, false, true, TII);
    }
    llvm::outs() << "======================================\n";
  }
  llvm::MachineFunctionProperties &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Properties.reset(llvm::MachineFunctionProperties::Property::TracksLiveness);

  MF.getRegInfo().freezeReservedRegs(MF);

  //        auto elfOrError =
  //        getELFObjectFileBase(symbol.getExecutable().getLoadedCodeObjects()[0].getStorageMemory());
  //        if (llvm::errorToBool(elfOrError.takeError())) {
  //            llvm::report_fatal_error("Failed to parse the elf.");
  //        }
  //        auto noteOrError = getElfNoteMetadataRoot(elfOrError.get().get());
  //        if (llvm::errorToBool(noteOrError.takeError())) {
  //            llvm::report_fatal_error("Failed to parse the note section");
  //        }
  //        noteOrError.get().toYAML(llvm::outs());
  //        llvm::outs() << "\n";
  ////        MF.dump();
  //        properties.print(llvm::outs());
  //        llvm::outs() << "\n";

  return std::make_tuple(std::move(Module), std::move(mmiwp));
  //  }
}

} // namespace luthier