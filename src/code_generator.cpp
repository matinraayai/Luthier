#include "code_generator.hpp"

#include <memory>

#include <AMDGPUResourceUsageAnalysis.h>
#include <AMDGPUTargetMachine.h>
#include <amd_comgr/amd_comgr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/LegacyPassManager.h>

#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "error.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_isa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "log.hpp"
#include "target_manager.hpp"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include "SIInstrInfo.h"

// #define GET_REGINFO_ENUM
// #include "AMDGPUGenRegisterInfo.inc"
//
// #define GET_SUBTARGETINFO_ENUM
// #include "AMDGPUGenSubtargetInfo.inc"
//
// #define GET_INSTRINFO_ENUM
// #define GET_AVAILABLE_OPCODE_CHECKER

#include "AMDGPUGenInstrInfo.inc"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Instructions.h"

namespace luthier {

llvm::Error CodeGenerator::compileRelocatableToExecutable(
    const llvm::ArrayRef<uint8_t> &code, const hsa::ISA &isa,
    llvm::SmallVectorImpl<uint8_t> &out) {
  amd_comgr_data_t dataIn;
  amd_comgr_data_set_t dataSetIn, dataSetOut;
  amd_comgr_action_info_t dataAction;

  auto isaName = isa.getName();
  LUTHIER_RETURN_ON_ERROR(isaName.takeError());

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_create_data_set(&dataSetIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_set_data(
      dataIn, code.size(), reinterpret_cast<const char *>(code.data()))));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_set_data_name(dataIn, "source.o"))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_data_set_add(dataSetIn, dataIn))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_data_set(&dataSetOut))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_action_info(&dataAction))));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_isa_name(dataAction, isaName->c_str()))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_option_list(dataAction, nullptr, 0))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                           dataAction, dataSetIn, dataSetOut))));

  amd_comgr_data_t dataOut;
  size_t dataOutSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_action_data_get_data(
          dataSetOut, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &dataOut))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_get_data(dataOut, &dataOutSize, nullptr))));
  out.resize(dataOutSize);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_get_data(
      dataOut, &dataOutSize, reinterpret_cast<char *>(out.data())))));

  return llvm::Error::success();
}

llvm::Error CodeGenerator::compileRelocatableToExecutable(
    const llvm::ArrayRef<uint8_t> &code, const hsa::GpuAgent &agent,
    llvm::SmallVectorImpl<uint8_t> &out) {
  auto Isa = agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());
  return compileRelocatableToExecutable(code, *Isa, out);
}

template <class ELFT>
static llvm::Error
getRelocationValueString(const llvm::object::ELFObjectFile<ELFT> *Obj,
                         const llvm::object::RelocationRef &RelRef,
                         llvm::SmallVectorImpl<char> &Result) {
  const llvm::object::ELFFile<ELFT> &EF = Obj->getELFFile();
  llvm::object::DataRefImpl Rel = RelRef.getRawDataRefImpl();
  auto SecOrErr = EF.getSection(Rel.d.a);
  if (!SecOrErr)
    return SecOrErr.takeError();

  int64_t Addend = 0;
  // If there is no Symbol associated with the relocation, we set the undef
  // boolean value to 'true'. This will prevent us from calling functions that
  // requires the relocation to be associated with a symbol.
  //
  // In SHT_REL case we would need to read the addend from section data.
  // GNU objdump does not do that and we just follow for simplicity atm.
  bool Undef = false;
  if ((*SecOrErr)->sh_type == llvm::ELF::SHT_RELA) {
    const typename ELFT::Rela *ERela = Obj->getRela(Rel);
    Addend = ERela->r_addend;
    Undef = ERela->getSymbol(false) == 0;
  } else if ((*SecOrErr)->sh_type == llvm::ELF::SHT_REL) {
    const typename ELFT::Rel *ERel = Obj->getRel(Rel);
    Undef = ERel->getSymbol(false) == 0;
  } else {
    return llvm::make_error<llvm::object::BinaryError>();
  }

  // Default scheme is to print Target, as well as "+ <addend>" for nonzero
  // addend. Should be acceptable for all normal purposes.
  std::string FmtBuf;
  llvm::raw_string_ostream Fmt(FmtBuf);

  if (!Undef) {
    llvm::object::symbol_iterator SI = RelRef.getSymbol();
    llvm::Expected<const typename ELFT::Sym *> SymOrErr =
        Obj->getSymbol(SI->getRawDataRefImpl());
    // TODO: test this error.
    if (!SymOrErr)
      return SymOrErr.takeError();

    if ((*SymOrErr)->getType() == llvm::ELF::STT_SECTION) {
      llvm::Expected<llvm::object::section_iterator> SymSI = SI->getSection();
      if (!SymSI)
        return SymSI.takeError();
      const typename ELFT::Shdr *SymSec =
          Obj->getSection((*SymSI)->getRawDataRefImpl());
      auto SecName = EF.getSectionName(*SymSec);
      if (!SecName)
        return SecName.takeError();
      Fmt << *SecName;
    } else {
      llvm::Expected<llvm::StringRef> SymName = SI->getName();
      if (!SymName)
        return SymName.takeError();
      Fmt << *SymName;
    }
  } else {
    Fmt << "*ABS*";
  }
  if (Addend != 0) {
    Fmt << (Addend < 0 ? "-" : "+")
        << llvm::format("0x%" PRIx64,
                        (Addend < 0 ? -(uint64_t)Addend : (uint64_t)Addend));
  }
  Fmt.flush();
  Result.append(FmtBuf.begin(), FmtBuf.end());
  return llvm::Error::success();
}

llvm::Error luthier::CodeGenerator::instrument(hsa::Instr &instr,
                                               const void *deviceFunc,
                                               luthier_ipoint_t point) {
  LUTHIER_LOG_FUNCTION_CALL_START
  auto Agent = instr.getAgent();
  auto &CodeObjectManager = luthier::CodeObjectManager::instance();
  auto &contextManager = luthier::TargetManager::instance();

  auto Isa = Agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());

  auto TargetInfo = contextManager.getTargetInfo(*Isa);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  auto KernelName = instr.getExecutableSymbol().getName();
  LUTHIER_RETURN_ON_ERROR(KernelName.takeError());

  auto TM = TargetInfo->getTargetMachine();

  auto Module = std::make_unique<llvm::Module>(*KernelName,
                                               *TargetInfo->getLLVMContext());



  Module->setDataLayout(TM->createDataLayout());

  auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TM);
  llvm::SmallVector<char> reloc;
  llvm::SmallVector<uint8_t> executable;



  LUTHIER_CHECK(MMIWP);


  LUTHIER_RETURN_ON_ERROR(luthier::CodeLifter::instance().liftAndAddToModule(
      instr.getExecutableSymbol(), *Module, MMIWP->getMMI()));

  auto MCInstInfo = TM->getMCInstrInfo();

  llvm::Expected<hsa::ExecutableSymbol> instrumentationFunc =
      CodeObjectManager.getInstrumentationFunction(deviceFunc, Agent);
  LUTHIER_RETURN_ON_ERROR(instrumentationFunc.takeError());

  // Get the Symbol



//  for (auto &F : *Module) {
//    auto MF = MMIWP->getMMI().getMachineFunction(F);
//    auto TII = reinterpret_cast<const llvm::SIInstrInfo *>(
//        TM->getSubtargetImpl(F)->getInstrInfo());
////    MF->getProperties().reset(
////        llvm::MachineFunctionProperties::Property::NoVRegs);
//    auto &MRI = MF->getRegInfo();
//    auto MBBIter = MF->end();
//    MBBIter--;
//    auto& MBB = *MBBIter;
////      llvm::Register PCReg =
////          MRI.createVirtualRegister(&llvm::AMDGPU::SReg_64RegClass);
//      auto I = MBB.end();
//      I--;
//
//      // We need to compute the offset relative to the instruction immediately
//      // after s_getpc_b64. Insert pc arithmetic code before last terminator.
////      llvm::MachineInstr *GetPC =
////          BuildMI(MBB, I, llvm::DebugLoc(),
////                  MCInstInfo->get(llvm::AMDGPU::S_GETPC_B64), llvm::AMDGPU::SGPR4_SGPR5);
//
//
//
////      auto SymLow = llvm::MachineOperand::CreateES(llvm::cantFail(instrumentationFunc->getName()).c_str(), llvm::SIInstrInfo::MO_REL32_LO);
//
////      auto SymHigh = llvm::MachineOperand::CreateES(llvm::cantFail(instrumentationFunc->getName()).c_str(), llvm::SIInstrInfo::MO_REL32_HI);
//
//      llvm::outs() << llvm::cantFail(instrumentationFunc->getName()).c_str() << "\n";
//
//      auto Address = reinterpret_cast<luthier_address_t>(llvm::cantFail(instrumentationFunc->getMachineCode()).data());
//      llvm::MachineInstrBuilder Builder = BuildMI(MBB, I, llvm::DebugLoc(), MCInstInfo->get(llvm::AMDGPU::S_MOV_B32_vi))
//          .addReg(llvm::AMDGPU::SGPR4, llvm::RegState::Define)
//                                              .addImm(static_cast<int64_t>(Address & 0xFFFFFFFF));
////          .addReg(llvm::AMDGPU::SGPR4, llvm::RegState::Define);
//
//      std::string error;
//      llvm::StringRef errorRef(error);
//
//      bool isInstCorrect = TII->verifyInstruction(*Builder, errorRef);
//
//      llvm::outs() << "Is inst correct? " << isInstCorrect << "\n";
//      llvm::outs() << errorRef << "\n";
//
//      Builder = BuildMI(MBB, I, llvm::DebugLoc(), MCInstInfo->get(llvm::AMDGPU::S_MOV_B32_vi))
//          .addReg(llvm::AMDGPU::SGPR5, llvm::RegState::Define)
//                .addImm(static_cast<int64_t>(Address >> 32));
////          .addReg(llvm::AMDGPU::SGPR5, llvm::RegState::Define);
////      Builder->addOperand(SymHigh);
////
//      isInstCorrect = TII->verifyInstruction(*Builder, errorRef);
////
//      llvm::outs() << "Is inst correct? " << isInstCorrect << "\n";
//      llvm::outs() << errorRef << "\n";
//
//      const llvm::MCInstrDesc &MCID = MCInstInfo->get(llvm::AMDGPU::S_SWAPPC_B64_vi);
//      Builder =
//          llvm::BuildMI(MBB, I, llvm::DebugLoc(), MCID).addReg(llvm::AMDGPU::SGPR30_SGPR31).addReg(llvm::AMDGPU::SGPR4_SGPR5);
//
//      isInstCorrect = TII->verifyInstruction(*Builder, errorRef);
//
//      llvm::outs() << "Is inst correct? " << isInstCorrect << "\n";
//      llvm::outs() << errorRef << "\n";
//      I--;
//      auto LastMBB = MBB.splitAt(*I);
//      MF->getFrameInfo().setHasCalls(true);
//    MF->print(llvm::outs());
//  }
  //  Module->getFunctionList().begin()->ge
  //  MMIWP->getMMI().get
  llvm::MCContext &MCContext = MMIWP->getMMI().getContext();

  //  auto TM = targetInfo->getTargetMachine();

  llvm::legacy::PassManager PM;

  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module->getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  llvm::TargetPassConfig *TPC = TM->createPassConfig(PM);

  PM.add(TPC);
//  TPC->addISelPasses();
//  TPC->addPrintPass("My stuff:");
  PM.add(MMIWP.release());
  //        TPC->printAndVerify("MachineFunctionGenerator::assemble");

  //        auto usageAnalysis =
  //        std::make_unique<llvm::AMDGPUResourceUsageAnalysis>();
  PM.add(new llvm::AMDGPUResourceUsageAnalysis());
  // Add target-specific passes.
  //        ET.addTargetSpecificPasses(PM);
  //        TPC->printAndVerify("After
  //        ExegesisTarget::addTargetSpecificPasses");
  // Adding the following passes:
  // - postrapseudos: expands pseudo return instructions used on some targets.
  // - machineverifier: checks that the MachineFunction is well formed.
  // - prologepilog: saves and restore callee saved registers.
  //        for (const char *PassName :
  //             {"postrapseudos", "machineverifier", "prologepilog"})
  //            if (addPass(PM, PassName, *TPC))
  //                return make_error<Failure>("Unable to add a mandatory
  //                pass");
  TPC->setInitialized();

  //        llvm::SmallVector<char> o;
  //        std::string out;
  //        llvm::raw_string_ostream outOs(out);
  llvm::raw_svector_ostream OutOS(reloc);

  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM->addAsmPrinter(PM, OutOS, nullptr, llvm::CodeGenFileType::ObjectFile,
                        MCContext))
    llvm::outs() << "Failed to add pass manager\n";
  //            return make_error<llvm::Failure>("Cannot add AsmPrinter
  //            passes");

  PM.run(*Module); // Run all the passes

  llvm::outs() << reloc << "\n";

  LUTHIER_RETURN_ON_ERROR(compileRelocatableToExecutable(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(reloc.data()),
                              reloc.size()),
      *Isa, executable));
  llvm::outs() << "Compiled to executable\n";



  llvm::outs() << llvm::toStringRef(executable) << "\n";

  //  llvm::outs() << elfFile->get()->getRelSection();
  LUTHIER_RETURN_ON_ERROR(CodeObjectManager.loadInstrumentedKernel(
      executable, instr.getExecutableSymbol()));

  //  llvm::Expected<const std::vector<hsa::Instr> *> InstFunction =
  //      CodeLifter::instance().disassemble( instr.getExecutableSymbol());
  //    for (const auto& i: *targetFunction) {
  //        const unsigned Opcode = i.getInstr().getOpcode();
  //        const llvm::MCInstrDesc &MCID =
  //        targetInfo.getMCInstrInfo()->get(Opcode); llvm::MachineInstrBuilder
  //        Builder = llvm::BuildMI(MBB, DL, MCID); for (unsigned OpIndex = 0, E
  //        = i.getInstr().getNumOperands(); OpIndex < E;
  //             ++OpIndex) {
  //            const MCOperand &Op = Inst.getOperand(OpIndex);
  //            if (Op.isReg()) {
  //                const bool IsDef = OpIndex < MCID.getNumDefs();
  //                unsigned Flags = 0;
  //                const MCOperandInfo &OpInfo =
  //                MCID.operands().begin()[OpIndex]; if (IsDef &&
  //                !OpInfo.isOptionalDef())
  //                    Flags |= RegState::Define;
  //                Builder.addReg(Op.getReg(), Flags);
  //            } else if (Op.isImm()) {
  //                Builder.addImm(Op.getImm());
  //            } else if (!Op.isValid()) {
  //                llvm_unreachable("Operand is not set");
  //            } else {
  //                llvm_unreachable("Not yet implemented");
  //            }
  //        }
  //    }

  //    new UnreachableInst(Module->getContext(), BB);
  //    return MMI->getOrCreateMachineFunction(*F);

  //        llvm::Function f;
  //    llvm::MachineFunction func;
  //    auto genInst = makeInstruction(
  //        agent.getIsa(), llvm::AMDGPU::S_ADD_I32,
  //        llvm::MCOperand::createReg(llvm::AMDGPU::VGPR3),
  //        llvm::MCOperand::createReg(llvm::AMDGPU::SGPR6),
  //        llvm::MCOperand::createReg(llvm::AMDGPU::VGPR1));

  //    const auto targetTriple =
  //    llvm::Triple(agent.getIsa().getLLVMTargetTriple()); llvm::MCContext
  //    ctx(targetTriple, targetInfo.getMCAsmInfo(),
  //    targetInfo.getMCRegisterInfo(),
  //                        targetInfo.getMCSubTargetInfo());
  //    auto mcEmitter =
  //    targetInfo.getTarget()->createMCCodeEmitter(*targetInfo.getMCInstrInfo(),
  //    ctx);

  //    llvm::SmallVector<char> osBack;
  //    llvm::SmallVector<llvm::MCFixup> fixUps;
  //    mcEmitter->encodeInstruction(genInst, osBack, fixUps,
  //    *targetInfo.getMCSubTargetInfo()); fmt::print("Assembled instruction
  //    :"); for (auto s: osBack) { fmt::print("{:#x}", s); } fmt::print("\n");

  //    auto oneMoreTime = Disassembler::instance().disassemble(
  //        agent.getIsa(), {reinterpret_cast<std::byte *>(osBack.data()),
  //        osBack.size()});

  auto instFunctionInstructions =
      CodeLifter::instance().disassemble(*instrumentationFunc);
  LUTHIER_RETURN_ON_ERROR(instFunctionInstructions.takeError());
  //
  for (const auto &i : **instFunctionInstructions) {
    std::string instStr;
    llvm::raw_string_ostream instStream(instStr);
    auto inst = i.getInstr();
    TargetInfo->getMCInstPrinter()->printInst(
        &inst, reinterpret_cast<luthier_address_t>(i.getAddress()), "",
        *TargetInfo->getMCSubTargetInfo(), llvm::outs());
    llvm::outs() << "\n";
  }
  //            targetInfo.getMCInstrInfo()->getName(inst.getOpcode()).str() <<
  //        "\n";
  //        //        fmt::println("Is call? {}",
  //        targetInfo.MII_->get(inst.getOpcode()).isCall());
  //        //        fmt::println("Is control flow? {}",
  //        //
  //        targetInfo.MII_->get(inst.getOpcode()).mayAffectControlFlow(inst,
  //        *targetInfo.MRI_));
  //        //        fmt::println("Num max operands? {}",
  //        targetInfo.MII_->get(inst.getOpcode()).getNumOperands());
  //        llvm::outs() << llvm::formatv("Num operands? {0:X}\n",
  //        inst.getNumOperands());
  //        //        fmt::println("May load? {}",
  //        targetInfo.MII_->get(inst.getOpcode()).mayLoad()); llvm::outs() <<
  //        llvm::formatv("Mnemonic: {0}\n",
  //        targetInfo.getMCInstPrinter()->getMnemonic(&inst).first);
  //
  //        for (int j = 0; j < inst.getNumOperands(); j++) {
  //            //            auto op = inst.getOperand(j);
  //            //            std::string opStr;
  //            //            llvm::raw_string_ostream opStream(opStr);
  //            //            op.print(opStream, targetInfo.MRI_.get());
  //            //            fmt::println("OP: {}", opStream.str());
  //            //            if (op.isReg()) {
  //            //                fmt::println("Reg idx: {}", op.getReg());
  //            //                fmt::println("Reg name: {}",
  //            targetInfo.MRI_->getName(op.getReg()));
  //            //                auto subRegIterator =
  //            targetInfo.MRI_->subregs(op.getReg());
  //            //                for (auto it = subRegIterator.begin(); it !=
  //            subRegIterator.end(); it++) {
  //            //                    fmt::println("\tSub reg Name: {}",
  //            targetInfo.MRI_->getName((*it)));
  //            //                }
  //            ////                auto superRegIterator =
  //            targetInfo.MRI_->superregs(op.getReg());
  //            ////                for (auto it = superRegIterator.begin();
  //            it
  //            != superRegIterator.end(); it++) {
  //            ////                    fmt::println("\tSuper reg Name: {}",
  //            targetInfo.MRI_->getName((*it)));
  //            ////                }
  //            //            }
  //            //            fmt::println("Is op {} reg? {}", j, op.isReg());
  //            //            fmt::println("Is op {} valid? {}", j,
  //            op.isValid());
  //        }

  //        llvm::outs() << instStream.str() << "\n";
  //        inst.getOpcode();
  //    }
  //    llvm::outs() << "==================================\n";*/
  return llvm::Error::success();
}

} // namespace luthier
