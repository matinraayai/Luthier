#include "code_generator.hpp"

#include <memory>

#include <AMDGPUResourceUsageAnalysis.h>
#include <AMDGPUTargetMachine.h>
#include <amd_comgr/amd_comgr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/PassRegistry.h>


#include <AMDGPU.h>
#include "SIInstrInfo.h"
#include <SIMachineFunctionInfo.h>
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
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>

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
    const llvm::ArrayRef<uint8_t> &Code, const hsa::ISA &Isa,
    llvm::SmallVectorImpl<uint8_t> &Out) {
  amd_comgr_data_t DataIn;
  amd_comgr_data_set_t DataSetIn, dataSetOut;
  amd_comgr_action_info_t DataAction;

  auto IsaName = Isa.getName();
  LUTHIER_RETURN_ON_ERROR(IsaName.takeError());

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_create_data_set(&DataSetIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataIn)));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_set_data(
      DataIn, Code.size(), reinterpret_cast<const char *>(Code.data()))));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_set_data_name(DataIn, "source.o"))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_data_set_add(DataSetIn, DataIn))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_data_set(&dataSetOut))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_action_info(&DataAction))));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_isa_name(DataAction, IsaName->c_str()))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_option_list(DataAction, nullptr, 0))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                           DataAction, DataSetIn, dataSetOut))));

  amd_comgr_data_t DataOut;
  size_t DataOutSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_action_data_get_data(
          dataSetOut, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataOut))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_get_data(DataOut, &DataOutSize, nullptr))));
  Out.resize(DataOutSize);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_get_data(
      DataOut, &DataOutSize, reinterpret_cast<char *>(Out.data())))));

  return llvm::Error::success();
}

llvm::Error CodeGenerator::compileRelocatableToExecutable(
    const llvm::ArrayRef<uint8_t> &Code, const hsa::GpuAgent &agent,
    llvm::SmallVectorImpl<uint8_t> &Out) {
  auto Isa = agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());
  return compileRelocatableToExecutable(Code, *Isa, Out);
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

llvm::Error luthier::CodeGenerator::instrument(hsa::Instr &Instr,
                                               const void *DeviceFunc,
                                               luthier_ipoint_t Point) {
  LUTHIER_LOG_FUNCTION_CALL_START
  auto Agent = Instr.getAgent();
  auto &CodeObjectManager = luthier::CodeObjectManager::instance();
  auto &ContextManager = luthier::TargetManager::instance();

  auto Isa = Agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());

  auto TargetInfo = ContextManager.getTargetInfo(*Isa);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  auto KernelName = Instr.getExecutableSymbol().getName();
  LUTHIER_RETURN_ON_ERROR(KernelName.takeError());

  auto TM = TargetInfo->getTargetMachine();

  auto Module = std::make_unique<llvm::Module>(*KernelName,
                                               *TargetInfo->getLLVMContext());

  Module->setDataLayout(TM->createDataLayout());

  auto MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(TM);
  llvm::SmallVector<char> Reloc;
  llvm::SmallVector<uint8_t> Executable;

  LUTHIER_CHECK(MMIWP);

  LUTHIER_RETURN_ON_ERROR(luthier::CodeLifter::instance().liftAndAddToModule(
      Instr.getExecutableSymbol(), *Module, MMIWP->getMMI()));

//  llvm::Expected<hsa::ExecutableSymbol> InstrumentationFunc =
//      CodeObjectManager.getInstrumentationFunction(DeviceFunc, Agent);
//  LUTHIER_RETURN_ON_ERROR(InstrumentationFunc.takeError());
//
//  LUTHIER_RETURN_ON_ERROR(luthier::CodeLifter::instance().liftAndAddToModule(
//      *InstrumentationFunc, *Module, MMIWP->getMMI()
//      ));
//
//  auto MCInstInfo = TM->getMCInstrInfo();

  // Get the Symbol
//  auto InstLLVMFunction = Module->getFunction(llvm::cantFail(InstrumentationFunc->getName()));
////  for (auto& MBB: *MMIWP->getMMI().getMachineFunction(*InstLLVMFunction)) {
////    for (auto& MI: MBB) {
////      if (MI.getOpcode() == llvm::AMDGPU::S_GETPC_B64_vi) {
////        MI.getNextNode()
////      }
////    }
////  }
//  for (auto &F : *Module) {
//    if (F.getName() == KernelName->substr(0, KernelName->rfind(".kd"))) {
//      auto MF = MMIWP->getMMI().getMachineFunction(F);
//      MF->getFrameInfo().setHasCalls(true);
//      MF->getProperties().set(llvm::MachineFunctionProperties::Property::TracksLiveness);
//      auto MFI = MF->getInfo<llvm::SIMachineFunctionInfo>();
//      //    MFI->setScratchRSrcReg(llvm::AMDGPU::SGPR6_SGPR7);
//      auto TII = reinterpret_cast<const llvm::SIInstrInfo *>(
//          TM->getSubtargetImpl(F)->getInstrInfo());
//      //    MF->getProperties().reset(
//      //        llvm::MachineFunctionProperties::Property::NoVRegs);
//      auto &MRI = MF->getRegInfo();
//      auto MBBIter = MF->end();
//      MBBIter--;
//      auto &MBB = *MBBIter;
//      //      llvm::Register PCReg =
//      //          MRI.createVirtualRegister(&llvm::AMDGPU::SReg_64RegClass);
//      auto I = MBB.end();
//      I--;
//      llvm::MachineInstrBuilder Builder; // =
//      BuildMI(MBB, I, llvm::DebugLoc(),
//              MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKUP))
//          .addImm(0)
//          .addImm(0);
//      auto Address = reinterpret_cast<luthier_address_t>(
//          llvm::cantFail(InstrumentationFunc->getMachineCode()).data());
//      llvm::outs() << llvm::formatv("Address: {0:x}\n", Address);
//
//      auto MIB = llvm::BuildMI(MBB, I, llvm::DebugLoc(), MCInstInfo->get(llvm::AMDGPU::SI_PC_ADD_REL_OFFSET))
//                                    .addDef(llvm::AMDGPU::SGPR4_SGPR5, llvm::RegState::Define);
//      MIB.addGlobalAddress(InstLLVMFunction, 0, llvm::SIInstrInfo::MO_REL32);
//      MIB.addGlobalAddress(InstLLVMFunction, 0, llvm::SIInstrInfo::MO_REL32 + 1);
//
//      Builder = BuildMI(MBB, I, llvm::DebugLoc(),
//                        MCInstInfo->get(llvm::AMDGPU::SI_CALL_ISEL))
//                    .addReg(llvm::AMDGPU::SGPR4_SGPR5,
//                            llvm::RegState::Kill).addGlobalAddress(InstLLVMFunction);
////      Builder; // =
//      BuildMI(MBB, I, llvm::DebugLoc(),
//              MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKDOWN))
//          .addImm(0)
//          .addImm(0);
//    }


//    //	s_add_u32 s4, s4, myCounter@rel32@lo+4
//    //	s_addc_u32 s5, s5, myCounter@rel32@hi+12
//    //	global_load_dwordx2 v[0:1], v0, s[4:5]
//    //	s_waitcnt vmcnt(0)
//    //	global_load_dword v2, v[0:1], off
//    //	s_waitcnt vmcnt(0)
//    //	v_add_u32_e32 v2, 1, v2
//    //	global_store_dword v[0:1], v2, off
//    //	s_waitcnt vmcnt(0)
//
//
//    //
//    //      // We need to compute the offset relative to the instruction
//    //      immediately
//    //      // after s_getpc_b64. Insert pc arithmetic code before last
//    //      terminator.
//    //    llvm::MachineInstr *GetPC =
//    //        BuildMI(MBB, I, llvm::DebugLoc(),
//    //                MCInstInfo->get(llvm::AMDGPU::S_GETPC_B64),
//    //                llvm::AMDGPU::SGPR4_SGPR5);
//    //
//    //
//    //
//    ////      auto SymLow =
//    /// llvm::MachineOperand::CreateES(llvm::cantFail(instrumentationFunc->getName()).c_str(),
//    /// llvm::SIInstrInfo::MO_REL32_LO);
//    //
//    ////      auto SymHigh =
//    /// llvm::MachineOperand::CreateES(llvm::cantFail(instrumentationFunc->getName()).c_str(),
//    /// llvm::SIInstrInfo::MO_REL32_HI);
//    //
//    //      llvm::outs() <<
//    //      llvm::cantFail(instrumentationFunc->getName()).c_str() << "\n";
//    //
//

////    Builder->addOperand(llvm::MachineOperand::CreateReg(
////        llvm::AMDGPU::EXEC, false, false, false, true));
//    std::string error;
//    llvm::StringRef errorRef(error);
//    //
////    bool isInstCorrect = TII->verifyInstruction(*Builder, errorRef);
//    //
////    llvm::outs() << "Is inst correct? " << isInstCorrect << "\n";
////    llvm::outs() << errorRef << "\n";
//

    ////          .addReg(llvm::AMDGPU::SGPR4, llvm::RegState::Define);
//
//    Builder = BuildMI(MBB, I, llvm::DebugLoc(),
//                      MCInstInfo->get(llvm::AMDGPU::S_MOV_B32_vi))
//                  .addReg(llvm::AMDGPU::SGPR5, llvm::RegState::Define)
//                  .addImm(static_cast<int64_t>(Address >> 32));
//
//    auto VirtReg = MRI.createVirtualRegister(&llvm::AMDGPU::SReg_128RegClass);
//    Builder = BuildMI(MBB, I, llvm::DebugLoc(),
//                      MCInstInfo->get(llvm::AMDGPU::WWM_COPY))
//                  .addReg(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, llvm::RegState::Define)
//                  .addReg(llvm::AMDGPU::PRIVATE_RSRC_REG);

//    //    .addReg(llvm::AMDGPU::SGPR5, llvm::RegState::Define)
//    //        .addImm(static_cast<int64_t>(Address >> 32));
//


            //
//            //          .addReg(llvm::AMDGPU::SGPR5, llvm::RegState::Define);
//            ////      Builder->addOperand(SymHigh);
//            ////
//            bool isInstCorrect = TII->verifyInstruction(*Builder, errorRef);
//    ////
//    //      llvm::outs() << "Is inst correct? " << isInstCorrect << "\n";
//    //      llvm::outs() << errorRef << "\n";
//    Builder = llvm::BuildMI(MBB, I, llvm::DebugLoc(),
//                            MCInstInfo->get(llvm::AMDGPU::S_SWAPPC_B64_vi))
//                  .addReg(llvm::AMDGPU::SGPR30_SGPR31, llvm::RegState::Define)
//                  .addReg(llvm::AMDGPU::SGPR4_SGPR5);
//
//    isInstCorrect = TII->verifyInstruction(*Builder, errorRef);
//    //
//    llvm::outs() << "Is inst correct? " << isInstCorrect << "\n";
//    llvm::outs() << errorRef << "\n";
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
  TM->setOptLevel(llvm::CodeGenOptLevel::None);

  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module->getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  llvm::TargetPassConfig *TPC = TM->createPassConfig(PM);

  PM.add(TPC);
  //  TPC->addISelPasses();
  //  TPC->addPrintPass("My stuff:");
  PM.add(MMIWP.release());
  TPC->addISelPasses();
  PM.add(llvm::createMachineFunctionPrinterPass(llvm::outs(), "After ISEL Passes"));
  TPC->printAndVerify("After ISEL Passes");
//  PM.add(llvm::Pass::createPass(&llvm::FinalizeISelID));
//  PM.add(llvm::Pass::createPass(&llvm::SIFixSGPRCopiesID));
//  PM.add(llvm::createSILowerI1CopiesPass());
  TPC->addMachinePasses();
  PM.add(llvm::createMachineFunctionPrinterPass(llvm::outs(), "After Machine Passes"));
//  TPC->printAndVerify("After Machine Passes");

  //  TPC->addPass(&llvm::FinalizeISelID);

  //        TPC->printAndVerify("MachineFunctionGenerator::assemble");

  //        auto usageAnalysis =
  //        std::make_unique<llvm::AMDGPUResourceUsageAnalysis>();
  auto UsageAnalysis = new llvm::AMDGPUResourceUsageAnalysis();
  PM.add(UsageAnalysis);
  PM.add(llvm::createMachineFunctionPrinterPass(llvm::outs(), "After Usage Analysis"));
//  TPC->printAndVerify("After Resource Usage");
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
//  TPC->setInitialized();

  //        llvm::SmallVector<char> o;
  //        std::string out;
  //        llvm::raw_string_ostream outOs(out);

  TPC->setInitialized();
  llvm::raw_svector_ostream OutOS(Reloc);

  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM->addAsmPrinter(PM, OutOS, nullptr, llvm::CodeGenFileType::AssemblyFile,
                        MCContext))
    llvm::outs() << "Failed to add pass manager\n";
  //            return make_error<llvm::Failure>("Cannot add AsmPrinter
  //            passes");

  PM.run(*Module); // Run all the passes

  llvm::outs() << Reloc << "\n";

  LUTHIER_RETURN_ON_ERROR(compileRelocatableToExecutable(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(Reloc.data()),
                              Reloc.size()),
      *Isa, Executable));
  llvm::outs() << "Compiled to executable\n";

  llvm::outs() << llvm::toStringRef(Executable) << "\n";

  //  llvm::outs() << elfFile->get()->getRelSection();
  LUTHIER_RETURN_ON_ERROR(CodeObjectManager.loadInstrumentedKernel(
      Executable, Instr.getExecutableSymbol()));

//  auto instFunctionInstructions =
//      CodeLifter::instance().disassemble(*InstrumentationFunc);
//  LUTHIER_RETURN_ON_ERROR(instFunctionInstructions.takeError());
//  //
//  for (const auto &i : **instFunctionInstructions) {
//    std::string instStr;
//    llvm::raw_string_ostream instStream(instStr);
//    auto inst = i.getInstr();
//    TargetInfo->getMCInstPrinter()->printInst(
//        &inst, reinterpret_cast<luthier_address_t>(i.getAddress()), "",
//        *TargetInfo->getMCSubTargetInfo(), llvm::outs());
//    llvm::outs() << "\n";
//  }
  return llvm::Error::success();
}

} // namespace luthier
