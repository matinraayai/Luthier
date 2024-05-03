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
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include "SIInstrInfo.h"
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
#include <AMDGPU.h>
#include <SIMachineFunctionInfo.h>
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
    const llvm::ArrayRef<uint8_t> &Code, const hsa::ISA &ISA,
    llvm::SmallVectorImpl<uint8_t> &Out) {
  amd_comgr_data_t DataIn;
  amd_comgr_data_set_t DataSetIn, DataSetOut;
  amd_comgr_action_info_t DataAction;

  auto IsaName = ISA.getName();
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
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_data_set(&DataSetOut))));

  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_create_action_info(&DataAction))));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_isa_name(DataAction, IsaName->c_str()))));
  //  std::vector<const char *> MyOptions{"-Wl",
  //  "--unresolved-symbols=ignore-all", "-shared", "--undefined-glob=1"};
  const char *MyOptions[]{"-Wl,--unresolved-symbols=ignore-all"};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_option_list(DataAction, MyOptions, 1))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                           DataAction, DataSetIn, DataSetOut))));

  amd_comgr_data_t DataOut;
  size_t DataOutSize;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_action_data_get_data(
          DataSetOut, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataOut))));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_get_data(DataOut, &DataOutSize, nullptr))));
  Out.resize(DataOutSize);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK((amd_comgr_get_data(
      DataOut, &DataOutSize, reinterpret_cast<char *>(Out.data())))));

  return llvm::Error::success();
}

llvm::Error CodeGenerator::compileRelocatableToExecutable(
    const llvm::ArrayRef<uint8_t> &Code, const hsa::GpuAgent &Agent,
    llvm::SmallVectorImpl<uint8_t> &Out) {
  auto Isa = Agent.getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());
  return compileRelocatableToExecutable(Code, *Isa, Out);
}

llvm::Error CodeGenerator::instrument(
    std::unique_ptr<llvm::Module> Module,
    std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP,
    const LiftedSymbolInfo &LSO, luthier::InstrumentationTask &ITask) {
  LUTHIER_LOG_FUNCTION_CALL_START
  auto Symbol = hsa::ExecutableSymbol::fromHandle(LSO.getSymbol());
  auto Agent = Symbol.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  auto Isa = Agent->getIsa();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());
  auto &CodeObjectManager = luthier::CodeObjectManager::instance();
  auto &ContextManager = luthier::TargetManager::instance();
  auto TargetInfo = ContextManager.getTargetInfo(*Isa);
  LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());

  auto &TM = *TargetInfo->getTargetMachine();

  llvm::SmallVector<char> Reloc;
  llvm::SmallVector<uint8_t> Executable;

  llvm::MCContext &MCContext = MMIWP->getMMI().getContext();

  auto AddedLSIs = applyInstrumentation(*Module, MMIWP->getMMI(), LSO, ITask);
  LUTHIER_RETURN_ON_ERROR(AddedLSIs.takeError());
  //  auto TM = targetInfo->getTargetMachine();

  llvm::legacy::PassManager PM;
  TM.setOptLevel(llvm::CodeGenOptLevel::None);

  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module->getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  llvm::TargetPassConfig *TPC = TM.createPassConfig(PM);

  PM.add(TPC);
  PM.add(MMIWP.release());
  TPC->addISelPasses();
  TPC->addMachinePasses();
  auto UsageAnalysis = new llvm::AMDGPUResourceUsageAnalysis();
  PM.add(UsageAnalysis);
  TPC->setInitialized();
  llvm::raw_svector_ostream OutOS(Reloc);

  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM.addAsmPrinter(PM, OutOS, nullptr, llvm::CodeGenFileType::ObjectFile,
                       MCContext))
    llvm::outs() << "Failed to add pass manager\n";

  PM.run(*Module); // Run all the passes
  std::error_code code;
  llvm::raw_fd_ostream("my_reloc.hsaco", code) << Reloc;

  LUTHIER_RETURN_ON_ERROR(compileRelocatableToExecutable(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(Reloc.data()),
                              Reloc.size()),
      *Isa, Executable));
  llvm::outs() << "Compiled to executable\n";

  llvm::outs() << llvm::toStringRef(Executable) << "\n";

  std::vector<hsa::ExecutableSymbol> ExternGVs;
  for (const auto &GV : LSO.getRelatedVariables())
    ExternGVs.push_back(hsa::ExecutableSymbol::fromHandle(GV));

  for (const auto &L : *AddedLSIs)
    for (const auto &GV : L.getRelatedVariables())
      ExternGVs.push_back(hsa::ExecutableSymbol::fromHandle(GV));

  LUTHIER_RETURN_ON_ERROR(
      CodeObjectManager.loadInstrumentedKernel(Executable, Symbol, ExternGVs));

  return llvm::Error::success();
}

llvm::Expected<std::vector<LiftedSymbolInfo>>
CodeGenerator::applyInstrumentation(llvm::Module &Module,
                                    llvm::MachineModuleInfo &MMI,
                                    const LiftedSymbolInfo &LSO,
                                    const InstrumentationTask &ITask) {
  return insertFunctionCalls(Module, MMI, LSO, ITask.getInsertCallTasks());
}

llvm::Expected<std::vector<LiftedSymbolInfo>>
CodeGenerator::insertFunctionCalls(
    llvm::Module &Module, llvm::MachineModuleInfo &MMI,
    const LiftedSymbolInfo &LSI,
    const InstrumentationTask::insert_call_tasks &Tasks) {
  std::vector<LiftedSymbolInfo> Out;
  for (const auto &[MI, V] : Tasks) {
    auto &[DevFunc, IPoint] = V;
    const auto &HsaInst = LSI.getHSAInstrOfMachineInstr(*MI);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(HsaInst.has_value()));
    auto Agent = (**HsaInst).getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());
    auto InstrumentationFunction =
        luthier::CodeObjectManager::instance().getInstrumentationFunction(
            DevFunc, hsa::GpuAgent(*Agent));
    LUTHIER_RETURN_ON_ERROR(InstrumentationFunction.takeError());
    auto IFLSI = luthier::CodeLifter::instance().liftSymbolAndAddToModule(
        *InstrumentationFunction, Module, MMI);
    LUTHIER_RETURN_ON_ERROR(IFLSI.takeError());

    auto MCInstInfo = MMI.getTarget().getMCInstrInfo();
    MI->getParent()->getParent()->getProperties().reset(
        llvm::MachineFunctionProperties::Property::NoVRegs);
    llvm::MachineInstrBuilder Builder;
    auto MBB = MI->getParent();
    auto PCReg = MBB->getParent()->getRegInfo().createVirtualRegister(
        &llvm::AMDGPU::SReg_64RegClass);
    auto InstPoint = IPoint == INSTR_POINT_BEFORE ? MI : MI->getNextNode();
    Builder = llvm::BuildMI(*MBB, InstPoint, llvm::DebugLoc(),
                            MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKUP))
                  .addImm(0)
                  .addImm(0);
    auto &CalleeMF = IFLSI->getMFofSymbol();
    auto &CalleeFunction = CalleeMF.getFunction();
    //    MBB->getParent()->dump();
    //    llvm::outs() << " ======== \n";
    Builder =
        llvm::BuildMI(*MBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::SI_PC_ADD_REL_OFFSET))
            .addDef(PCReg)
            .addGlobalAddress(&CalleeFunction, 0,
                              llvm::SIInstrInfo::MO_REL32_LO)
            .addGlobalAddress(&CalleeFunction, 0,
                              llvm::SIInstrInfo::MO_REL32_HI);
    //    MBB->getParent()->dump();
    //    llvm::outs() << " ======== \n";
    auto SaveReg = MBB->getParent()->getRegInfo().createVirtualRegister(
        &llvm::AMDGPU::SGPR_128RegClass);
    Builder =
        llvm::BuildMI(*MBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::COPY))
            .addDef(SaveReg)
            .addReg(llvm::AMDGPU::PRIVATE_RSRC_REG);
    //    MBB->getParent()->dump();
    //    llvm::outs() << " ======== \n";
    Builder =
        llvm::BuildMI(*MBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::COPY))
            .addDef(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3)
            .addReg(SaveReg);
    //    MBB->getParent()->dump();
    //    llvm::outs() << " ======== \n";
    auto TRI = CalleeMF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
    auto RegMask =
        TRI->getCallPreservedMask(CalleeMF, CalleeFunction.getCallingConv());

    Builder =
        llvm::BuildMI(*MBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::SI_CALL))
            .addReg(llvm::AMDGPU::SGPR30_SGPR31, llvm::RegState::Define)
            .addReg(PCReg, llvm::RegState::Kill)
            .addGlobalAddress(&CalleeFunction)
            .add(llvm::MachineOperand::CreateReg(
                llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, false, true))
            .addRegMask(RegMask);

    //    MBB->getParent()->dump();
    //    llvm::outs() << " ======== \n";
    Builder =
        llvm::BuildMI(*MBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKDOWN))
            .addImm(0)
            .addImm(0);
    //    MBB->getParent()->dump();
    //    llvm::outs() << " ======== \n";
    //                                    .addDef(llvm::AMDGPU::SGPR4_SGPR5,
    //                                    llvm::RegState::Define);
    //      MIB.addGlobalAddress(InstLLVMFunction, 0,
    //      llvm::SIInstrInfo::MO_REL32); MIB.addGlobalAddress(InstLLVMFunction,
    //      0, llvm::SIInstrInfo::MO_REL32 + 1);
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
    Out.push_back(*IFLSI);
  }
  return Out;
}

} // namespace luthier
