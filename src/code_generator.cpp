#include "code_generator.hpp"

#include <memory>

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/CSEConfigBase.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SaveAndRestore.h"
#include <AMDGPUResourceUsageAnalysis.h>
#include <AMDGPUTargetMachine.h>
#include <amd_comgr/amd_comgr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/ScopedNoAliasAA.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/CodeGen/LiveIntervals.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>

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
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
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

#include "AMDGPUGenInstrInfo.inc"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Instructions.h"

namespace luthier {

template<> CodeGenerator* Singleton<CodeGenerator>::Instance{nullptr};

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

llvm::Error CodeGenerator::instrument(
    std::unique_ptr<llvm::Module> Module,
    std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP,
    const LiftedSymbolInfo &LSO, luthier::InstrumentationTask &ITask) {
  LUTHIER_LOG_FUNCTION_CALL_START
  auto Symbol = hsa::ExecutableSymbol::fromHandle(LSO.getSymbol());
  LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
  auto LCO = Symbol->getLoadedCodeObject();
  LUTHIER_RETURN_ON_ERROR(LCO.takeError());
  auto Isa = LCO->getISA();
  LUTHIER_RETURN_ON_ERROR(Isa.takeError());
  auto &CodeObjectManager = CodeObjectManager::instance();
  auto &ContextManager = TargetManager::instance();
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

  llvm::outs() << "\nAdd required Instruction Selection Passes\n";
  TPC->addCodeGenPrepare();
  PM.add(getPass(TPC, &llvm::FinalizeISelID));
  llvm::outs() << "\nFinished adding Instruction Selection Passes\n";

  llvm::outs() << "\nAdd Machine Passes\n";
  // Add reg alloc passes -- Don't need all of these, however they might be 
  //                         good optimizations to run
  // TPC->insertPass(&llvm::PHIEliminationID, &llvm::SILowerControlFlowID);
  // TPC->insertPass(&llvm::TwoAddressInstructionPassID, &llvm::SIWholeQuadModeID);
  // PM.add(getPass(TPC, &llvm::PHIEliminationID));
  // PM.add(getPass(TPC, &llvm::TwoAddressInstructionPassID));
  PM.add(llvm::createFastRegisterAllocator());
  llvm::outs() << "\n ~ Added Reg Alloc passes\n";

  PM.add(llvm::createPrologEpilogInserterPass());
  llvm::outs() << "\n ~ Added prologue/epilogue insertion pass \n";

  // Expand pseudo instructions before second scheduling pass.
  // Apparently we need this. Idk what the "second scheduling pass" is bc 
  // we aren't doing a second ISel pass. But it breaks when I comment this out
  PM.add(getPass(TPC, &llvm::ExpandPostRAPseudosID));
  llvm::outs() << "\n ~ Added ExpandPostRAPseudosID\n";

  // FROM LLVM: Collect register usage information and produce a register mask of
  //            clobbered registers, to be used to optimize call sites.
  // Able to run w/o this pass
  // PM.add(llvm::createRegUsageInfoCollector()); 
  // llvm::outs() << "\n ~ Added createRegUsageInfoCollector pass\n";

  // Keep this in case we want to dump the stack frame
  // PM.add(llvm::createStackFrameLayoutAnalysisPass());
  // llvm::outs() << "\n ~ Added StackFrameLayoutAnalysis pass\n";
  llvm::outs() << "\nFinished adding Machine Passes\n\n";


  auto UsageAnalysis = new llvm::AMDGPUResourceUsageAnalysis();
  PM.add(UsageAnalysis);
  TPC->setInitialized();
  llvm::raw_svector_ostream OutOS(Reloc);

  // AsmPrinter is responsible for generating the assembly into AsmBuffer.
  if (TM.addAsmPrinter(PM, OutOS, nullptr, llvm::CodeGenFileType::ObjectFile,
                       MCContext))
    llvm::outs() << "Failed to add pass manager\n";

  PM.run(*Module); // Run all the passes

  LUTHIER_RETURN_ON_ERROR(compileRelocatableToExecutable(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(Reloc.data()),
                              Reloc.size()),
      *Isa, Executable));
  //  llvm::outs() << "Compiled to executable\n";

  //  llvm::outs() << llvm::toStringRef(Executable) << "\n";

  std::vector<hsa::ExecutableSymbol> ExternGVs;
  for (const auto &GV : LSO.getRelatedVariables()) {
    auto GVWrapper = hsa::ExecutableSymbol::fromHandle(GV);
    LUTHIER_RETURN_ON_ERROR(GVWrapper.takeError());
    ExternGVs.push_back(*GVWrapper);
  }

  for (const auto &L : *AddedLSIs)
    for (const auto &GV : L.getRelatedVariables()) {
      auto GVWrapper = hsa::ExecutableSymbol::fromHandle(GV);
      LUTHIER_RETURN_ON_ERROR(GVWrapper.takeError());
      ExternGVs.push_back(*GVWrapper);
    }

  LUTHIER_RETURN_ON_ERROR(
      CodeObjectManager.loadInstrumentedKernel(Executable, *Symbol, ExternGVs));

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
    const LiftedSymbolInfo &TargetLSI,
    const InstrumentationTask::insert_call_tasks &Tasks) {
  std::vector<LiftedSymbolInfo> Out;
  for (const auto &[IPointMI, IFuncAndIPoint] : Tasks) {
    auto &[IFuncShadowHostPtr, IPoint] = IFuncAndIPoint;
    // Get the hsa::GpuAgent of the Target Kernel
    auto TargetKernelSymbol =
        hsa::ExecutableSymbol::fromHandle(TargetLSI.getSymbol());
    LUTHIER_RETURN_ON_ERROR(TargetKernelSymbol.takeError());
    auto Agent = TargetKernelSymbol->getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());
    // Figure out the hsa::ExecutableSymbol of the instrumentation function
    // loaded on the target kernel's hsa::GpuAgent
    auto IFunctionSymbol =
        luthier::CodeObjectManager::instance().getInstrumentationFunction(
            IFuncShadowHostPtr, *Agent);
    LUTHIER_RETURN_ON_ERROR(IFunctionSymbol.takeError());
    // Lift the instrumentation function's symbol and add it to the compilation
    // module
    auto IFLSI = luthier::CodeLifter::instance().liftSymbolAndAddToModule(
        *IFunctionSymbol, Module, MMI);
    LUTHIER_RETURN_ON_ERROR(IFLSI.takeError());

    auto MCInstInfo = MMI.getTarget().getMCInstrInfo();
    auto ToBeInstrumentedMF = IPointMI->getParent()->getParent();
    auto TRI = ToBeInstrumentedMF->getSubtarget<llvm::GCNSubtarget>()
                   .getRegisterInfo();
    // Start of instrumentation logic
    // Spill the registers to the stack before calling the instrumentation
    // function

    IPointMI->getParent()->getParent()->getProperties().reset(
        llvm::MachineFunctionProperties::Property::NoVRegs);
    llvm::MachineInstrBuilder Builder;
    auto MBB = IPointMI->getParent();
    auto PCReg = MBB->getParent()->getRegInfo().createVirtualRegister(
        &llvm::AMDGPU::SReg_64RegClass);
    auto InstPoint =
        IPoint == INSTR_POINT_BEFORE ? IPointMI : IPointMI->getNextNode();
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
llvm::Error CodeGenerator::convertToVirtual(Module &Module,
                                            MachineModuleInfo &MMI,
                                            const LiftedSymbolInfo &LSI) {
  auto &MF = LSI.getMFofSymbol();
  auto &MFI = *MF.getInfo<llvm::SIMachineFunctionInfo>();
  llvm::DenseMap<llvm::Register, llvm::Register> PhysToVirtReg;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      MI.print(llvm::outs(), true, false, true, true,
               MF.getSubtarget().getInstrInfo());
      llvm::outs() << "\n";

      const llvm::MCInstrDesc &MCID =
          MF.getTarget().getMCInstrInfo()->get(MI.getOpcode());
      for (int I = 0; I < MI.getNumOperands(); I++) {
        auto &Op = MI.getOperand(I);
        if (Op.isReg() && !Op.isImplicit()) {
          auto PhysReg = Op.getReg();
          if (!PhysToVirtReg.contains(PhysReg)) {
            //            MF.getFrameInfo().CreateFixedObject()
            auto *TRI = MF.getSubtarget().getRegisterInfo();
            //            TRI->getSubRegIndex();
            auto OpRegClassFromMCID = MCID.operands()[I].RegClass;
            auto RegClass =
                MF.getSubtarget().getRegisterInfo()->getMinimalPhysRegClass(
                    PhysReg.asMCReg());
            llvm::outs()
                << "Reg class selected for "
                << printReg(PhysReg, MF.getSubtarget().getRegisterInfo())
                << "is "
                << MF.getSubtarget().getRegisterInfo()->getRegClassName(
                       RegClass)
                << " \n";
            llvm::outs() << "What MCInstDesc tells me is: "
                         << TRI->getRegClassName(
                                TRI->getRegClass(OpRegClassFromMCID))
                         << "\n";
            //          auto RegClass = MF.getRegInfo().getRegClass(PhysReg);
            PhysToVirtReg.insert(
                {Op.getReg(), MF.getRegInfo().createVirtualRegister(RegClass)});
          }
          //          auto& VirtReg = PhysToVirtReg.at(PhysReg);
          //          Op.setReg(VirtReg);
        }
      }
      MI.print(llvm::outs(), true, false, true, true,
               MF.getSubtarget().getInstrInfo());
      llvm::outs() << "\n";
    };
  }
  //  MF.addLiveIn()
  for (const auto &[PhysReg, VirtReg] : PhysToVirtReg) {
    llvm::outs() << "Regs that I found: \n";
    llvm::outs() << printReg(PhysReg, MF.getSubtarget().getRegisterInfo())
                 << "\n";
    llvm::outs() << printReg(VirtReg, MF.getSubtarget().getRegisterInfo())
                 << "\n";
  }
  return llvm::Error::success();
}

} // namespace luthier
