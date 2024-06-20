#include "code_generator.hpp"

#include <memory>

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/CSEConfigBase.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Value.h"
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
// #include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/Cloning.h>

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
#include <SIRegisterInfo.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/Passes/PassBuilder.h>
#include <optional>
// #include <strstream>

#include "AMDGPUGenInstrInfo.inc"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Instructions.h"

namespace luthier {

template <> CodeGenerator *Singleton<CodeGenerator>::Instance{nullptr};

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
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_destroy_data_set(DataSetIn)));
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_COMGR_SUCCESS_CHECK(amd_comgr_destroy_data_set(DataSetOut)));
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
  llvm:outs() << "\napplyInstrumentation DONE!\n";

  LUTHIER_RETURN_ON_ERROR(AddedLSIs.takeError());

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
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(!TM.addAsmPrinter(
      PM, OutOS, nullptr, llvm::CodeGenFileType::ObjectFile, MCContext)));

  PM.run(*Module); // Run all the passes

  // llvm::outs() << "\n=====> Dump compilation module right before compiling to reloc\n";
  // Module->dump();

  LUTHIER_RETURN_ON_ERROR(compileRelocatableToExecutable(
      llvm::ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(Reloc.data()),
                              Reloc.size()),
      *Isa, Executable));

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

  llvm::outs() << "\n=====> Begin CodeGenerator::insertFunctionCalls\n";

  std::vector<LiftedSymbolInfo> Out;
  for (const auto &[IPointMI, IFuncAndIPoint] : Tasks) {
    llvm::MachineBasicBlock *IPointMBB = IPointMI->getParent();
    auto &[IFuncShadowHostPtr, IPoint] = IFuncAndIPoint;
    // Get the hsa::GpuAgent of the Target Kernel
    auto TargetKernelSymbol =
        hsa::ExecutableSymbol::fromHandle(TargetLSI.getSymbol());
    LUTHIER_RETURN_ON_ERROR(TargetKernelSymbol.takeError());
    auto Agent = TargetKernelSymbol->getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());
    // Figure out the hsa::ExecutableSymbol of the instrumentation function
    // loaded on the target kernel's hsa::GpuAgent
    llvm::outs() << "=====> Get executable symbol of instrumentation function\n";
    auto IFunctionSymbol =
        luthier::CodeObjectManager::instance().getInstrumentationFunction(
            IFuncShadowHostPtr, *Agent);
    LUTHIER_RETURN_ON_ERROR(IFunctionSymbol.takeError());
    llvm::outs() << "=====\n\n";
    llvm::outs() << "=====> run getModuleContainingInstrumentationFunctions\n";
    auto InstrumentationModule =
        CodeObjectManager::instance()
            .getModuleContainingInstrumentationFunctions({*IFunctionSymbol});
    LUTHIER_RETURN_ON_ERROR(InstrumentationModule.takeError());

    // Create the analysis managers.
    // These must be declared in this order so that they are destroyed in the
    // correct order due to inter-analysis-manager references.
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    llvm::PassBuilder PB;

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Create the pass manager.
    ModulePassManager MPM =
        PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);

    llvm::outs() << "\n=====> Create Master Kernel\n";
    llvm::Type *const MKRetType = 
        llvm::Type::getVoidTy(InstrumentationModule->get()->getContext());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MKRetType != nullptr));
    llvm::FunctionType *MKFuncType =
        llvm::FunctionType::get(MKRetType, {}, false);
    // TODO: Try adding AttributeList when calling getOrInsertFunction
    llvm::FunctionCallee MKFuncCallee = 
        InstrumentationModule->get()
                             ->getOrInsertFunction("MasterKernel", MKFuncType);

    auto MK = llvm::dyn_cast<llvm::Function>(MKFuncCallee.getCallee());

    auto &InstrumentationContext = InstrumentationModule.get()->getContext();
    llvm::IRBuilder<> MKBuilder(InstrumentationContext);
    
    llvm::outs() << "     > Set IFuncs to always inline\n";
    for (llvm::Function &F : **InstrumentationModule) {
      F.removeFnAttr(llvm::Attribute::OptimizeNone);
      if (F.getName() == "instrumentationHook") {
        F.removeFnAttr(llvm::Attribute::NoInline);
        F.addFnAttr(llvm::Attribute::AlwaysInline);

//         for (llvm::BasicBlock &BB : F) {
//           for (llvm::Instruction &I : BB) {
//             // Edit the add instruction to verify that we're running the 
//             // optimized function from the instrumentaiton module
//             // In kernel_instrument.cpp the instrumentationHook should have a
//             // hard-coded 'GlobalCounter += 10000'
//             if (I.getOpcode() == 13) {
//               for (auto &Op : I.operands()) {
//                 if (auto *IMM = llvm::dyn_cast<llvm::ConstantInt>(&Op)) {
//                   llvm::Constant *NewRHS = llvm::ConstantInt::get(
//                     llvm::Type::getInt32Ty(InstrumentationContext), 20000);
//                   I.setOperand(Op.getOperandNo(), NewRHS);
//                 }
//               }
//             }
// //          if (auto *AllocaInst = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
// //            new llvm::AllocaInst(
// //                llvm::PointerType::get(AllocaInst->getAllocatedType(),
// //                                       AMDGPUAS::GLOBAL_ADDRESS),
// //                InstrumentationModule.get().get()->getDataLayout().getAllocaAddrSpace(), AllocaInst->getArraySize(),
// //                AllocaInst->getAlign(), AllocaInst->getName(),
// //                AllocaInst->getIterator());
// //            AllocaInst->eraseFromParent();
// //          }
//           }
//         }
      } 
    }

    auto TargetKernelMD = TargetKernelSymbol->getKernelMetadata();
    LUTHIER_RETURN_ON_ERROR(TargetKernelMD.takeError());
    // auto TargetKernelArgSeg = TargetKernelMD->KernArgSegmentSize;
    // auto TargetKernelGroupSeg = TargetKernelMD->GroupSegmentFixedSize;
    auto TargetKernelPrivSegFixedSize = TargetKernelMD
                                          ->PrivateSegmentFixedSize;
    llvm::Value *TargetKernelPrivSeg = 
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(InstrumentationContext), 
                               TargetKernelPrivSegFixedSize, false);
    // Builder = llvm::BuildMI(*IPointMBB, InstPoint, llvm::DebugLoc(),
                            // MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKUP))
    //               .addImm(0)
    //               .addImm(0);

    // TODO: Need to add blocks to denote the lifetime start/end of stack in
    //       master kernel
    llvm::outs() << "     > Populate MK w/ basic blocks:\n";
    llvm::BasicBlock *MKBB_BEGIN = 
        llvm::BasicBlock::Create(InstrumentationContext, 
                                 "RESERVED_stack_lifetime_begin", MK); //, MKBB);
    llvm::BasicBlock *MKBB = 
        llvm::BasicBlock::Create(InstrumentationContext, 
                                 "InstruPoint", MK); //, MKBB_END);
    llvm::BasicBlock *MKBB_END = 
        llvm::BasicBlock::Create(InstrumentationContext, 
                                 "RESERVED_stack_lifetime_end", MK);
    
    llvm::outs() << "     > \tCreate call save stack lifetime begin in MK\n";
    MKBuilder.SetInsertPoint(MKBB_BEGIN);
    // auto SaveStackInstr = MKBuilder.CreateStackSave();
    auto StartStackAlloca = 
        MKBuilder.CreateAlloca(MKBuilder.getInt32Ty(), 5, TargetKernelPrivSeg);
    MKBuilder.CreateLifetimeStart(StartStackAlloca);
    MKBuilder.CreateBr(MKBB);

    llvm::outs() << "     > \tCreate call to instrumentation func in MK\n";
    MKBuilder.SetInsertPoint(MKBB);
    auto Hook = InstrumentationModule.get()->getFunction("instrumentationHook");
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Hook != nullptr));
    MKBuilder.CreateCall(Hook);
    MKBuilder.CreateBr(MKBB_END);
    
    llvm::outs() << "     > \tCreate call save stack lifetime end in MK\n";
    MKBuilder.SetInsertPoint(MKBB_END);
    MKBuilder.CreateLifetimeEnd(StartStackAlloca); // , llvm::ConstantInt::get(
    //   MKBuilder.getInt64Ty(), TargetKernelPrivSeg));
    // auto RestoreStackInstr = MKBuilder.CreateStackRestore(SaveStackInstr);
    // Creating a void return at the end of the basic block stops LLVM from
    // complaining. We should experiment with trying to get rid of this.
    // However, if we cannot, then all we have to do is to exclude the final
    // instruction of IPoint basic block when patching the compilation module
    MKBuilder.CreateRetVoid();

    llvm::outs() << "     > Dump instrumentation module after adding MK\n";
    InstrumentationModule->get()->dump();

    // Optimize the IR!
    llvm::outs() << "\n=====> Run IR optimization for InstrumentationModule\n";
    MPM.run(**InstrumentationModule, MAM);
    
    
    llvm::outs() << "     > Dump instrumentation module after IR optimization\n";
    InstrumentationModule->get()->dump();

    
    auto ISA = llvm::cantFail(IFunctionSymbol->getLoadedCodeObject()).getISA();
    LUTHIER_RETURN_ON_ERROR(ISA.takeError());
    auto TargetInfo = TargetManager::instance().getTargetInfo(*ISA);
    LUTHIER_RETURN_ON_ERROR(TargetInfo.takeError());
    auto &TM = *TargetInfo->getTargetMachine();
    llvm::legacy::PassManager PM;
    TM.setOptLevel(llvm::CodeGenOptLevel::Aggressive);

    auto MMIWP = new llvm::MachineModuleInfoWrapperPass(&TM);

    llvm::TargetLibraryInfoImpl TLII(
        llvm::Triple((*InstrumentationModule)->getTargetTriple()));
    PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

    llvm::TargetPassConfig *TPC = TM.createPassConfig(PM);

    PM.add(TPC);
    PM.add(MMIWP);

    TPC->addISelPasses();

    // Get Liveness for IPoint MBB and copy that info into the Instrumentation
    // Hook's basic block in the Master Kernel
    PM.add(new llvm::LivenessCopy(IPointMBB));

    TPC->addMachinePasses();
    auto UsageAnalysis = new llvm::AMDGPUResourceUsageAnalysis();
    PM.add(UsageAnalysis);
    TPC->setInitialized();

    llvm::outs() << "=====> Run codegen \n";
    PM.run(**InstrumentationModule); // Run all the passes

    llvm::outs() << "\n=====> Dump instrumentation module's machine functions\n";
    for (const auto &F : **InstrumentationModule) {
      const auto MF = MMIWP->getMMI().getMachineFunction(F);
      if (MF != nullptr) {
        MF->dump();
      } 
    } 

    // // Lift the instrumentation function's symbol and add it to the compilation
    // // module
    auto IFLSI = luthier::CodeLifter::instance().liftSymbolAndAddToModule(
        *IFunctionSymbol, Module, MMI);
    LUTHIER_RETURN_ON_ERROR(IFLSI.takeError());
    
    auto &CalleeMF = IFLSI->getMFofSymbol();
    auto &CalleeFunction = CalleeMF.getFunction();
    // auto CalleeMF = MMIWP->getMMI().getMachineFunction(*MK);
    // auto CalleeFunction = MK;
   
    auto MCInstInfo = MMI.getTarget().getMCInstrInfo();
    auto ToBeInstrumentedMF = IPointMI->getParent()->getParent();
    auto TRI = ToBeInstrumentedMF->getSubtarget<llvm::GCNSubtarget>()
                   .getRegisterInfo();
    // Start of instrumentation logic
    // Spill the registers to the stack before calling the instrumentation
    // function
    llvm::outs() << "\n=====> Start of Instrumentation Logic:\n"
                 << "       Split machine basic block of IPointMI \n";

    llvm::MachineBasicBlock *NewMBB = 
        ToBeInstrumentedMF->CreateMachineBasicBlock(IPointMBB->getBasicBlock());
    // llvm::MachineFunction::iterator MBBIT(IPointMBB);
    // ToBeInstrumentedMF->insert(MBBIT, NewMBB);
    ToBeInstrumentedMF->push_back(NewMBB);
    
    IPointMI->getParent()->getParent()->getProperties().reset(
        llvm::MachineFunctionProperties::Property::NoVRegs);
    llvm::MachineInstrBuilder Builder;
    // auto IPointMBB = IPointMI->getParent();
    
    auto PCReg = IPointMBB->getParent()->getRegInfo().createVirtualRegister(
        &llvm::AMDGPU::SReg_64RegClass);
   
    auto InstPoint =
        IPoint == INSTR_POINT_BEFORE ? IPointMI : IPointMI->getNextNode();

    NewMBB->splice(NewMBB->end(), IPointMBB, 
                   llvm::MachineBasicBlock::iterator(InstPoint), IPointMBB->end());
    // NewMBB->transferSuccessors(IPointMBB);
    // IPointMBB->addSuccessor(NewMBB);

    ToBeInstrumentedMF->dump();


    // Builder = llvm::BuildMI(*IPointMBB, InstPoint, llvm::DebugLoc(),
    //                         MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKUP))
    //               .addImm(0)
    //               .addImm(0);
/*
    Builder =
        llvm::BuildMI(*IPointMBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::SI_PC_ADD_REL_OFFSET))
            .addDef(PCReg)
            .addGlobalAddress(&CalleeFunction, 0,
                              llvm::SIInstrInfo::MO_REL32_LO)
            .addGlobalAddress(&CalleeFunction, 0,
                              llvm::SIInstrInfo::MO_REL32_HI);

    auto SaveReg = IPointMBB->getParent()->getRegInfo().createVirtualRegister(
        &llvm::AMDGPU::SGPR_128RegClass);
    Builder =
        llvm::BuildMI(*IPointMBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::COPY))
            .addDef(SaveReg)
            .addReg(llvm::AMDGPU::PRIVATE_RSRC_REG);

    Builder =
        llvm::BuildMI(*IPointMBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::COPY))
            .addDef(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3)
            .addReg(SaveReg);

    auto RegMask =
        TRI->getCallPreservedMask(CalleeMF, CalleeFunction.getCallingConv());

    Builder =
        llvm::BuildMI(*IPointMBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::SI_CALL))
            .addReg(llvm::AMDGPU::SGPR30_SGPR31, llvm::RegState::Define)
            .addReg(PCReg, llvm::RegState::Kill)
            .addGlobalAddress(&CalleeFunction)
            .add(llvm::MachineOperand::CreateReg(
                llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, false, true))
            .addRegMask(RegMask);
*/

    
    llvm::outs() << "       Insert call to IFunc into compilation module\n";
    const auto MKMF = MMIWP->getMMI().getMachineFunction(*MK);
    for (auto &MKBB : *MKMF) {
      if (MKBB.getName() == "InstruPoint") {
        llvm::MachineBasicBlock *IFuncMBB = 
            ToBeInstrumentedMF->CreateMachineBasicBlock(MKBB.getBasicBlock());
        llvm::MachineFunction::iterator MBBIT(NewMBB);
        ToBeInstrumentedMF->insert(MBBIT, IFuncMBB);

        for (auto &MKMI : MKBB) {
          // llvm::MachineInstr *NewMI = MKMF->CloneMachineInstr(&MKMI);
          llvm::MachineInstr *NewMI = ToBeInstrumentedMF->CloneMachineInstr(&MKMI);
          NewMI->dump();
          // IFuncMBB->insert(IFuncMBB->end(), NewMI);
        }
        IPointMBB->addSuccessor(IFuncMBB);
        IFuncMBB->addSuccessor(NewMBB);
        
        // IFuncMBB->splice(llvm::MachineBasicBlock::iterator(InstPoint), &MKBB,
        //             MKBB.begin(), MKBB.end());
        // IFuncMBB->splice(IFuncMBB->begin(), &MKBB, MKBB.begin(), MKBB.end());
      }
    }


    // llvm::outs() << "\n       Insert Adjust callstack up \n"; 
    // Builder =
    //     llvm::BuildMI(*IPointMBB, Builder.getInstr()->getNextNode(), llvm::DebugLoc(),
    //                   MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKDOWN))
    //         .addImm(0)
    //         .addImm(0);

    //    IPointMBB->getParent()->dump();
    //                                    .addDef(llvm::AMDGPU::SGPR4_SGPR5,
    //                                    llvm::RegState::Define);
    //      MIB.addGlobalAddress(InstLLVMFunction, 0,
    //      llvm::SIInstrInfo::MO_REL32); MIB.addGlobalAddress(InstLLVMFunction,
    //      0, llvm::SIInstrInfo::MO_REL32 + 1);
    //
    //      Builder = BuildMI(IPointMBB, I, llvm::DebugLoc(),
    //                        MCInstInfo->get(llvm::AMDGPU::SI_CALL_ISEL))
    //                    .addReg(llvm::AMDGPU::SGPR4_SGPR5,
    //                            llvm::RegState::Kill).addGlobalAddress(InstLLVMFunction);
    ////      Builder; // =
    //      BuildMI(IPointMBB, I, llvm::DebugLoc(),
    //              MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKDOWN))
    //          .addImm(0)
    //          .addImm(0);
    llvm::outs() << "=====> End of Instrumentation Logic\n"
                 << "       Dump new App Machine Function\n";
                 // << "       Dump new IPoint Basic Block\n";
    // IPointMBB->dump();
    ToBeInstrumentedMF->dump();

    Out.push_back(*IFLSI);
  }
  return Out;
}

} // namespace luthier
