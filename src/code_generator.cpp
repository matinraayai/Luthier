#include "code_generator.hpp"

#include <memory>

#include "llvm/ADT/Any.h"
#include "llvm/ADT/StringRef.h"
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
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
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
#include <llvm/CodeGen/LivePhysRegs.h>
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
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include "llvm/CodeGen/TargetInstrInfo.h"
#include <llvm/CodeGen/TargetSubtargetInfo.h>
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
#include <vector>
// #include <strstream>

#include "AMDGPUGenInstrInfo.inc"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/StandardInstrumentations.h"

namespace llvm {

namespace {

// This pass is defined by Luthier. Its job is to iterate over the LiveIns
// of the IPointMI's parent Machine Basic Block and set them as reserved
// so that the register allocator does not overwrite them in the generated
// instrumentation kernel
struct LuthierReserveLiveRegs : public MachineFunctionPass {
  static char ID;
  llvm::LivePhysRegs LiveRegs;
  // llvm::StringRef InstruFuncName;

  LuthierReserveLiveRegs() : MachineFunctionPass(ID) {}

  explicit LuthierReserveLiveRegs(const llvm::MachineBasicBlock* IPointBlock) :
                           // const llvm::StringRef FunctionName) : 
      MachineFunctionPass(ID) {
    llvm::computeLiveIns(LiveRegs, *IPointBlock);
    // InstruFuncName = FunctionName;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    // My idea here was to also have it so that this pass will only run on the 
    // instrumentation kernel by saving the name of the target MF.
    // Currently, this function will run on all Machine Functions in the
    // instrumentation module, which is not a big deal at the moment.
    // if (InstruFuncName.empty() || InstruFuncName != MF.getName())
    //   return true;

    llvm::outs() << "=====> Run LuthierReserveLiveRegs on function: " 
                 << MF.getName() << "\n";
    
    auto &MRI = MF.getRegInfo();
    auto *TRI = MF.getSubtarget().getRegisterInfo();
    
    // MRI.freezeReservedRegs(MF);
    
    // SGPR0-3 are ALWAYS reserved
    MRI.reserveReg(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, TRI);
    // llvm::outs() << "     > Are SGPR0-3 reserved after calling reserveReg()?\t";
    // MRI.isReserved(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3) ? 
    //     llvm::outs() << "YES is reserved\n" : 
    //     llvm::outs() << "NO is not reserved\n";

    llvm::outs() << "     > Set the liveins of the original App func as reserved regs\n";
    for (auto &Reg : LiveRegs)
      MRI.reserveReg(Reg, TRI);

    // MRI.reservedRegsFrozen() ? 
    //     llvm::outs() << "\n Reserved REGs already frozen\n" : 
    //     llvm::outs() << "\n Reserved regs not frozxen yet\n";
    // MRI.canReserveReg(llvm::AMDGPU::SGPR0_SGPR1) ? 
    //     llvm::outs() << "\n YES can reserve\n" : 
    //     llvm::outs() << "\n NO cannot reserve\n";
    // MRI.reserveReg(llvm::AMDGPU::SGPR0_SGPR1, TRI);
    // MRI.freezeReservedRegs();
    
    // Need to freeze regs after reserving them
    MRI.freezeReservedRegs();
    
    // llvm::outs() << "     > Are SGPR0-3 STILL reserved after freezeReservedRegs()?\t";
    // MRI.isReserved(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3) ? 
    //     llvm::outs() << "YES is still reserved\n" : 
    //     llvm::outs() << "NO longer reserved\n";

    // MF.addLiveIn(llvm::AMDGPU::SGPR0_SGPR1, &llvm::AMDGPU::SReg_64RegClass);
    // LiveRegs.addReg(llvm::AMDGPU::SGPR0);
    // LiveRegs.addReg(llvm::AMDGPU::SGPR1);
    // LiveRegs.addReg(llvm::AMDGPU::SGPR0_SGPR1);
    // LiveRegs.addReg(llvm::AMDGPU::SGPR2);
    // LiveRegs.addReg(llvm::AMDGPU::SGPR3);
    
    // MF.addLiveIn(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, 
    //             &llvm::AMDGPU::SReg_64RegClass);
    // LiveRegs.addReg(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3);
    // for (auto &MBB : MF) {
    //   llvm::outs() << "     > Add LiveIns to Block: " << MBB.getName() << "\n";
    //   llvm::addLiveIns(MBB, LiveRegs);
    // }
    
    llvm::outs() << "     > End of LuthierReserveLiveRegs for " << MF.getName() << "\n\n";
    return true;
  }
};

// This custom pass iterates through the Instrumentation Modules frame objects
// and adds the amount of stack allocated by the IPointMI's parent Machine 
// Function to the frame object offset
struct LuthierStackFrameOffset : public MachineFunctionPass {
  static char ID;
  unsigned int StackSize = 0; 

  LuthierStackFrameOffset() : MachineFunctionPass(ID) {}
  
  explicit LuthierStackFrameOffset(unsigned int PrivSegSize) : 
      MachineFunctionPass(ID) {
    StackSize = PrivSegSize;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MFI = MF.getFrameInfo();
    
    llvm::outs() << "Running LuthierStackFrameOffset for machine function " 
                 << MF.getName() << "\n"
                 << "\t - Stack Size: " << MFI.getStackSize() << "\n"
                 << "\t - Contains " << MFI.getNumObjects() << " Stack Objects\n";
    
    for (int SOIdx = 0; SOIdx < MFI.getNumObjects(); ++SOIdx) {
      llvm::outs() << "\t\t - Stack Object Num "      << SOIdx << "\n"
                   << "\t\t   Stack ID:             " << MFI.getStackID(SOIdx)    << "\n"
                   << "\t\t   Stack object Size:    " << MFI.getObjectSize(SOIdx) << "\n";
      // Add to Stack Frame object offset
      auto NewOffset = MFI.getObjectOffset(SOIdx) + StackSize;
      llvm::outs() << "   Stack Pointer Offset: " << NewOffset << "\n";
      MFI.setObjectOffset(SOIdx, NewOffset);
    }
    return false;
  }
};

} // namespace anonymous 

char LuthierReserveLiveRegs::ID = 0;
char LuthierStackFrameOffset::ID = 0;

} // namespace llvm


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
  
  llvm:outs() << "\napplyInstrumentation DONE!\n"
              << "Dump Compilation module:\n";
  Module->dump();

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
    auto &[IFuncShadowHostPtr, IArgs, IPoint] = IFuncAndIPoint;
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
    llvm::outs() << "=====> Get Intrumentation Module from executable symbol\n";
    auto InstrumentationModule =
        CodeObjectManager::instance()
            .getModuleContainingInstrumentationFunctions({*IFunctionSymbol});
    LUTHIER_RETURN_ON_ERROR(InstrumentationModule.takeError());
    InstrumentationModule.get()->dump();

    llvm::outs() << "=====> Create Kernel for IPoint Hook and add to instrumentation module\n";
    llvm::Type *const IKRetTy = 
        llvm::Type::getVoidTy(InstrumentationModule.get()->getContext());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(IKRetTy != nullptr));
    
    // TODO: Try adding an AttributeList when calling getOrInsertFunction i.e. Optlevel
    auto IKTy = llvm::FunctionType::get(IKRetTy, {}, false);
    auto IKCallee = InstrumentationModule.get()
                      ->getOrInsertFunction("Instrumentation_Kernel", IKTy);
    auto IK = llvm::dyn_cast<llvm::Function>(IKCallee.getCallee());
    IK->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    
    auto &InstrumentationContext = InstrumentationModule.get()->getContext();
    llvm::IRBuilder<> IKBuilder(InstrumentationContext);
    
    for (llvm::Function &F : **InstrumentationModule) {
      if (F.getName() == "instrumentationHook") {
        llvm::outs() << "=====> Set instrumentation hook to always inline\n";
        F.removeFnAttr(llvm::Attribute::OptimizeNone);
        F.removeFnAttr(llvm::Attribute::NoInline);
        F.addFnAttr(llvm::Attribute::AlwaysInline);
      } 
    }

    llvm::BasicBlock *IKBB = llvm::BasicBlock::Create(InstrumentationContext, 
                                                      "call_to_hook", IK);
    IKBuilder.SetInsertPoint(IKBB);
    auto Hook = InstrumentationModule.get()->getFunction("instrumentationHook");
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Hook != nullptr));
    
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Hook->arg_size() == IArgs.size()));
    // TODO: Only print this message when we have an error:
    llvm::outs() << "     > Check number of Instrumentation Hook arguments versus number of received arguments from Luthier call\n"
                 << "     >>> Hook expects " << Hook->arg_size() << " Function args\n"
                 << "     >>> Luthier received " << IArgs.size() << " args for Hook Call\n";

    llvm::SmallVector<llvm::Value*> LoadInsts;
    for (unsigned int ArgNo = 0; ArgNo < Hook->arg_size(); ArgNo++) {
      auto HookArg = Hook->getArg(ArgNo);
      auto IArg = IArgs[ArgNo];
      
      llvm::Type *IArgTy = IArg->getInitializer()->getType();

      // I don't know if this is the correct way to use our Assertion
      auto Err = LUTHIER_ASSERTION(HookArg->getType() == IArgTy);
      if (Err) {
        llvm::outs() << "     >>> Hook Arg Type for Arg No " << ArgNo << ": ";
        HookArg->getType()->dump();
        llvm::outs() << "     >>> Type for received Arg No " << ArgNo << ": ";
        IArg->getInitializer()->getType()->dump();
        LUTHIER_RETURN_ON_ERROR(Err);
      }
      LoadInsts.push_back(IKBuilder.CreateLoad(IArgTy, IArg));
    }
    llvm::outs() << "     > Create call to instrumentation func in IK\n";
    auto CalltoHook = IKBuilder.CreateCall(Hook, LoadInsts);
    IKBuilder.CreateRetVoid();
    llvm::outs() << "     > Finised creating the Instrumentation Kernel\n";
    llvm::outs() << "     > Dump instrumentation module\n";
    InstrumentationModule->get()->dump();

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
    llvm::PassInstrumentationCallbacks PIC;
    llvm::PrintIRInstrumentation PII;
    PII.registerCallbacks(PIC);
    
    llvm::PassBuilder PB(const_cast<llvm::LLVMTargetMachine *>(&MMI.getTarget()), 
                         llvm::PipelineTuningOptions(), std::nullopt, &PIC);
    
    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Create the pass manager.
    ModulePassManager MPM =
        PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
    // Optimize the IR!
    MPM.run(**InstrumentationModule, MAM);
    
    llvm::outs() << "=====> Run IR optimization for InstrumentationModule\n"
                 << "     > Dump instrumentation module after IR optimization\n";
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

    llvm::MachineBasicBlock *IPointMBB = IPointMI->getParent();
    PM.add(new llvm::LuthierReserveLiveRegs(IPointMBB));
    
    auto TargetKernelMD = TargetKernelSymbol->getKernelMetadata();
    LUTHIER_RETURN_ON_ERROR(TargetKernelMD.takeError());
    auto TargetPivSegSize = TargetKernelMD->PrivateSegmentFixedSize;
    llvm::outs() << "=====> Target Kernel Private Seg Size:\t" 
                 << TargetPivSegSize << "\n";
    PM.add(new llvm::LuthierStackFrameOffset(TargetPivSegSize));

    TPC->addMachinePasses();
    
    auto UsageAnalysis = new llvm::AMDGPUResourceUsageAnalysis();
    PM.add(UsageAnalysis);
    TPC->setInitialized();

    llvm::outs() << "=====> Run codegen \n";
    PM.run(**InstrumentationModule); // Run all the passes
    
    llvm::outs() << "     > Insert wait counts at beginning and end of Instrumentation func \n";
    llvm::MachineInstrBuilder MIRBuilder;
    const auto IKMF = MMIWP->getMMI().getMachineFunction(*IK);
    auto IKInstInfo = MMIWP->getMMI().getTarget().getMCInstrInfo(); // "Instrumentation Kernel Instr Info"...

    for (auto &IKBB : *IKMF) {
      if (IKBB.getName() != "call_to_hook")
        continue;
      
      MIRBuilder = llvm::BuildMI(IKBB, IKBB.begin(), llvm::DebugLoc(),
                                 IKInstInfo->get(llvm::AMDGPU::S_WAITCNT))
                                .addImm(0);
      MIRBuilder = llvm::BuildMI(IKBB, IKBB.end(), llvm::DebugLoc(),
                                 IKInstInfo->get(llvm::AMDGPU::S_WAITCNT))
                                .addImm(0); // for some reason this second waitcnt got placed AFTER the IK's endpgm
      // I'm pretty sure this should actually be non-pseudo ver. in the enum
      // IKInstInfo->get(llvm::AMDGPU::S_WAITCNT_vi)
      // Also need to use a different value for this immediate
    }
    
    llvm::outs() << "     > Dump instrumentation function's MIR representation\n";
    IKMF->dump();

    // Lift the instrumentation function's symbol and add it to the compilation
    // module
    auto IFLSI = luthier::CodeLifter::instance().liftSymbolAndAddToModule(
        *IFunctionSymbol, Module, MMI);
    LUTHIER_RETURN_ON_ERROR(IFLSI.takeError());
   
    auto MCInstInfo = MMI.getTarget().getMCInstrInfo();
    auto ToBeInstrumentedMF = IPointMI->getParent()->getParent();
    // auto TRI = ToBeInstrumentedMF->getSubtarget<llvm::GCNSubtarget>()
    //                .getRegisterInfo();
    
    // Start of instrumentation logic
    // Spill the registers to the stack before calling the instrumentation
    // function

    // IPointMI->getParent()->getParent()->getProperties().reset(
    //     llvm::MachineFunctionProperties::Property::NoVRegs);
    // llvm::MachineInstrBuilder MIRBuilder;
    
    // auto PCReg = IPointMBB->getParent()->getRegInfo().createVirtualRegister(
    //     &llvm::AMDGPU::SReg_64RegClass);
   
    auto InstPoint =
        IPoint == INSTR_POINT_BEFORE ? IPointMI : IPointMI->getNextNode();
  
    llvm::outs() << "\n=====> Start of Instrumentation Logic:\n"
                 <<   "     > Split machine basic block of IPointMI \n";
    
    // TODO: Call this function instead of manually splitting
    // IPointMBB->splitAt(*InstPoint);

    llvm::MachineBasicBlock *IPointBlockSplit = 
        ToBeInstrumentedMF->CreateMachineBasicBlock(IPointMBB->getBasicBlock());
    ToBeInstrumentedMF->push_back(IPointBlockSplit);
    IPointBlockSplit->splice(IPointBlockSplit->end(), IPointMBB, 
                   llvm::MachineBasicBlock::iterator(InstPoint), IPointMBB->end());
    
    ToBeInstrumentedMF->dump();
    llvm::outs() << "     > Patch call to instrumentation function into compilation module\n";
    for (auto &IKBB : *IKMF) {
      if (IKBB.getName() != "call_to_hook")
        continue;
    
      llvm::MachineBasicBlock *IFuncMBB = 
          ToBeInstrumentedMF->CreateMachineBasicBlock(IKBB.getBasicBlock());
      ToBeInstrumentedMF->insert(llvm::MachineFunction::iterator(IPointBlockSplit), IFuncMBB);
      IPointMBB->addSuccessor(IFuncMBB);
      IFuncMBB->addSuccessor(IPointBlockSplit);
      
      // for (auto &IKMI : IKBB) {
      //   if (IKMI.getOpcode() == llvm::AMDGPU::S_SETPC_B64_return)
      //     continue;
      //   
      //   // create a new MI by copying the operands from the instrumentation 
      //   // module's instructions
      //   // This is the wrong overload for instructions that need a destination reg
      //   MIRBuilder = llvm::BuildMI(*IFuncMBB, IFuncMBB->end(), llvm::DebugLoc(),
      //                           MCInstInfo->get(IKMI.getOpcode()));
      //   MIRBuilder->dump();
      //   for (auto &Op : MIRBuilder->operands()) {
      //     Op.dump();
      //   }
      // }
    }
    llvm::outs() << "=====> End of Instrumentation Logic\n"
                 << "     > Dump new App Machine Function\n";
    ToBeInstrumentedMF->dump();

// Old code for patching in call to Instru Func into App module
/*
    MIRBuilder = llvm::BuildMI(*IPointMBB, InstPoint, llvm::DebugLoc(),
                            MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKUP))
                  .addImm(0)
                  .addImm(0);
    MIRBuilder =
        llvm::BuildMI(*IPointMBB, MIRBuilder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::SI_PC_ADD_REL_OFFSET))
            .addDef(PCReg)
            .addGlobalAddress(&CalleeFunction, 0,
                              llvm::SIInstrInfo::MO_REL32_LO)
            .addGlobalAddress(&CalleeFunction, 0,
                              llvm::SIInstrInfo::MO_REL32_HI);

    auto SaveReg = IPointMBB->getParent()->getRegInfo().createVirtualRegister(
        &llvm::AMDGPU::SGPR_128RegClass);
    MIRBuilder =
        llvm::BuildMI(*IPointMBB, MIRBuilder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::COPY))
            .addDef(SaveReg)
            .addReg(llvm::AMDGPU::PRIVATE_RSRC_REG);

    MIRBuilder =
        llvm::BuildMI(*IPointMBB, MIRBuilder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::COPY))
            .addDef(llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3)
            .addReg(SaveReg);

    auto RegMask =
        TRI->getCallPreservedMask(CalleeMF, CalleeFunction.getCallingConv());

    MIRBuilder =
        llvm::BuildMI(*IPointMBB, MIRBuilder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::SI_CALL))
            .addReg(llvm::AMDGPU::SGPR30_SGPR31, llvm::RegState::Define)
            .addReg(PCReg, llvm::RegState::Kill)
            .addGlobalAddress(&CalleeFunction)
            .add(llvm::MachineOperand::CreateReg(
                llvm::AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, false, true))
            .addRegMask(RegMask);


    llvm::outs() << "\n       Insert Adjust callstack up \n"; 
    MIRBuilder =
        llvm::BuildMI(*IPointMBB, MIRBuilder.getInstr()->getNextNode(), llvm::DebugLoc(),
                      MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKDOWN))
            .addImm(0)
            .addImm(0);

*/
    //    IPointMBB->getParent()->dump();
    //                                    .addDef(llvm::AMDGPU::SGPR4_SGPR5,
    //                                    llvm::RegState::Define);
    //      MIB.addGlobalAddress(InstLLVMFunction, 0,
    //      llvm::SIInstrInfo::MO_REL32); MIB.addGlobalAddress(InstLLVMFunction,
    //      0, llvm::SIInstrInfo::MO_REL32 + 1);
    //
    //      MIRBuilder = BuildMI(IPointMBB, I, llvm::DebugLoc(),
    //                        MCInstInfo->get(llvm::AMDGPU::SI_CALL_ISEL))
    //                    .addReg(llvm::AMDGPU::SGPR4_SGPR5,
    //                            llvm::RegState::Kill).addGlobalAddress(InstLLVMFunction);
    ////      MIRBuilder; // =
    //      BuildMI(IPointMBB, I, llvm::DebugLoc(),
    //              MCInstInfo->get(llvm::AMDGPU::ADJCALLSTACKDOWN))
    //          .addImm(0)
    //          .addImm(0);

    Out.push_back(*IFLSI);
  }
  return Out;
}

} // namespace luthier

