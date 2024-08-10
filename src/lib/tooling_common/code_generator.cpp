#include "tooling_common/code_generator.hpp"

#include <memory>

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
#include <llvm/CodeGen/BasicBlockSectionUtils.h>
#include <llvm/CodeGen/LiveIntervals.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "AMDGPUGenInstrInfo.inc"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "common/error.hpp"
#include "common/log.hpp"
#include "hsa/hsa.hpp"
#include "hsa/hsa_agent.hpp"
#include "hsa/hsa_executable.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include "hsa/hsa_intercept.hpp"
#include "hsa/hsa_isa.hpp"
#include "hsa/hsa_loaded_code_object.hpp"
#include "tooling_common/code_lifter.hpp"
#include "tooling_common/target_manager.hpp"
#include "tooling_common/tool_executable_manager.hpp"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Instructions.h"
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
#include <cstdlib>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/Passes/PassBuilder.h>
#include <ranges>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-generator"

namespace luthier {

template <> CodeGenerator *Singleton<CodeGenerator>::Instance{nullptr};

llvm::Error CodeGenerator::linkRelocatableToExecutable(
    const llvm::ArrayRef<char> &Code, const hsa::ISA &ISA,
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

  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      amd_comgr_set_data(DataIn, Code.size(), Code.data())));

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
  const char *LinkOptions[]{"-Wl,--unresolved-symbols=ignore-all"};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_COMGR_SUCCESS_CHECK(
      (amd_comgr_action_info_set_option_list(DataAction, LinkOptions, 1))));
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

static llvm::CallInst *passRegAsArgument(llvm::IRBuilderBase &Builder,
                                         llvm::MCRegister Reg, llvm::Module &M,
                                         llvm::Argument *HookArg) {
  auto &LLVMContext = M.getContext();
  auto *HookArgType = HookArg->getType();

  auto *UnsignedIntType = llvm::Type::getInt32Ty(LLVMContext);
  auto *MCRegConstantInt = llvm::ConstantInt::get(UnsignedIntType, Reg);
  auto *ReadRegFuncType =
      llvm::FunctionType::get(HookArgType, {UnsignedIntType}, false);
  // Format the readReg intrinsic function name
  // TODO: create a common place/method for constructing luthier intrinsics
  std::string FormattedIntrinsicName{"luthier::readReg"};
  llvm::raw_string_ostream FINOS(FormattedIntrinsicName);
  // Format the output type
  FINOS << ".";
  ReadRegFuncType->getReturnType()->print(FINOS);
  // Format the input type
  FINOS << ".";
  UnsignedIntType->print(FINOS);
  // Create the read reg intrinsic function
  auto ReadRegFunc =
      M.getOrInsertFunction(FormattedIntrinsicName, ReadRegFuncType,
                            llvm::AttributeList().addFnAttribute(
                                LLVMContext, LUTHIER_INTRINSIC_ATTRIBUTE));

  return Builder.CreateCall(ReadRegFunc, {MCRegConstantInt});
}

llvm::Expected<llvm::Function &> CodeGenerator::generateHookIR(
    const llvm::MachineInstr &MI,
    const llvm::ArrayRef<InstrumentationTask::hook_invocation_descriptor>
        HookSpecs,
    const llvm::GCNTargetMachine &TM, llvm::Module &IModule) {
  // 1. Create an empty kernel per MI insertion point in the LCO Module with
  // no input arguments
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(IModule.getContext());
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {}, false);
  auto *HookIRKernel =
      llvm::Function::Create(FunctionType, llvm::GlobalValue::ExternalLinkage,
                             llvm::formatv("HookAt{0}", &MI), IModule);
  HookIRKernel->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  // Prevent emission of the prologue/epilogue code, but still lower the stack
  // operands
  HookIRKernel->addFnAttr(llvm::Attribute::Naked);
  // Create an empty basic block to fill in with calls to hooks
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(IModule.getContext(), "", HookIRKernel);
  llvm::IRBuilder<> Builder(BB);
  LLVM_DEBUG(llvm::dbgs() << "Number of hooks to be inserted for MI: "
                          << HookSpecs.size() << "\n");
  for (const auto &HookSpec : HookSpecs) {
    auto Hook = IModule.getFunction(HookSpec.HookName);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Hook != nullptr));
    // Get the arguments of the hook
    llvm::SmallVector<llvm::Value *, 4> Operands;
    for (const auto &[Idx, Op] : llvm::enumerate(HookSpec.Args)) {
      if (holds_alternative<llvm::MCRegister>(Op)) {
        // Create a call to the read reg intrinsic to load the MC register
        // into the value
        auto ReadRegVal =
            passRegAsArgument(Builder, std::get<llvm::MCRegister>(Op),
                              *Hook->getParent(), Hook->getArg(Idx));
        Operands.push_back(ReadRegVal);
      } else {
        // Otherwise it's a constant, we can just push it back as an argument
        Operands.push_back(std::get<llvm::Constant *>(Op));
      }
    }
    (void)Builder.CreateCall(Hook, Operands);
  }
  // Put a ret void to end the kernel
  (void)Builder.CreateRetVoid();
  return *HookIRKernel;
}

static void optimizeHookModuleIR(llvm::Module &M, llvm::GCNTargetMachine *TM) {
  // Create the analysis managers.
  // These must be declared in this order so that they are destroyed in
  // the correct order due to inter-analysis-manager references.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  llvm::PassBuilder PB(TM);

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
  MPM.run(M, MAM);
}

std::pair<llvm::MachineModuleInfoWrapperPass *,
          std::unique_ptr<llvm::legacy::PassManager>>
CodeGenerator::runCodeGenPipeline(
    llvm::Module &M, llvm::GCNTargetMachine *TM,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &MIToHookFuncMap,
    const llvm::StringMap<IntrinsicIRLoweringInfo> &ToBeLoweredIntrinsics,
    bool DisableVerify, const LiftedRepresentation &LR) {
  // Create a new pass manager on the heap to have better control over its
  // lifetime and the lifetime of the MMIWP
  auto PM = std::make_unique<llvm::legacy::PassManager>();

  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(M.getTargetTriple()));

  PM->add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  auto *TPC = TM->createPassConfig(*PM);

  TPC->setDisableVerify(DisableVerify);

  PM->add(TPC);

  auto IMMIWP = new llvm::MachineModuleInfoWrapperPass(TM);

  PM->add(IMMIWP);

  TPC->addISelPasses();
  PM->add(new IntrinsicMIRLoweringPass(ToBeLoweredIntrinsics,
                                       IntrinsicsProcessors));
  PM->add(new luthier::DefineLiveRegsAndAppStackUsagePass(MIToHookFuncMap, LR));
  TPC->addMachinePasses();
  TPC->setInitialized();

  PM->run(M);

  for (const auto &F : M) {
    llvm::outs() << "AFTER MOVE:\n";
    llvm::outs() << IMMIWP->getMMI().getModule() << "\n";
    llvm::outs() << "Function name: " << F.getName() << "\n";
    if (auto MF = IMMIWP->getMMI().getMachineFunction(F)) {
      llvm::outs() << "wrapper pass has it:\n";
      MF->print(llvm::outs());
    }
    if (auto MF = IMMIWP->getMMI().getMachineFunction(F)) {
      llvm::outs() << "MMI has it:\n";
      MF->print(llvm::outs());
    }
  }

  return {IMMIWP, std::move(PM)};
}

llvm::Error patchLiftedRepresentation(
    const llvm::Module &IModule, const llvm::MachineModuleInfo &IMMI,
    llvm::Module &LRModule, llvm::MachineModuleInfo &LRMMI,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &MIToHookFuncMap) {
  // A mapping between Global Variables in the instrumentation module and
  // their corresponding Global Variables in the instrumented code
  llvm::ValueToValueMapTy VMap;
  // Clone the instrumentation module Global Variables into the instrumented
  // code
  for (const auto &GV : IModule.globals()) {
    auto *NewGV = new GlobalVariable(
        LRModule, GV.getValueType(), GV.isConstant(), GV.getLinkage(), nullptr,
        GV.getName(), nullptr, GV.getThreadLocalMode(),
        GV.getType()->getAddressSpace());
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
  }
  // Clone the MBBs
  for (const auto &[InsertionPointMI, HookF] : MIToHookFuncMap) {
    // A mapping between a machine basic block in the instrumentation MMI
    // and its destination in the patched instrumented code
    llvm::DenseMap<const llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>
        MBBMap;

    const auto &HookMF = *IMMI.getMachineFunction(*HookF);
    auto &InsertionPointMBB = *InsertionPointMI->getParent();
    auto &ToBeInstrumentedMF = *InsertionPointMBB.getParent();

    // Number of return blocks in the hook
    unsigned int NumReturnBlocksInHook{0};
    // The last return block of the Hook function
    const llvm::MachineBasicBlock *HookLastReturnMBB{nullptr};
    // The target block in the instrumented function which the last return
    // block of the hook will be prepended to if NumReturnBlocksInHook is 1
    // Otherwise, it is the successor of all the return blocks in the hook
    llvm::MachineBasicBlock *HookLastReturnMBBDest{nullptr};

    // Find the last return block of the Hook function + count the number
    // of the return blocks in the hook
    for (const auto &MBB : std::ranges::reverse_view(HookMF)) {
      if (MBB.isReturnBlock() && HookLastReturnMBB == nullptr) {
        HookLastReturnMBB = &MBB;
        NumReturnBlocksInHook++;
      }
    }
    if (HookLastReturnMBB == nullptr)
      llvm_unreachable("No return block found inside the hook");

    if (HookMF.size() > 1) {
      // If there's multiple blocks inside the hook, and the insertion point is
      // not the beginning of the instrumented basic block, then split the
      // insertion point MBB right before the insertion MI, the destination of
      // the last return block of the hook will be the newly-created block
      if (InsertionPointMI->getIterator() != InsertionPointMBB.begin())
        HookLastReturnMBBDest =
            InsertionPointMBB.splitAt(*InsertionPointMI->getPrevNode());
      else
        HookLastReturnMBBDest = &InsertionPointMBB;
      if (NumReturnBlocksInHook == 1)
        MBBMap.insert({HookLastReturnMBB, HookLastReturnMBBDest});
      // If number of return blocks is greater than 1 (very unlikely) we
      // will create a block for it in the next loop
    }

    for (const auto &HookMBB : HookMF) {

      // Special handling of the entry block
      if (HookMBB.isEntryBlock()) {
        // If the Insertion Point is not before the very first instruction,
        // then the Insertion Point MBB will be where the content of the entry
        // block will be appended to
        if (InsertionPointMI->getIterator() != InsertionPointMBB.begin()) {
          MBBMap.insert({&HookMBB, &InsertionPointMBB});
        }
        // If the insertion point is right before the very first instruction
        // of the block, then it should be appended to the return block of the
        // hook instead, unless the hook has only a single basic block
        else if (HookMF.size() == 1) {
          // If there's only a single basic block in the instrumentation
          // function, then the insertion point MBB will be where the hook's
          // first (and last) MBB appends to
          MBBMap.insert({&HookMF.front(), &InsertionPointMBB});
        } else {
          // Otherwise, create a new basic block for the entry block of the
          // hook, and Make all the pred blocks of the Insertion point MBB to
          // point to this newly created block instead
          auto *NewEntryBlock = ToBeInstrumentedMF.CreateMachineBasicBlock();
          ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                    NewEntryBlock);
          // First add the NewEntryBlock as a pred to all
          // InsertionPointMBB's preds
          llvm::SmallVector<llvm::MachineBasicBlock *, 2> PredMBBs;
          for (auto It = InsertionPointMBB.pred_begin();
               It != InsertionPointMBB.pred_end(); ++It) {
            auto PredMBB = *It;
            PredMBB->addSuccessor(NewEntryBlock);
            PredMBBs.push_back(PredMBB);
          }
          // Remove the insertion point mbb from the PredMBB's successor list
          for (auto &PredMBB : PredMBBs) {
            PredMBB->removeSuccessor(&InsertionPointMBB);
          }
          // Add the insertion point MBB as the successor of this block
          NewEntryBlock->addSuccessor(&InsertionPointMBB);
          //
          MBBMap.insert({&HookMF.front(), NewEntryBlock});
        }
      }
      // Special handling for the return blocks
      else if (HookMBB.isReturnBlock()) {
        // If this is not the last return block, or there's more than one return
        // block, then we have to create a new block for it in the target
        // function
        if (NumReturnBlocksInHook > 1 || &HookMBB != HookLastReturnMBB) {
          auto *TargetReturnBlock =
              ToBeInstrumentedMF.CreateMachineBasicBlock();
          ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                    TargetReturnBlock);
          MBBMap.insert({&HookMBB, TargetReturnBlock});
        }
      } else {
        // Otherwise, we only need to create a new basic block in the
        // instrumented code and just copy its contents over
        auto *NewHookMBB = ToBeInstrumentedMF.CreateMachineBasicBlock();
        ToBeInstrumentedMF.insert(HookLastReturnMBBDest->getIterator(),
                                  NewHookMBB);
        MBBMap.insert({&HookMBB, NewHookMBB});
      }
    }

    // Link blocks
    for (auto &MBB : HookMF) {
      auto *DstMBB = MBBMap[&MBB];
      for (auto It = MBB.succ_begin(), IterEnd = MBB.succ_end(); It != IterEnd;
           ++It) {
        auto *SrcSuccMBB = *It;
        auto *DstSuccMBB = MBBMap[SrcSuccMBB];
        if (!DstMBB->isSuccessor(DstSuccMBB))
          DstMBB->addSuccessor(DstSuccMBB, MBB.getSuccProbability(It));
      }
      if (MBB.isReturnBlock() && NumReturnBlocksInHook > 1) {
        // Add the LastHookMBBDest as a successor to the return block
        // if there's more than one return block in the hook
        if (!DstMBB->isSuccessor(HookLastReturnMBBDest))
          DstMBB->addSuccessor(HookLastReturnMBBDest);
      }
    }
    // Finally, clone the instructions into the new MBBs
    const llvm::TargetSubtargetInfo &STI = ToBeInstrumentedMF.getSubtarget();
    const llvm::TargetInstrInfo *TII = STI.getInstrInfo();
    const llvm::TargetRegisterInfo *TRI = STI.getRegisterInfo();
    auto &TargetMFMRI = ToBeInstrumentedMF.getRegInfo();

    llvm::DenseSet<const uint32_t *> ConstRegisterMasks;

    // Track predefined/named regmasks which we ignore.
    for (const uint32_t *Mask : TRI->getRegMasks())
      ConstRegisterMasks.insert(Mask);
    for (const auto &MBB : HookMF) {
      auto *DstMBB = MBBMap[&MBB];
      llvm::MachineBasicBlock::iterator InsertionPoint;
      if (MBB.isReturnBlock() && NumReturnBlocksInHook == 1) {
        InsertionPoint = DstMBB->begin();
      } else if (MBB.isEntryBlock() && HookMF.size() == 1) {
        InsertionPoint = InsertionPointMI->getIterator();
      } else
        InsertionPoint = DstMBB->end();
      if (MBB.isEntryBlock()) {
        auto *DstMI = ToBeInstrumentedMF.CreateMachineInstr(
            TII->get(AMDGPU::S_WAITCNT), llvm::DebugLoc(),
            /*NoImplicit=*/true);
        DstMBB->insert(InsertionPoint, DstMI);
        DstMI->addOperand(llvm::MachineOperand::CreateImm(0));
      }
      for (auto &SrcMI : MBB.instrs()) {
        auto *PreInstrSymbol = SrcMI.getPreInstrSymbol();
        if (MBB.isReturnBlock() && SrcMI.isTerminator()) {
          break;
        }
        // Don't clone the bundle headers
        if (SrcMI.isBundle())
          continue;
        const auto &MCID = TII->get(SrcMI.getOpcode());
        // TODO: Properly import the debug location
        auto *DstMI =
            ToBeInstrumentedMF.CreateMachineInstr(MCID, llvm::DebugLoc(),
                                                  /*NoImplicit=*/true);
        DstMI->setFlags(SrcMI.getFlags());
        DstMI->setAsmPrinterFlag(SrcMI.getAsmPrinterFlags());
        DstMBB->insert(InsertionPoint, DstMI);
        for (auto &SrcMO : SrcMI.operands()) {
          llvm::MachineOperand DstMO(SrcMO);
          DstMO.clearParent();

          // Update MBB.
          if (DstMO.isMBB())
            DstMO.setMBB(MBBMap[DstMO.getMBB()]);
          else if (DstMO.isRegMask()) {
            TargetMFMRI.addPhysRegsUsedFromRegMask(DstMO.getRegMask());

            if (!ConstRegisterMasks.count(DstMO.getRegMask())) {
              uint32_t *DstMask = ToBeInstrumentedMF.allocateRegMask();
              std::memcpy(
                  DstMask, SrcMO.getRegMask(),
                  sizeof(*DstMask) *
                      llvm::MachineOperand::getRegMaskSize(TRI->getNumRegs()));
              DstMO.setRegMask(DstMask);
            }
          } else if (DstMO.isGlobal()) {
            auto GVEntry = VMap.find(DstMO.getGlobal());
            LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(GVEntry != VMap.end()));
            auto *DestGV = cast<llvm::GlobalValue>(GVEntry->second);
            DstMO.ChangeToGA(DestGV, DstMO.getOffset(), DstMO.getTargetFlags());
          }

          DstMI->addOperand(DstMO);
        }
      }
      if (MBB.isReturnBlock() || (MBB.isEntryBlock() && HookMF.size() == 1)) {
        auto *DstMI = ToBeInstrumentedMF.CreateMachineInstr(
            TII->get(AMDGPU::S_WAITCNT), llvm::DebugLoc(),
            /*NoImplicit=*/true);
        DstMBB->insert(InsertionPoint, DstMI);
        DstMI->addOperand(llvm::MachineOperand::CreateImm(0));
      }
      if (NumReturnBlocksInHook > 1 && MBB.isReturnBlock()) {
        TII->insertUnconditionalBranch(*DstMBB, HookLastReturnMBBDest,
                                       llvm::DebugLoc());
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
CodeGenerator::printAssembly(llvm::Module &Module, llvm::GCNTargetMachine &TM,
                             llvm::MachineModuleInfoWrapperPass *MMIWP,
                             llvm::SmallVectorImpl<char> &CompiledObjectFile,
                             llvm::CodeGenFileType FileType) {
  // Argument error checking
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(FileType != CodeGenFileType::Null));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(MMIWP != nullptr));

  auto &MMI = MMIWP->getMMI();
  // Create the legacy pass manager with minimal passes to print the
  // assembly file
  llvm::legacy::PassManager PM;
  // Add the target library info pass
  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(Module.getTargetTriple()));
  PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
  // TargetPassConfig is expected by the passes involved, so it must be added
  auto *TPC = TM.createPassConfig(PM);
  // Don't run the machine verifier after each pass
  TPC->setDisableVerify(true);
  TPC->setInitialized();
  PM.add(TPC);
  // Add the MMI Wrapper pass
  PM.add(MMIWP);
  // Add the resource usage analysis, which is in charge of calculating the
  // kernel descriptor and the metadata fields
  PM.add(new llvm::AMDGPUResourceUsageAnalysis());

  // Finally, add the Assembly printer pass
  llvm::raw_svector_ostream ObjectFileOS(CompiledObjectFile);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(!TM.addAsmPrinter(
      PM, ObjectFileOS, nullptr, FileType, MMI.getContext())));

  // Run the passes on the module to print the assembly
  PM.run(Module);
  return llvm::Error::success();
}

llvm::Error CodeGenerator::processInstModuleIntrinsicsAtIRLevel(
    llvm::Module &Module, const llvm::GCNTargetMachine &TM,
    llvm::StringMap<IntrinsicIRLoweringInfo> &InlineAsmMIRMap) {
  int NextIntrinsicIdx = 1;
  for (auto &F : llvm::make_early_inc_range(Module.functions())) {
    if (F.hasFnAttribute(LUTHIER_INTRINSIC_ATTRIBUTE)) {
      // Find the processor for this intrinsic
      auto IntrinsicName =
          F.getFnAttribute(LUTHIER_INTRINSIC_ATTRIBUTE).getValueAsString();
      // Ensure the processor is indeed registered with the Code Generator
      auto It = IntrinsicsProcessors.find(IntrinsicName);
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(It != IntrinsicsProcessors.end()));
      llvm::outs() << "Num uses: " << F.getNumUses() << "\n";
      // Iterate over all users of the intrinsic
      for (auto *User : llvm::make_early_inc_range(F.users())) {
        User->dump();
        // Ensure the user is a Call instruction; Anything other usage is
        // illegal
        auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User);
        LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CallInst != nullptr));
        auto IRLoweringInfo = It->second.IRProcessor(F, *CallInst, TM);
        LUTHIER_RETURN_ON_ERROR(IRLoweringInfo.takeError());
        IRLoweringInfo->setIntrinsicName(IntrinsicName);
        // Add the output operand constraint, if the output is not void
        auto ReturnValInfo = IRLoweringInfo->getReturnValueInfo();
        std::string Constraint;
        if (!ReturnValInfo.Val->getType()->isVoidTy())
          Constraint += "=" + ReturnValInfo.Constraint;
        // Construct argument type vector
        llvm::SmallVector<llvm::Type *, 4> ArgTypes;
        llvm::SmallVector<llvm::Value *, 4> ArgValues;
        ArgTypes.reserve(IRLoweringInfo->getArgsInfo().size());
        ArgValues.reserve(IRLoweringInfo->getArgsInfo().size());
        for (const auto &ArgInfo : IRLoweringInfo->getArgsInfo()) {
          ArgTypes.push_back(ArgInfo.Val->getType());
          ArgValues.push_back(const_cast<llvm::Value *>(ArgInfo.Val));
          Constraint += ReturnValInfo.Constraint;
        }
        auto *AsmCallInst = llvm::CallInst::Create(
            llvm::InlineAsm::get(
                llvm::FunctionType::get(ReturnValInfo.Val->getType(), ArgTypes,
                                        false),
                llvm::to_string(NextIntrinsicIdx), Constraint, true),
            ArgValues);
        llvm::outs() << "After mods before inserting\n";
        AsmCallInst->insertBefore(*CallInst->getParent(),
                                  CallInst->getIterator());
        CallInst->replaceAllUsesWith(AsmCallInst);
        // transfer debug info from the original invoke to the inline assembly
        AsmCallInst->setDebugLoc(CallInst->getDebugLoc());
        AsmCallInst->print(llvm::outs());
        CallInst->eraseFromParent();
        InlineAsmMIRMap.insert(
            {llvm::to_string(NextIntrinsicIdx), std::move(*IRLoweringInfo)});
        NextIntrinsicIdx++;
        //        Module.print(llvm::outs(), nullptr);
      }
      F.dropAllReferences();
      F.eraseFromParent();
      llvm::outs() << "After mods after inserting\n";
    }
  }
  return llvm::Error::success();
}

llvm::Error CodeGenerator::insertHooks(LiftedRepresentation &LR,
                                       const InstrumentationTask &Task) {
  // Early exit if no hooks are to be inserted into the LR
  if (Task.getHookInsertionTasks().empty())
    return llvm::Error::success();
  auto Lock = LR.getContext().getLock();
  // Since (at least in theory) each LCO can have its own ISA, we need to
  // create a module/MMI per LCO to generate instrumentation code for easier
  // separation between different LCOs' Machine code
  for (auto &[LCO, LCOModule] : LR) {
    auto Agent = hsa::LoadedCodeObject(LCO).getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());
    // Load the bitcode of the instrumentation module into the LR's context
    auto IModuleTS =
        Task.getModule().readBitcodeIntoContext(LR.getContext(), *Agent);
    LUTHIER_RETURN_ON_ERROR(IModuleTS.takeError());
    auto &IModule = *IModuleTS->getModuleUnlocked();
    LLVM_DEBUG(llvm::dbgs() << "Instrumentation Module Contents: \n";
               IModule.print(llvm::dbgs(), nullptr););
    auto &TM = LR.getTargetMachine<llvm::GCNTargetMachine>();
    // Now that everything has been created we can start inserting hooks
    LLVM_DEBUG(llvm::dbgs() << "Number of MIs to get hooks: "
                            << Task.getHookInsertionTasks().size() << "\n");
    // A map to keep track of Hooks per MI
    llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> MIToHookFuncMap;

    for (const auto &[MI, HookSpecs] : Task.getHookInsertionTasks()) {
      // Generate the Hooks for each MI
      auto HookFunc = generateHookIR(*MI, HookSpecs, TM, IModule);
      LUTHIER_RETURN_ON_ERROR(HookFunc.takeError());
      MIToHookFuncMap.insert({MI, &(*HookFunc)});
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Instrumentation Module After Hooks Inserted:\n");
    LLVM_DEBUG(IModule.print(llvm::dbgs(), nullptr));
    // With the Hook IR generated, we put it through the normal IR
    // pipeline
    optimizeHookModuleIR(IModule, &TM);
    LLVM_DEBUG(llvm::dbgs()
                   << "Instrumentation Module after IR optimization:\n";
               IModule.print(llvm::dbgs(), nullptr));
    // Replace all Luthier intrinsics with dummy inline assembly and correct
    // arguments
    llvm::StringMap<IntrinsicIRLoweringInfo> DummyAsmToIRLoweringInfoMap;
    LUTHIER_RETURN_ON_ERROR(processInstModuleIntrinsicsAtIRLevel(
        IModule, TM, DummyAsmToIRLoweringInfoMap));
    // Run the code gen pipeline, while enforcing the stack and register
    // constraints
    auto [MMIWP, PM] = runCodeGenPipeline(
        IModule, &TM, MIToHookFuncMap, DummyAsmToIRLoweringInfoMap, false, LR);
    PM.release();
    for (const auto &F : IModule) {
      llvm::outs() << "AFTER MOVE:\n";
      llvm::outs() << MMIWP->getMMI().getModule() << "\n";
    }
    // Finally, patch in the generated machine code into the lifted
    // representation
    LUTHIER_RETURN_ON_ERROR(
        patchLiftedRepresentation(IModule, MMIWP->getMMI(), *LCOModule.first,
                                  *LCOModule.second, MIToHookFuncMap));

    for (const auto &F : *LCOModule.first) {
      llvm::outs() << "Function name inside: " << F.getName() << "\n";
      llvm::outs() << LCOModule.second << "\n";
      if (auto MF = LCOModule.second->getMachineFunction(F)) {
        llvm::outs() << "Function pointer location: " << MF << "\n";
        MF->print(llvm::outs());
      }
    }
  };
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<LiftedRepresentation>> CodeGenerator::instrument(
    const LiftedRepresentation &LR,
    llvm::function_ref<llvm::Error(InstrumentationTask &,
                                   LiftedRepresentation &)>
        Mutator) {
  // Clone the Lifted Representation
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(
      std::unique_ptr<LiftedRepresentation>, ClonedLR,
      CodeLifter::instance().cloneRepresentation(LR));
  // Run the mutator function on the Lifted Representation and populate the
  // instrumentation task
  InstrumentationTask IT(*ClonedLR);
  LUTHIER_RETURN_ON_ERROR(Mutator(IT, *ClonedLR));
  // Insert the hooks inside the Lifted Representation
  LUTHIER_RETURN_ON_ERROR(insertHooks(*ClonedLR, IT));
  for (const auto &[LCO, LCOModule] : *ClonedLR) {
    for (const auto &F : *LCOModule.first) {
      llvm::outs() << "Function name outside: " << F.getName() << "\n";
      llvm::outs() << LCOModule.second << "\n";
      if (auto MF = LCOModule.second->getMachineFunction(F)) {
        llvm::outs() << "Function pointer location: " << MF << "\n";
        MF->print(llvm::outs());
      }
    }
  }

  return std::move(ClonedLR);
}

char DefineLiveRegsAndAppStackUsagePass::ID = 0;

DefineLiveRegsAndAppStackUsagePass::DefineLiveRegsAndAppStackUsagePass(
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &MIToHookFuncMap,
    const LiftedRepresentation &LR)
    : llvm::MachineFunctionPass(ID) {
  // Fetch the Live-ins at each instruction for before MI hooks from the LR
  for (const auto &[MI, HookKernel] : MIToHookFuncMap) {
    auto *MBB = MI->getParent();
    auto *MF = MBB->getParent();
    auto &MILivePhysRegs = *LR.getLiveInPhysRegsOfMachineInstr(*MI);
    (void)HookLiveRegs.insert(
        {HookKernel, const_cast<llvm::LivePhysRegs *>(&MILivePhysRegs)});

    LLVM_DEBUG(llvm::dbgs() << "Live regs at instruction " << MI << ": \n";
               auto *TRI = MF->getSubtarget().getRegisterInfo();
               for (const auto &Reg
                    : MILivePhysRegs) {
                 llvm::dbgs() << TRI->getRegAsmName(Reg) << "\n";
               });
  }
  // Fetch the upper bound of the stack used by each insertion point
  for (const auto &[MI, HookKernel] : MIToHookFuncMap) {
    // Get the function of this MI
    auto &InstrumentedMF = *MI->getParent()->getParent();
    auto &InstrumentedFunction = InstrumentedMF.getFunction();
    // If the function of the MI is a kernel, then its private segment usage
    // is known by the metadata, or it has dynamic stack usage
    auto CC = InstrumentedFunction.getCallingConv();
    if (CC == llvm::CallingConv::AMDGPU_KERNEL) {
      auto &FrameInfo = InstrumentedMF.getFrameInfo();
      if (FrameInfo.hasVarSizedObjects())
        llvm_unreachable("Dynamic stack kernels are not yet implemented");
      else
        StaticSizedHooksToStackSize.insert(
            {HookKernel, FrameInfo.getStackSize()});
    } else {
      // If this is a device function, then its stack usage is the kernel that
      // calls it with the largest stack usage
      size_t LargestStackUsage = 0;
      llvm::SmallPtrSet<llvm::Function *, 3> VisitedFunctions{};
      llvm::SmallVector<llvm::MachineInstr *> UnvisitedUses(
          LR.getUsesOfGlobalValue(InstrumentedFunction));
      while (UnvisitedUses.empty()) {
        auto &CurrentUse = UnvisitedUses.front();
        auto &UseFunction = CurrentUse->getParent()->getParent()->getFunction();
        if (UseFunction.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
          auto &FrameInfo = InstrumentedMF.getFrameInfo();
          if (FrameInfo.hasVarSizedObjects())
            llvm_unreachable("Dynamic stack kernels are not yet implemented");
          else {
            if (LargestStackUsage < FrameInfo.getStackSize()) {
              LargestStackUsage = FrameInfo.getStackSize();
            }
          }
        } else {
          if (!VisitedFunctions.contains(&UseFunction)) {
            for (const auto &NewUse : LR.getUsesOfGlobalValue(UseFunction)) {
              UnvisitedUses.push_back(NewUse);
            }
            VisitedFunctions.insert(&UseFunction);
          }
        }
        UnvisitedUses.erase(UnvisitedUses.begin());
      }
    }
  }
}

bool DefineLiveRegsAndAppStackUsagePass::runOnMachineFunction(
    MachineFunction &MF) {
  auto *F = &MF.getFunction();
  auto &MRI = MF.getRegInfo();
  auto *TRI = MF.getSubtarget().getRegisterInfo();
  auto &EntryMBB = MF.front();

  // Find the entry block of the function, and mark the live-ins
  auto &LivePhysRegs = *HookLiveRegs[F];

  for (auto &LiveIn : EntryMBB.liveins()) {
    LivePhysRegs.addReg(LiveIn.PhysReg);
  }
  EntryMBB.clearLiveIns();

  // TODO: Fix machine verifier's live-in use detection issue
  for (auto &MBB : MF) {
    llvm::addLiveIns(MBB, LivePhysRegs);
  }

  // If the SCC bit is live before entering the hook, then we need to save in
  // the very first instruction and restore it before all return instructions
  if (EntryMBB.isLiveIn(llvm::AMDGPU::SCC)) {
    // Put a copy from the SCC register to a virtual scalar 32-bit register in
    // the
    auto &CopyMCID = MF.getSubtarget().getInstrInfo()->get(llvm::AMDGPU::COPY);
    auto VirtualSReg =
        MF.getRegInfo().createVirtualRegister(&llvm::AMDGPU::SReg_32RegClass);
    llvm::BuildMI(EntryMBB, EntryMBB.begin(), llvm::DebugLoc(), CopyMCID)
        .addReg(VirtualSReg, llvm::RegState::Define)
        .addReg(llvm::AMDGPU::SCC, llvm::RegState::Kill);
    // Iterate over all MBBs, and add a copy back before the term instruction
    // inside all return blocks
    for (auto &MBB : MF) {
      if (MBB.isReturnBlock()) {
        llvm::BuildMI(MBB, MBB.getFirstTerminator(), llvm::DebugLoc(), CopyMCID)
            .addReg(llvm::AMDGPU::SCC, llvm::RegState::Define)
            .addReg(VirtualSReg, llvm::RegState::Kill);
      }
    }
  }

  //  for (auto &MBB: MF) {
  //    for (auto &LiveIn : MF.front().liveins()) {
  ////      if (MBB.back().getOpcode() != llvm::AMDGPU::SI_END_CF)
  //// MBB.back().addOperand(llvm::MachineOperand::CreateReg(LiveIn.PhysReg,
  /// false, true)); /      auto MCID =
  /// MF.getSubtarget().getInstrInfo()->get(llvm::AMDGPU::COPY); /
  /// llvm::BuildMI(*MF.begin(), MF.begin()->begin(), llvm::DebugLoc(),
  /// MCID).addReg(LiveIn, llvm::RegState::Define); /
  /// MBB.begin()->addOperand(llvm::MachineOperand::CreateReg(LiveIn.PhysReg,
  /// true, true)); /      MBB.begin()->print(llvm::outs());
  //    }
  //  }

  if (StaticSizedHooksToStackSize.contains(F) &&
      StaticSizedHooksToStackSize.at(F) != 0) {
    // Create a fixed stack operand at the bottom
    MF.getFrameInfo().CreateFixedObject(StaticSizedHooksToStackSize.at(F), 0,
                                        true);
  }

  for (auto &MBB : MF) {
    if (MBB.isReturnBlock()) {
      for (auto &LiveIn : MF.begin()->liveins()) {
        MBB.back().addOperand(
            llvm::MachineOperand::CreateReg(LiveIn.PhysReg, false, true));
      }
    }
  }

  return true;
}
void DefineLiveRegsAndAppStackUsagePass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreservedID(llvm::MachineLoopInfoID);
  AU.addPreserved<llvm::SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
};

char IntrinsicMIRLoweringPass::ID = 0;

bool IntrinsicMIRLoweringPass::runOnMachineFunction(MachineFunction &MF) {
  bool Changed{false};
  // The set of physical registers used without being defined in the body
  // after intrinsics has been lowered
  llvm::LivePhysRegs InsertedPhysRegs(*MF.getSubtarget().getRegisterInfo());
  for (auto &MBB : MF) {
    for (auto &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.isInlineAsm()) {
        // The Asm string is of type symbol
        auto IntrinsicIdx =
            MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
        auto It = MIRLoweringMap.find(IntrinsicIdx);
        if (It == MIRLoweringMap.end())
          MF.getFunction().getContext().emitError(
              "Intrinsic ID was not found in the MIR Lowering Map.");
        llvm::SmallVector<std::pair<InlineAsm::Flag, llvm::Register>, 4> ArgVec;
        for (unsigned I = InlineAsm::MIOp_FirstOperand,
                      NumOps = MI.getNumOperands();
             I < NumOps; ++I) {
          const MachineOperand &MO = MI.getOperand(I);
          if (!MO.isImm())
            continue;
          const InlineAsm::Flag F(MO.getImm());
          const llvm::Register Reg(MI.getOperand(I + 1).getReg());
          ArgVec.emplace_back(F, Reg);
          // Skip to one before the next operand descriptor, if it exists.
          I += F.getNumOperandRegisters();
        }
        auto *TII = MF.getSubtarget().getInstrInfo();
        llvm::SmallVector<llvm::MachineInstr *, 4> AddedMIs;

        auto MIBuilder = [&](int Opcode) {
          auto Builder =
              llvm::BuildMI(MBB, MI, llvm::MIMetadata(MI), TII->get(Opcode));
          AddedMIs.push_back(Builder.getInstr());
          return Builder;
        };

        auto IRProcessor =
            IntrinsicsProcessors.find(It->second.getIntrinsicName());
        if (IRProcessor == IntrinsicsProcessors.end())
          MF.getFunction().getContext().emitError(
              "Intrinsic processor was not found in the intrinsic processor "
              "map.");
        if (auto Err = IRProcessor->second.MIRProcessor(It->second, ArgVec,
                                                        MIBuilder)) {
          MF.getFunction().getContext().emitError(
              "Failed to lower the intrinsic; Error message: " +
              toString(std::move(Err)));
        }
        // Remove the dummy inline assembly
        MI.eraseFromParent();
        Changed = true;
        // Take the
        for (auto *AddedMI : AddedMIs) {
          for (const auto &Op : AddedMI->operands()) {
            if (Op.isReg() && Op.isUse() && Op.getReg().isPhysical()) {
              InsertedPhysRegs.addReg(Op.getReg().asMCReg());
            }
          }
        }
      }
    }
  }
  // Update the used physical defs without defines to keep the machine verifier
  // happy
  llvm::addLiveIns(MF.front(), InsertedPhysRegs);
  return Changed;
}
} // namespace luthier
