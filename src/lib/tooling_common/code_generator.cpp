#include "tooling_common/code_generator.hpp"

#include <memory>

#define protected public
#include <llvm/CodeGen/TargetPassConfig.h>
#undef protected
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
#include <llvm/ExecutionEngine/Orc/IndirectionUtils.h>
#include <llvm/Passes/PassBuilder.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-generator"

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

llvm::Expected<llvm::InlineAsm *>
createInlineAsmMCRegLoad(llvm::MCRegister Reg, const llvm::GCNTargetMachine &TM,
                         const llvm::Function &Kernel) {
  // First detect what kind of Physical register we're dealing with
  auto *TRI = TM.getSubtargetImpl(Kernel)->getRegisterInfo();
  auto *PhysRegClass = TRI->getPhysRegBaseClass(Reg);
  auto RegName = TRI->getRegAsmName(Reg);
  llvm::outs() << "Reg name" << RegName << "\n";
  auto PhysRegWidth = llvm::AMDGPU::getRegBitWidth(*PhysRegClass);
  if (PhysRegWidth != 32)
    llvm_unreachable("Non-32 bit register arguments are not yet supported");
  if (llvm::SIRegisterInfo::isVGPRClass(PhysRegClass)) {
    return llvm::InlineAsm::get(
        llvm::FunctionType::get(llvm::Type::getInt32Ty(Kernel.getContext()),
                                {}),
        llvm::formatv("v_mov_b32 $0 {0}", RegName).str(), "=v", true);
  } else if (llvm::SIRegisterInfo::isSGPRClass(PhysRegClass)) {
    return llvm::InlineAsm::get(
        llvm::FunctionType::get(llvm::Type::getInt32Ty(Kernel.getContext()),
                                {}),
        llvm::formatv("s_mov_b32 $0 {0}", RegName).str(), "=s", true);
  } else
    llvm_unreachable(
        "Adding arguments to the specified register is not implemented yet");
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
                             llvm::formatv("HookAt{1}", &MI), IModule);
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
    for (const auto &Op : HookSpec.Args) {
      if (holds_alternative<llvm::MCRegister>(Op)) {
        // Create an inline assembly to load the MC register into the value
        auto RegLoadInlineAsm = createInlineAsmMCRegLoad(
            std::get<llvm::MCRegister>(Op), TM, *HookIRKernel);
        LUTHIER_RETURN_ON_ERROR(RegLoadInlineAsm.takeError());
        // Put a call to the inline assembly
        auto *InlineAsmCall = Builder.CreateCall(*RegLoadInlineAsm);
        Operands.push_back(InlineAsmCall);
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
    bool DisableVerify) {
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
  PM->add(new luthier::ReserveLiveRegs(MIToHookFuncMap));
  // TODO: Stack Frame Offset correction
  //   PM->add(new luthier::StackFrameOffset(<#initializer #>, <#initializer #>,
  //                                         <#initializer #>))
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
  llvm::ValueToValueMapTy VMap;
  llvm::DenseMap<const llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>
      MBBMap;
  llvm::DenseSet<const llvm::MachineBasicBlock *> ReturnBlocks;
  // Clone the Global variables
  for (const auto &GV : IModule.globals()) {
    auto *NewGV = new GlobalVariable(
        LRModule, GV.getValueType(), GV.isConstant(), GV.getLinkage(), nullptr,
        GV.getName(), nullptr, GV.getThreadLocalMode(),
        GV.getType()->getAddressSpace());
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
  }
  // Clone the MBBs
  for (const auto &[MI, HookF] : MIToHookFuncMap) {
    const auto &HookMF = *IMMI.getMachineFunction(*HookF);
    auto &TargetMBB = *MI->getParent();
    auto &TargetMF = *TargetMBB.getParent();

    for (const auto &MBB : HookMF) {
      if (MBB.isEntryBlock()) {
        if (MI->getIterator() != TargetMBB.begin())
          MBBMap.insert({&MBB, &TargetMBB});
      }
      // This else/if statement will count an MBB that is both entry/return as
      // an entry block only. This will cause ReturnBlocks to have zero blocks
      else if (MBB.isReturnBlock()) {
        ReturnBlocks.insert(&MBB);
      } else {
        auto *NewTargetMBB = TargetMF.CreateMachineBasicBlock();
        TargetMF.push_back(NewTargetMBB);
        MBBMap.insert({&MBB, NewTargetMBB});
      }
    }
    if (MI->getIterator() == TargetMBB.begin()) {
      if (ReturnBlocks.empty()) {
        MBBMap.insert({&HookMF.front(), &TargetMBB});
      } else {
        auto *NewEntryBlock = TargetMF.CreateMachineBasicBlock();
        TargetMF.insert(TargetMBB.getIterator(), NewEntryBlock);
        for (auto It = TargetMBB.pred_begin(), IterEnd = TargetMBB.pred_end();
             It != IterEnd; ++It) {
          auto PredMBB = *It;
          PredMBB->removeSuccessor(&TargetMBB);
          PredMBB->addSuccessor(NewEntryBlock);
        }
        MBBMap.insert({&HookMF.front(), NewEntryBlock});
      }
    }

    // Special logic for handling return blocks of hooks and creating the
    // term block
    llvm::MachineBasicBlock *TermBlock{nullptr};
    // If there's at least a return block, then splice the MBB right before the
    // MI
    // If MI is the first instruction in the block, then the TargetMBB will
    // become the return block's dest
    if (!ReturnBlocks.empty()) {
      if (MI->getIterator() != TargetMBB.begin())
        TermBlock = TargetMBB.splitAt(*MI->getPrevNode());
      else
        TermBlock = &TargetMBB;
      // if there's only one return block, then the return block will be mapped
      // to the term block
      if (ReturnBlocks.size() == 1) {
        MBBMap.insert({*ReturnBlocks.begin(), TermBlock});
        // if there are more than one return blocks, then we need to create
        // blocks for each of them
      } else if (ReturnBlocks.size() > 1) {
        for (const auto *ReturnBlock : ReturnBlocks) {
          auto *TargetReturnBlock = TargetMF.CreateMachineBasicBlock();
          TargetMF.push_back(TargetReturnBlock);
          MBBMap.insert({ReturnBlock, TargetReturnBlock});
        }
      };
    }
    // Link blocks
    for (auto &MBB : HookMF) {
      auto *DstMBB = MBBMap[&MBB];
      for (auto It = MBB.succ_begin(), IterEnd = MBB.succ_end(); It != IterEnd;
           ++It) {
        auto *SrcSuccMBB = *It;
        auto *DstSuccMBB = MBBMap[SrcSuccMBB];
        if (DstMBB->succ_empty())
          DstMBB->addSuccessor(DstSuccMBB, MBB.getSuccProbability(It));
      }
    }
    if (TermBlock != nullptr && TermBlock->pred_empty()) {
      for (const auto *ReturnBlock : ReturnBlocks) {
        auto *TargetReturnBlock = MBBMap[ReturnBlock];
        TargetReturnBlock->addSuccessor(TermBlock, BranchProbability::getOne());
      }
    }
    // Finally, clone the instructions into the new MBBs
    const llvm::TargetSubtargetInfo &STI = TargetMF.getSubtarget();
    const llvm::TargetInstrInfo *TII = STI.getInstrInfo();
    const llvm::TargetRegisterInfo *TRI = STI.getRegisterInfo();
    auto &TargetMFMRI = TargetMF.getRegInfo();

    llvm::DenseSet<const uint32_t *> ConstRegisterMasks;

    // Track predefined/named regmasks which we ignore.
    for (const uint32_t *Mask : TRI->getRegMasks())
      ConstRegisterMasks.insert(Mask);
    for (const auto &MBB : HookMF) {

      auto *DstMBB = MBBMap[&MBB];
      llvm::MachineBasicBlock::iterator InsertionPoint;
      if (ReturnBlocks.contains(&MBB)) {
        InsertionPoint = DstMBB->begin();
      } else if (MBB.isEntryBlock() && HookMF.size() == 1) {
        InsertionPoint = MI->getIterator();
      } else
        InsertionPoint = DstMBB->end();
      if (MBB.isEntryBlock()) {
        auto *DstMI = TargetMF.CreateMachineInstr(TII->get(AMDGPU::S_WAITCNT),
                                                  llvm::DebugLoc(),
                                                  /*NoImplicit=*/true);
        DstMBB->insert(InsertionPoint, DstMI);
        DstMI->addOperand(llvm::MachineOperand::CreateImm(0));
      }
      for (auto &SrcMI : MBB.instrs()) {
        auto *PreInstrSymbol = SrcMI.getPreInstrSymbol();
        if (PreInstrSymbol != nullptr &&
            PreInstrSymbol->getName() == "use-instr") {
          break;
        }
        // Don't clone the bundle headers
        if (SrcMI.isBundle())
          continue;
        const auto &MCID = TII->get(SrcMI.getOpcode());
        // TODO: Properly import the debug location
        auto *DstMI = TargetMF.CreateMachineInstr(MCID, llvm::DebugLoc(),
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
              uint32_t *DstMask = TargetMF.allocateRegMask();
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
      if (ReturnBlocks.contains(&MBB) ||
          (MBB.isEntryBlock() && HookMF.size() == 1)) {
        auto *DstMI = TargetMF.CreateMachineInstr(TII->get(AMDGPU::S_WAITCNT),
                                                  llvm::DebugLoc(),
                                                  /*NoImplicit=*/true);
        DstMBB->insert(InsertionPoint, DstMI);
        DstMI->addOperand(llvm::MachineOperand::CreateImm(0));
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
printAssembly(LiftedRepresentation &LR,
              llvm::DenseMap<hsa::LoadedCodeObject, llvm::SmallVector<char>>
                  &CompiledRelocatables,
              llvm::DenseMap<hsa::LoadedCodeObject, llvm::SmallVector<char>>
                  *AssemblyFiles = nullptr) {
  auto Lock = LR.getContext().getLock();
  for (auto &[LCO, LCOModule] : LR) {
    for (const auto &F : *LCOModule.first) {
      llvm::outs() << F.getName() << "\n";
      if (auto MF = LCOModule.second->getMachineFunction(F)) {
        MF->print(llvm::outs());
      }
    }
    auto &[M, MMI] = LCOModule;
    llvm::legacy::PassManager PM;
    llvm::MCContext &MCContext = MMI->getContext();
    // Get the target machine of the LCO's ISA
    hsa::LoadedCodeObject LCOWrapper(LCO);
    auto &TM = LR.getTargetMachine<llvm::GCNTargetMachine>();
    llvm::TargetLibraryInfoImpl TLII(llvm::Triple(M->getTargetTriple()));
    PM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
    auto *TPC = TM.createPassConfig(PM);
    TPC->setDisableVerify(true);
    TPC->setInitialized();
    PM.add(TPC);
    auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(std::move(*MMI));
    PM.add(MMIWP);
    auto UsageAnalysis = new llvm::AMDGPUResourceUsageAnalysis();
    PM.add(UsageAnalysis);
    if (!CompiledRelocatables.contains(LCOWrapper)) {
      CompiledRelocatables.insert({LCOWrapper, {}});
    }
    llvm::raw_svector_ostream ObjectFileOS(CompiledRelocatables[LCOWrapper]);

    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        !TM.addAsmPrinter(PM, ObjectFileOS, nullptr,
                          llvm::CodeGenFileType::ObjectFile, MCContext)));

    std::unique_ptr<llvm::raw_svector_ostream> AssemblyFileOS;
    if (AssemblyFiles != nullptr) {
      if (!AssemblyFiles->contains(LCOWrapper)) {
        AssemblyFiles->insert({LCOWrapper, {}});
      }
      AssemblyFileOS = std::make_unique<llvm::raw_svector_ostream>(
          (*AssemblyFiles)[LCOWrapper]);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
          !TM.addAsmPrinter(PM, *AssemblyFileOS, nullptr,
                            llvm::CodeGenFileType::ObjectFile, MCContext)));
    }
    llvm::outs() << "Running the printer pass manager\n";
    PM.run(*M); // Run all the passes
                //    MMI =
    //    std::make_unique<llvm::MachineModuleInfo>(std::move(MMIWP->getMMI()));
    //    llvm::outs() << "AFTER MOVE: \n";
    //    for (const auto &F : *LCOModule.first.getModuleUnlocked()) {
    //      llvm::outs() << F.getName() << "\n";
    //      if (auto MF = MMI->getMachineFunction(F)) {
    //        MF->print(llvm::outs());
    //      }
    //    }
    return llvm::Error::success();
  }
  return llvm::Error::success();
}

llvm::Error CodeGenerator::processInstModuleIntrinsicsAtIRLevel(
    llvm::Module &Module, const llvm::GCNTargetMachine &TM,
    llvm::StringMap<IntrinsicIRLoweringInfo> &InlineAsmMIRMap) {
  int NextIntrinsicIdx = 1;
  llvm::outs() << "Before mods\n";
  Module.print(llvm::outs(), nullptr);
  for (auto &F : llvm::make_early_inc_range(Module.functions())) {
    if (F.hasFnAttribute(LUTHIER_INTRINSIC_ATTRIBUTE)) {
      // Find the processor for this intrinsic
      auto DemangledFuncName =
          CodeGenerator::getDemangledIntrinsicName(F.getName());
      LUTHIER_RETURN_ON_ERROR(DemangledFuncName.takeError());
      // Ensure the processor is indeed registered with the Code Generator
      auto It = IntrinsicsProcessors.find(*DemangledFuncName);
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(It != IntrinsicsProcessors.end()));
      // Iterate over all users of the intrinsic
      for (auto *User : F.users()) {
        // Ensure the user is a Call instruction; Anything other usage is
        // illegal
        auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User);
        LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CallInst != nullptr));
        auto IRLoweringInfo = It->second.IRProcessor(F, *CallInst, TM);
        LUTHIER_RETURN_ON_ERROR(IRLoweringInfo.takeError());
        IRLoweringInfo->setIntrinsicName(*DemangledFuncName);
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
        AsmCallInst->print(llvm::outs());
        Module.print(llvm::outs(), nullptr);
        AsmCallInst->insertBefore(*CallInst->getParent(),
                                  CallInst->getIterator());
        CallInst->replaceAllUsesWith(AsmCallInst);
        // transfer debug info from the original invoke to the inline assembly
        AsmCallInst->setDebugLoc(CallInst->getDebugLoc());
        CallInst->eraseFromParent();
        InlineAsmMIRMap.insert(
            {llvm::to_string(NextIntrinsicIdx), std::move(*IRLoweringInfo)});
        NextIntrinsicIdx++;
      }
      F.dropAllReferences();
      F.eraseFromParent();
      llvm::outs() << "After mods after inserting\n";
      Module.print(llvm::outs(), nullptr);
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
    // Load the bitcode of the instrumentation module into the LR's context
    auto IModuleTS = Task.getModule().readBitcodeIntoContext(LR.getContext());
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
    IModule.print(llvm::outs(), nullptr);
    // Run the code gen pipeline, while enforcing the stack and register
    // constraints
    auto [MMIWP, PM] = runCodeGenPipeline(IModule, &TM, MIToHookFuncMap,
                                          DummyAsmToIRLoweringInfoMap);
    LR.managePassManagerLifetime(std::move(PM));
    for (const auto &F : IModule) {
      llvm::outs() << "AFTER MOVE:\n";
      llvm::outs() << MMIWP->getMMI().getModule() << "\n";
      //        if (auto MF = MMI->getMachineFunction(F)) {
      //          MF->print(llvm::outs());
      //        }
    }
    // Finally, patch in the generated machine code into the lifted
    // representation
    LUTHIER_RETURN_ON_ERROR(
        patchLiftedRepresentation(IModule, MMIWP->getMMI(), *LCOModule.first,
                                  *LCOModule.second, MIToHookFuncMap));
    for (const auto &F : *LCOModule.first) {
      llvm::outs() << "Function name inside: " << F.getName() << "\n";
      llvm::outs() << LCOModule.second->getModule() << "\n";
      if (auto MF = LCOModule.second->getMachineFunction(F)) {
        MF->print(llvm::outs());
      }
    }
    return llvm::Error::success();
  };
  return llvm::Error::success();
}

llvm::Error CodeGenerator::instrument(
    const LiftedRepresentation &LR,
    llvm::function_ref<llvm::Error(InstrumentationTask &,
                                   LiftedRepresentation &)>
        Mutator,
    llvm::DenseMap<hsa::LoadedCodeObject, llvm::SmallVector<uint8_t>>
        &CompiledCodeObjects,
    llvm::DenseMap<hsa::LoadedCodeObject, llvm::SmallVector<char>>
        *AssemblyFiles) {
  // Clone the Lifted Representation
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(
      std::unique_ptr<LiftedRepresentation>, ClonedLR,
      CodeLifter::instance().cloneRepresentation(LR));
  // Run the mutator function on the Lifted Representation and populate the
  // instrumentation task
  InstrumentationTask IT("");
  LUTHIER_RETURN_ON_ERROR(Mutator(IT, *ClonedLR));
  // Insert the hooks inside the Lifted Representation
  LUTHIER_RETURN_ON_ERROR(insertHooks(*ClonedLR, IT));
  // Generate the relocatable objects
  llvm::DenseMap<hsa::LoadedCodeObject, llvm::SmallVector<char>>
      CompiledRelocatables;
  LUTHIER_RETURN_ON_ERROR(
      printAssembly(*ClonedLR, CompiledRelocatables, AssemblyFiles));
  // Link the executables
  for (const auto &[LCO, Reloc] : CompiledRelocatables) {
    if (!CompiledCodeObjects.contains(LCO))
      CompiledCodeObjects.insert({LCO, {}});
    LUTHIER_RETURN_ON_ERROR(compileRelocatableToExecutable(
        llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Reloc.data()),
                                Reloc.size()),
        llvm::cantFail(LCO.getISA()), CompiledCodeObjects[LCO]));
  }
  return llvm::Error::success();
}
llvm::Expected<std::string>
CodeGenerator::getDemangledIntrinsicName(llvm::StringRef MangledIntrinsicName) {
  // Get the name of the function, without its template arguments
  llvm::ItaniumPartialDemangler Demangler;
  // Ensure successful partial demangle operation
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
      !Demangler.partialDemangle(MangledIntrinsicName.data())));
  // Output string
  std::string Out;
  // Output string's ostream
  llvm::raw_string_ostream OS(Out);

  size_t BufferSize;
  llvm::StringRef FuncNamespace(
      Demangler.getFunctionDeclContextName(nullptr, &BufferSize),
      BufferSize - 1);
  OS << FuncNamespace;
  OS << "::";
  llvm::StringRef FuncNameBase(
      Demangler.getFunctionBaseName(nullptr, &BufferSize), BufferSize - 1);
  OS << FuncNameBase;
  return Out;
}

char ReserveLiveRegs::ID = 0;

ReserveLiveRegs::ReserveLiveRegs(
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &MIToHookFuncMap)
    : llvm::MachineFunctionPass(ID) {

  // Calculate the Live-ins at each instruction for before MI hooks
  for (const auto &[MI, HookKernel] : MIToHookFuncMap) {
    (void)HookToInsertionPointMap.insert({HookKernel, MI});
    auto *MBB = MI->getParent();
    auto *MF = MBB->getParent();
    (void)HookLiveRegs.insert(
        {HookKernel, std::make_unique<llvm::LivePhysRegs>(
                         *MF->getSubtarget().getRegisterInfo())});
    auto &LiveRegs = *HookLiveRegs[HookKernel];
    LiveRegs.addLiveOuts(*MBB);
    for (auto I = MBB->rbegin(); I != MI->getReverseIterator(); I++) {
      LiveRegs.stepBackward(*I);
    }
    // Before MI hooks need to remove the defs of the MI itself
    LiveRegs.stepBackward(*MI);
    LLVM_DEBUG(
        llvm::dbgs() << "Live regs at instruction " << MI << ": \n";
        auto *TRI = MF->getSubtarget().getRegisterInfo();
        for (const auto &Reg
             : LiveRegs) { llvm::dbgs() << TRI->getRegAsmName(Reg) << "\n"; });
  }
}
bool ReserveLiveRegs::runOnMachineFunction(MachineFunction &MF) {
  auto *F = &MF.getFunction();
  auto &MRI = MF.getRegInfo();
  auto *TRI = MF.getSubtarget().getRegisterInfo();
  // Find the entry block of the function, and mark the live-ins
  const auto &LivePhysRegs = *HookLiveRegs.at(F);
  llvm::addLiveIns(MF.front(), LivePhysRegs);

  // Find the return blocks and create dummy uses at the end of them
  // Mark the dummy uses with MC symbols to keep track of them
  // Check if s[0:1] is included in the LivePhysRegs; If not, define it
  bool S01IsLive = LivePhysRegs.contains(AMDGPU::SGPR0_SGPR1);
  // Check if v0 is included in the LivePhysRegs; If not, define it
  bool V0IsLive = LivePhysRegs.contains(AMDGPU::VGPR0);
  // MCInstrInfo for building instruction opcodes
  auto MCII = MF.getTarget().getMCInstrInfo();
  for (auto &MBB : MF) {
    if (MBB.isReturnBlock()) {
      // If s[0:1] is not live, define it
      if (!S01IsLive) {
        // s0 = S_MOV_B32 0
        llvm::BuildMI(MBB, MBB.back(), llvm::DebugLoc(),
                      MCII->get(AMDGPU::S_MOV_B32))
            .addDef(llvm::AMDGPU::SGPR0)
            .addImm(0)
            ->setPreInstrSymbol(MF,
                                MF.getContext().getOrCreateSymbol("use-instr"));
        // s1 = S_MOV_B32 0
        llvm::BuildMI(MBB, MBB.back(), llvm::DebugLoc(),
                      MCII->get(AMDGPU::S_MOV_B32))
            .addDef(llvm::AMDGPU::SGPR1)
            .addImm(0)
            ->setPreInstrSymbol(MF,
                                MF.getContext().getOrCreateSymbol("use-instr"));
      }
      if (!V0IsLive) {
        // v0 = V_MOV_B32 0
        llvm::BuildMI(MBB, MBB.back(), llvm::DebugLoc(),
                      MCII->get(AMDGPU::V_MOV_B32_e32))
            .addDef(llvm::AMDGPU::VGPR0)
            .addImm(0)
            ->setPreInstrSymbol(MF,
                                MF.getContext().getOrCreateSymbol("use-instr"));
      }
      for (const auto &[PhysLiveInReg, LaneMask] : MF.front().liveins()) {
        auto PhysRegClass = TRI->getPhysRegBaseClass(PhysLiveInReg);
        auto BitWidth = llvm::AMDGPU::getRegBitWidth(*PhysRegClass);
        if (SIRegisterInfo::isSGPRClass(PhysRegClass)) {
          if (BitWidth == 32) {
            // s0 = S_ADD_U32 s0, $LiveIn
            // S_STORE_DWORDX2_IMM s[0:1], s[0:1], 0, 0
            //            llvm::BuildMI(MBB, MBB.back(), llvm::DebugLoc(),
            //                          MCII->get(AMDGPU::S_ADD_U32))
            //                .addReg(llvm::AMDGPU::SGPR0)
            //                .addReg(llvm::AMDGPU::SGPR0)
            //                .addReg(PhysLiveInReg)
            //                ->setPreInstrSymbol(
            //                    MF,
            //                    MF.getContext().getOrCreateSymbol("use-instr"));
            llvm::BuildMI(MBB, MBB.back(), llvm::DebugLoc(),
                          MCII->get(AMDGPU::S_STORE_DWORD_IMM))
                .addReg(PhysLiveInReg)
                .addReg(llvm::AMDGPU::SGPR0_SGPR1)
                .addImm(0)
                .addImm(0)
                ->setPreInstrSymbol(
                    MF, MF.getContext().getOrCreateSymbol("use-instr"));
          } else if (BitWidth == 64) {
            // S_STORE_DWORDX2_IMM s[0:1] $LiveIn, 0, 0
            llvm::BuildMI(MBB, MBB.back(), llvm::DebugLoc(),
                          MCII->get(AMDGPU::S_STORE_DWORDX2_IMM))
                .addReg(llvm::AMDGPU::SGPR0_SGPR1)
                .addReg(PhysLiveInReg)
                .addImm(0)
                .addImm(0)
                ->setPreInstrSymbol(
                    MF, MF.getContext().getOrCreateSymbol("use-instr"));
          } else
            llvm_unreachable("not implemented");
        }
      }
    }
  }

  return true;
}
void ReserveLiveRegs::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreservedID(llvm::MachineLoopInfoID);
  AU.addPreserved<llvm::SlotIndexesWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
};

char StackFrameOffset::ID = 0;

StackFrameOffset::StackFrameOffset(
    const LiftedRepresentation &LR,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &BeforeMIHooks,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &AfterMIHooks)
    : MachineFunctionPass(ID) {
  for (const auto &[Func, LLVMFunc] : LR.functions()) {
  }
}

bool StackFrameOffset::runOnMachineFunction(MachineFunction &MF) {
  auto &MFI = MF.getFrameInfo();

  llvm::outs() << "machine function " << MF.getName() << "\n"
               << "\tStack Size of: " << MFI.getStackSize() << "\n"
               << "\tContains " << MFI.getNumObjects() << " Stack Objects\n";

  for (int SOIdx = 0; SOIdx < MFI.getNumObjects(); ++SOIdx) {
    llvm::outs() << " - Stack Object Num " << SOIdx << "\n"
                 << "   Stack ID:             " << MFI.getStackID(SOIdx) << "\n"
                 << "   Stack object Size:    " << MFI.getObjectSize(SOIdx)
                 << "\n";
    // Add to Stack Frame object offset
    auto NewOffset =
        MFI.getObjectOffset(SOIdx) +
        FrameOffset.at(&MF.getFunction()); // value to add: amount of stack
                                           // the original app is using
    MFI.setObjectOffset(SOIdx, NewOffset);
    llvm::outs() << "   Stack Pointer Offset: " << NewOffset << "\n";
  }
  return false;
}

char InstBundler::ID = 0;
bool InstBundler::runOnMachineFunction(MachineFunction &MF) {
  for (auto &MBB : MF) {
    auto Bundler = llvm::MIBundleBuilder(MBB, MBB.begin(), MBB.end());
    llvm::finalizeBundle(MBB, Bundler.begin());
  }
  return false;
}

char IntrinsicMIRLoweringPass::ID = 0;

bool IntrinsicMIRLoweringPass::runOnMachineFunction(MachineFunction &MF) {
  bool Changed{false};
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
        auto MIBuilder = [&](int Opcode) {
          return llvm::BuildMI(MBB, MI, llvm::MIMetadata(MI), TII->get(Opcode));
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
      }
    }
  }
  return Changed;
}
} // namespace luthier
