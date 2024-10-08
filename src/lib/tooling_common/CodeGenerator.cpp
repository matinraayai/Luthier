//===-- CodeGenerator.cpp - Luthier Code Generator ------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Luthier's code generator.
//===----------------------------------------------------------------------===//
#include "tooling_common/CodeGenerator.hpp"

#include <memory>

#include "tooling_common/TPCOverrides.hpp"

#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SaveAndRestore.h"
#include <AMDGPUResourceUsageAnalysis.h>
#include <AMDGPUTargetMachine.h>
#include <amd_comgr/amd_comgr.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "AMDGPUGenInstrInfo.inc"
#include "common/Error.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include "hsa/ISA.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "tooling_common/CodeLifter.hpp"
#include "tooling_common/PreKernelEmitter.hpp"
#include "tooling_common/ToolExecutableLoader.hpp"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/MCSymbolELF.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/LRRegisterLiveness.h>
#include <tooling_common/IntrinsicMIRLoweringPass.hpp>
#include <tooling_common/PhysicalRegAccessVirtualizationPass.hpp>

#include <ranges>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-code-generator"

namespace luthier {

/// All injected payload <tt>llvm::Function</tt>s must have this attribute
static constexpr const char *InjectedPayloadAttribute =
    "luthier_injected_payload";

template <> CodeGenerator *Singleton<CodeGenerator>::Instance{nullptr};

llvm::Error CodeGenerator::linkRelocatableToExecutable(
    const llvm::ArrayRef<char> &Code, const hsa::ISA &ISA,
    llvm::SmallVectorImpl<uint8_t> &Out) {
  llvm::TimeTraceScope Scope("Comgr Executable Linking");
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

llvm::Expected<llvm::Function &>
CodeGenerator::generateInjectedPayloadForApplicationMI(
    llvm::Module &IModule,
    llvm::ArrayRef<InstrumentationTask::hook_invocation_descriptor>
        HookInvocationSpecs,
    const llvm::MachineInstr &ApplicationMI) {
  auto &LLVMContext = IModule.getContext();
  // Create an empty function to house the code injected before the
  // target application MI
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(LLVMContext), {}, false);
  // Name of the injected payload function will contain the application MI
  // it will be injected before, as well as the number of the MI's MBB
  std::string IFuncName;
  llvm::raw_string_ostream NameOS(IFuncName);
  NameOS << "MI: " << ApplicationMI
         << ", MMB ID: " << ApplicationMI.getParent()->getNumber();
  auto *InjectedPayload = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::ExternalLinkage, IFuncName, IModule);
  // The instrumentation function has the C-Calling convention
  InjectedPayload->setCallingConv(llvm::CallingConv::C);
  // Prevent emission of the prologue/epilogue code, but still lower the stack
  // operands
  InjectedPayload->addFnAttr(llvm::Attribute::Naked);
  // Set an attribute indicating that this is the top-level function for an
  // injected payload
  InjectedPayload->addFnAttr(InjectedPayloadAttribute);

  LLVM_DEBUG(

      llvm::dbgs() << "Generating instrumentation function for MI: "
                   << ApplicationMI << ", MBB: "
                   << ApplicationMI.getParent()->getNumber() << "\n";
      llvm::dbgs()
      << "Number of hooks to be called in instrumentation function: "
      << HookInvocationSpecs.size() << "\n"

  );

  // Create an empty basic block to fill in with calls to hooks in the order
  // specified by the spec
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(IModule.getContext(), "", InjectedPayload);
  llvm::IRBuilder<> Builder(BB);
  for (const auto &HookInvSpec : HookInvocationSpecs) {
    // Find the hook function inside the instrumentation module
    auto HookFunc = IModule.getFunction(HookInvSpec.HookName);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(HookFunc != nullptr));
    // Construct the operands of the hook call
    llvm::SmallVector<llvm::Value *, 4> Operands;
    for (const auto &[Idx, Op] : llvm::enumerate(HookInvSpec.Args)) {
      if (holds_alternative<llvm::MCRegister>(Op)) {
        // Create a call to the read reg intrinsic to load the MC register
        // into a value, then pass it to the hook
        auto ReadRegVal = insertCallToIntrinsic(
            *HookFunc->getParent(), Builder, "luthier::readReg",
            *HookFunc->getArg(Idx)->getType(), std::get<llvm::MCRegister>(Op).id());
        Operands.push_back(ReadRegVal);
      } else {
        // Otherwise it's a constant, we can just pass it directly
        Operands.push_back(std::get<llvm::Constant *>(Op));
      }
    }
    // Finally, create a call to the hook
    (void)Builder.CreateCall(HookFunc, Operands);
  }
  // Put a ret void at the end of the instrumentation function to indicate
  // nothing is returned
  (void)Builder.CreateRetVoid();
  return *InjectedPayload;
}

/// Runs the IR optimization pipeline on the \p IModule
/// \param IModule The instrumentation module being optimized
/// \param TM The \c llvm::TargetMachine of the \p IModule used to
/// schedule passes related to the AMDGPU backend
static void runIROptimizationPipelineOnIModule(llvm::Module &IModule,
                                               llvm::GCNTargetMachine &TM) {
  llvm::TimeTraceScope Scope("Instrumentation Module IR Optimization");
  // Create the analysis managers.
  // These must be declared in this order so that they are destroyed in
  // the correct order due to inter-analysis-manager references.
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  llvm::PassBuilder PB(&TM);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create the pass manager.
  llvm::ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // Run the scheduled passes
  MPM.run(IModule, MAM);
  // TODO: Is there a way to return an \c llvm::Error in case the pipeline
  // fails at any point?
}

void calculatePhysicalRegsAccessedByHooksNotToBeClobbered(
    const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
        &HookFuncToMIMap,
    const llvm::ArrayRef<std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
        ToBeLoweredIntrinsics,
    const LRRegisterLiveness &LRLiveRegs, const llvm::GCNTargetMachine &TM,
    llvm::LivePhysRegs &Out) {
  // This function does not require the hook insertion tasks to be passed,
  // since passing arguments is done using luthier::readReg
  for (const auto &[Func, LoweringInfo] : ToBeLoweredIntrinsics) {
    if (Func->hasFnAttribute(InjectedPayloadAttribute)) {
      for (const auto &UsedPhysReg :
           LoweringInfo.getAccessedPhysicalRegisters()) {
        auto *MIInsertionPoint = HookFuncToMIMap.at(Func);
        if (!LRLiveRegs.getLiveInPhysRegsOfMachineInstr(*MIInsertionPoint)
                 ->contains(UsedPhysReg)) {
          if (Out.empty())
            Out.init(*TM.getSubtargetImpl(*Func)->getRegisterInfo());
          Out.addReg(UsedPhysReg);
        }
      }
    }
  }
}

std::tuple<llvm::MachineModuleInfoWrapperPass *,
           std::unique_ptr<llvm::legacy::PassManager>,
           PreKernelEmissionDescriptor>
CodeGenerator::runCodeGenPipeline(
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &MIToHookFuncMap,
    const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
        &HookFuncToMIMap,
    const llvm::SmallVectorImpl<
        std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
        &ToBeLoweredIntrinsics,
    bool DisableVerify, const LiftedRepresentation &LR,
    const hsa::LoadedCodeObject &LCO, llvm::GCNTargetMachine *TM,
    llvm::Module &M) {
  // Calculate the live registers at every point of the LR
  LRRegisterLiveness LRLiveRegs(LR);
  {
    llvm::TimeTraceScope Scope("Liveness Analysis Computation");
    LRLiveRegs.recomputeLiveIns();
  }

  // Analyze the call graph
  auto CG = LRCallGraph::analyse(LR);
  LUTHIER_REPORT_FATAL_ON_ERROR(CG.takeError());

  // Calculate all the physical registers accessed by the instrumentation
  // logic that are not live at hook points
  llvm::LivePhysRegs PhysicalRegisterNotToClobber;
  calculatePhysicalRegsAccessedByHooksNotToBeClobbered(
      HookFuncToMIMap, ToBeLoweredIntrinsics, LRLiveRegs, *TM,
      PhysicalRegisterNotToClobber);
  // Store specs of the pre-kernel here
  PreKernelEmissionDescriptor PKInfo;
  auto SVLocations = LRStateValueLocations::create(LR, LCO, MIToHookFuncMap,
                                                   PhysicalRegisterNotToClobber,
                                                   LRLiveRegs, PKInfo);
  LUTHIER_REPORT_FATAL_ON_ERROR(SVLocations.takeError());

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
  auto *PhysRegAccessPass = new PhysicalRegAccessVirtualizationPass(
      LR, PhysicalRegisterNotToClobber, **CG, **SVLocations, HookFuncToMIMap,
      ToBeLoweredIntrinsics, LRLiveRegs);
  PM->add(PhysRegAccessPass);
  PM->add(new IntrinsicMIRLoweringPass(ToBeLoweredIntrinsics,
                                       IntrinsicsProcessors));
  auto *PEPass =
      new HookPEIPass(LR, **SVLocations, *PhysRegAccessPass, HookFuncToMIMap,
                      LRLiveRegs, PhysicalRegisterNotToClobber, PKInfo);

  luthier::addMachinePassesToTPC(*TPC, *PEPass, *PM);
  //  PM->add(PEPass);
  TPC->setInitialized();

  PM->run(M);

  return {IMMIWP, std::move(PM), PKInfo};
}

llvm::Error patchLiftedRepresentation(
    const llvm::Module &IModule, const llvm::MachineModuleInfo &IMMI,
    llvm::Module &LRModule, llvm::MachineModuleInfo &LRMMI,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &MIToHookFuncMap,
    PreKernelEmissionDescriptor &PKInfo) {
  llvm::TimeTraceScope Scope("Lifted Representation Patching");
  // A mapping between Global Variables in the instrumentation module and
  // their corresponding Global Variables in the instrumented code
  llvm::ValueToValueMapTy VMap;
  // Clone the instrumentation module Global Variables into the instrumented
  // code
  for (const auto &GV : IModule.globals()) {
    auto *NewGV = new llvm::GlobalVariable(
        LRModule, GV.getValueType(), GV.isConstant(), GV.getLinkage(), nullptr,
        GV.getName(), nullptr, GV.getThreadLocalMode(),
        GV.getType()->getAddressSpace());
    NewGV->copyAttributesFrom(&GV);
    VMap[&GV] = NewGV;
  }
  for (const auto &[InsertionPointMI, HookF] : MIToHookFuncMap) {
    // A mapping between a machine basic block in the instrumentation MMI
    // and its destination in the patched instrumented code
    llvm::DenseMap<const llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>
        MBBMap;

    const auto &HookMF = *IMMI.getMachineFunction(*HookF);
    auto &InsertionPointMBB = *InsertionPointMI->getParent();
    auto &ToBeInstrumentedMF = *InsertionPointMBB.getParent();

    auto &HookMFFrameInfo = HookMF.getFrameInfo();

    if (HookMFFrameInfo.hasStackObjects()) {
      ToBeInstrumentedMF.getFrameInfo().setStackSize(
          ToBeInstrumentedMF.getFrameInfo().getStackSize() +
          HookMFFrameInfo.getStackSize());
    }

    if (HookMFFrameInfo.hasStackObjects()) {
      // Clone the frame objects
      auto &ToBeInstrumentedMFI = ToBeInstrumentedMF.getFrameInfo();

      auto CopyObjectProperties = [](llvm::MachineFrameInfo &DstMFI,
                                     const llvm::MachineFrameInfo &SrcMFI,
                                     int FI) {
        if (SrcMFI.isStatepointSpillSlotObjectIndex(FI))
          DstMFI.markAsStatepointSpillSlotObjectIndex(FI);
        DstMFI.setObjectSSPLayout(FI, SrcMFI.getObjectSSPLayout(FI));
        DstMFI.setObjectZExt(FI, SrcMFI.isObjectZExt(FI));
        DstMFI.setObjectSExt(FI, SrcMFI.isObjectSExt(FI));
      };

      for (int i = 0, e = HookMFFrameInfo.getNumObjects() -
                          HookMFFrameInfo.getNumFixedObjects();
           i != e; ++i) {
        int NewFI;

        assert(!HookMFFrameInfo.isFixedObjectIndex(i));
        if (!HookMFFrameInfo.isDeadObjectIndex(i)) {
          if (HookMFFrameInfo.isVariableSizedObjectIndex(i)) {
            NewFI = ToBeInstrumentedMFI.CreateVariableSizedObject(
                HookMFFrameInfo.getObjectAlign(i),
                HookMFFrameInfo.getObjectAllocation(i));
          } else {
            NewFI = ToBeInstrumentedMFI.CreateStackObject(
                HookMFFrameInfo.getObjectSize(i),
                HookMFFrameInfo.getObjectAlign(i),
                HookMFFrameInfo.isSpillSlotObjectIndex(i),
                HookMFFrameInfo.getObjectAllocation(i),
                HookMFFrameInfo.getStackID(i));
            ToBeInstrumentedMFI.setObjectOffset(
                NewFI, HookMFFrameInfo.getObjectOffset(i));
          }
          CopyObjectProperties(ToBeInstrumentedMFI, HookMFFrameInfo, i);

          (void)NewFI;
          assert(i == NewFI && "expected to keep stable frame index numbering");
        }
      }

      // Copy the fixed frame objects backwards to preserve frame index numbers,
      // since CreateFixedObject uses front insertion.
      for (int i = -1; i >= (int)-HookMFFrameInfo.getNumFixedObjects(); --i) {
        assert(HookMFFrameInfo.isFixedObjectIndex(i));
        if (!HookMFFrameInfo.isDeadObjectIndex(i)) {
          int NewFI = ToBeInstrumentedMFI.CreateFixedObject(
              HookMFFrameInfo.getObjectSize(i),
              HookMFFrameInfo.getObjectOffset(i),
              HookMFFrameInfo.isImmutableObjectIndex(i),
              HookMFFrameInfo.isAliasedObjectIndex(i));
          CopyObjectProperties(ToBeInstrumentedMFI, HookMFFrameInfo, i);

          (void)NewFI;
          assert(i == NewFI && "expected to keep stable frame index numbering");
        }
      }
    }

    // Clone the MBBs

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
        //        auto *DstMI = ToBeInstrumentedMF.CreateMachineInstr(
        //            TII->get(llvm::AMDGPU::S_WAITCNT), llvm::DebugLoc(),
        //            /*NoImplicit=*/true);
        //        DstMBB->insert(InsertionPoint, DstMI);
        //        DstMI->addOperand(llvm::MachineOperand::CreateImm(0));
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
            TII->get(llvm::AMDGPU::S_WAITCNT), llvm::DebugLoc(),
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
  llvm::TimeTraceScope scope("LLVM Assembly Printing");
  // Argument error checking
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(FileType != llvm::CodeGenFileType::Null));
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

llvm::Error CodeGenerator::processIModuleIntrinsicUsersAtIRLevel(
    llvm::Module &IModule, const llvm::GCNTargetMachine &TM,
    llvm::SmallVectorImpl<std::pair<llvm::Function *, IntrinsicIRLoweringInfo>>
        &InlineAsmMIRMap) {
  int NextIntrinsicIdx = 0;
  for (auto &F : llvm::make_early_inc_range(IModule.functions())) {
    if (F.hasFnAttribute(LUTHIER_INTRINSIC_ATTRIBUTE)) {
      // Find the processor for this intrinsic
      auto IntrinsicName =
          F.getFnAttribute(LUTHIER_INTRINSIC_ATTRIBUTE).getValueAsString();
      // Ensure the processor is indeed registered with the Code Generator
      auto It = IntrinsicsProcessors.find(IntrinsicName);
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ASSERTION(It != IntrinsicsProcessors.end()));

      LLVM_DEBUG(

          llvm::dbgs() << "Intrinsic function being processed: " << F << "\n";
          llvm::dbgs() << "Base name of the intrinsic: " << IntrinsicName
                       << "\n";
          llvm::dbgs() << "Num uses of the intrinsic function : "
                       << F.getNumUses() << "\n"

      );

      // Iterate over all users of the intrinsic
      for (auto *User : llvm::make_early_inc_range(F.users())) {
        LLVM_DEBUG(llvm::dbgs() << "User being processed: \n";
                   User->print(llvm::dbgs()););
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
        AsmCallInst->insertBefore(*CallInst->getParent(),
                                  CallInst->getIterator());
        CallInst->replaceAllUsesWith(AsmCallInst);
        // transfer debug info from the original invoke to the inline assembly
        AsmCallInst->setDebugLoc(CallInst->getDebugLoc());
        CallInst->eraseFromParent();
        auto *ParentFunction = AsmCallInst->getParent()->getParent();
        // If the function using the intrinsic is not a hook (i.e a device
        // function called from a hook), check if it's not requesting access to
        // a physical register
        if (!ParentFunction->hasFnAttribute(InjectedPayloadAttribute)) {
          LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
              IRLoweringInfo->getAccessedPhysicalRegisters().empty()));
          LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
              IRLoweringInfo->getRequestedKernelArgument().empty()));
        }
        LLVM_DEBUG(llvm::dbgs()
                       << "Use's inline assembly after IR processing: \n";
                   AsmCallInst->print(llvm::dbgs()););
        InlineAsmMIRMap.emplace_back(std::make_pair(
            AsmCallInst->getFunction(), std::move(*IRLoweringInfo)));
        NextIntrinsicIdx++;
      }
      F.dropAllReferences();
      F.eraseFromParent();
    }
  }
  return llvm::Error::success();
}

llvm::Error
CodeGenerator::applyInstrumentationTask(const InstrumentationTask &Task,
                                        LiftedRepresentation &LR) {
  // Early exit if no hooks are to be inserted into the LR
  if (Task.getHookInsertionTasks().empty())
    return llvm::Error::success();
  // Acquire the Lifted Representation's lock
  auto Lock = LR.getLock();
  // Each LCO will get its own copy of the instrumented module
  for (auto &[LCO, LCOModule] : LR) {
    hsa::LoadedCodeObject LCOWrapper(LCO);
    auto Agent = LCOWrapper.getAgent();
    LUTHIER_RETURN_ON_ERROR(Agent.takeError());
    // Load the bitcode of the instrumentation module into the
    // Lifted Representation's context
    auto *ProfilingEntry =
        llvm::timeTraceProfilerBegin("LLVM Bitcode Loading", "");
    auto IModuleTS =
        Task.getModule().readBitcodeIntoContext(LR.getContext(), *Agent);
    llvm::timeTraceProfilerEnd(ProfilingEntry);
    LUTHIER_RETURN_ON_ERROR(IModuleTS.takeError());
    auto &IModule = *IModuleTS->getModuleUnlocked();
    LLVM_DEBUG(llvm::dbgs() << "Instrumentation Module Contents: \n";
               IModule.print(llvm::dbgs(), nullptr););
    auto &TM = LR.getTargetMachine<llvm::GCNTargetMachine>();
    // Now that everything has been created we can start inserting hooks
    LLVM_DEBUG(llvm::dbgs() << "Number of MIs to get hooks: "
                            << Task.getHookInsertionTasks().size() << "\n");
    // A map to keep track of the injected payload functions inside the
    // instrumentation module given the MI they will be patched into
    llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        AppMIToInjectedPayloadMap;
    // An inverse mapping of the above DenseMap, relating each injected payload
    // function to its target MI in the application
    llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
        InjectedPayloadToAppMIMap;
    // Generate and populate the injected payload functions in the
    // instrumentation module and keep track of them inside the map
    ProfilingEntry = llvm::timeTraceProfilerBegin(
        "Instrumentation Module IR Generation", "");
    for (const auto &[ApplicationMI, HookSpecs] : Task.getHookInsertionTasks()) {
      // Generate the Hooks for each MI
      auto HookFunc =
          generateInjectedPayloadForApplicationMI(IModule, HookSpecs, *ApplicationMI);
      LUTHIER_RETURN_ON_ERROR(HookFunc.takeError());
      AppMIToInjectedPayloadMap.insert({ApplicationMI, &(*HookFunc)});
      InjectedPayloadToAppMIMap.insert({&(*HookFunc), ApplicationMI});
    }
    llvm::timeTraceProfilerEnd(ProfilingEntry);
    LLVM_DEBUG(llvm::dbgs()
               << "Instrumentation Module After Hooks Inserted:\n");
    LLVM_DEBUG(IModule.print(llvm::dbgs(), nullptr));
    // With the Hook IR generated, we put it through the normal IR optimization
    // pipeline
    runIROptimizationPipelineOnIModule(IModule, TM);

    LLVM_DEBUG(

        llvm::dbgs() << "Instrumentation Module after IR optimization:\n";
        IModule.print(llvm::dbgs(), nullptr)

    );

    ProfilingEntry = llvm::timeTraceProfilerBegin(
        "Instrumentation Module MIR CodeGen Optimization", "");
    // Replace all Luthier intrinsics with dummy inline assembly and correct
    // arguments
    llvm::SmallVector<std::pair<llvm::Function *, IntrinsicIRLoweringInfo>, 4>
        DummyAsmToIRLoweringInfoMap;
    LUTHIER_RETURN_ON_ERROR(processIModuleIntrinsicUsersAtIRLevel(
        IModule, TM, DummyAsmToIRLoweringInfoMap));
    // Run the code gen pipeline, while enforcing the stack and register
    // constraints
    auto [MMIWP, PM, PKInfo] = runCodeGenPipeline(
        AppMIToInjectedPayloadMap, InjectedPayloadToAppMIMap, DummyAsmToIRLoweringInfoMap, false,
        LR, hsa::LoadedCodeObject(LCO), &TM, IModule);
    LLVM_DEBUG(llvm::dbgs() << "The instrumentation Machine Code before being "
                               "patched into the Lifted Representation:\n";
               llvm::dbgs() << "Location of instr. MMI in memory: "
                            << &MMIWP->getMMI() << "\n";
               llvm::dbgs() << "Location of instr. Module in memory: "
                            << MMIWP->getMMI().getModule() << "\n";
               for (const auto &F : IModule) {
                 if (auto MF = MMIWP->getMMI().getMachineFunction(F)) {
                   MF->print(llvm::dbgs());
                 }
               });
    llvm::timeTraceProfilerEnd(ProfilingEntry);

    // Finally, patch in the generated machine code into the lifted
    // representation
    LUTHIER_RETURN_ON_ERROR(
        patchLiftedRepresentation(IModule, MMIWP->getMMI(), *LCOModule.first,
                                  *LCOModule.second,
        AppMIToInjectedPayloadMap, PKInfo));

    // TODO: remove this once the move constructor for MMI makes it to master
    delete PM.release();
  };
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<LiftedRepresentation>> CodeGenerator::instrument(
    const LiftedRepresentation &LR,
    llvm::function_ref<llvm::Error(InstrumentationTask &,
                                   LiftedRepresentation &)>
        Mutator) {
  // Acquire the context lock for thread-safety
  auto Lock = LR.getLock();
  std::unique_ptr<LiftedRepresentation> ClonedLR;
  // Clone the Lifted Representation
  LUTHIER_RETURN_ON_ERROR(
      CodeLifter::instance().cloneRepresentation(LR).moveInto(ClonedLR));
  // Create an instrumentation task to keep track of the hooks called before
  // each MI of the application
  InstrumentationTask IT(*ClonedLR);
  // Run the mutator function on the Lifted Representation and populate the
  // instrumentation task
  LUTHIER_RETURN_ON_ERROR(Mutator(IT, *ClonedLR));
  // Apply the instrumentation task to the Lifted Representation
  LUTHIER_RETURN_ON_ERROR(applyInstrumentationTask(IT, *ClonedLR));

  LLVM_DEBUG(

      llvm::dbgs() << "Final instrumented Lifted Representation Code:\n";
      for (const auto &[LCO, LCOModule] : *ClonedLR) {
        for (const auto &F : *LCOModule.first) {
          llvm::dbgs() << "Function name in the LLVM Module: " << F.getName()
                       << "\n";
          if (auto MF = LCOModule.second->getMachineFunction(F)) {
            llvm::dbgs() << "Location of the Machine Function in memory: " << MF
                         << "\n";
            MF->print(llvm::dbgs());
          }
        }
      }

  );

  return std::move(ClonedLR);
}
} // namespace luthier
