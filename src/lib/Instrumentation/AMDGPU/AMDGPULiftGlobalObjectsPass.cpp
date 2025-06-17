//===-- AMDGPULiftGlobalObjectsPass.cpp -----------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// \file
/// Implements the <tt>AMDGPULiftGlobalObjectsPass</tt> class.
//===----------------------------------------------------------------------===//
#include "luthier/Instrumentation/ELFRelocationResolverAnalysisPass.h"
#include "luthier/Instrumentation/GlobalObjectOffsetsAnalysis.h"

#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Instrumentation/AMDGPU/AMDGPULiftGlobalObjectsPass.h>
#include <luthier/Instrumentation/AMDGPU/KernelDescriptor.h>
#include <luthier/Instrumentation/AMDGPU/LiftedKernelSymbolAnalysisPass.h>
#include <luthier/Instrumentation/AMDGPU/Metadata.h>
#include <luthier/Instrumentation/GlobalObjectSymbolsAnalysis.h>
#include <luthier/Instrumentation/ObjectFileAnalysisPass.h>
#include <luthier/Object/AMDGCNObjectFile.h>

namespace luthier {

static llvm::Expected<const llvm::GlobalVariable &>
initGlobalVariable(luthier::object::AMDGCNElfSymbolRef VarSym,
                   llvm::Module &M) {
  auto &LLVMContext = M.getContext();
  llvm::Expected<llvm::StringRef> GVNameOrErr = VarSym.getName();
  LUTHIER_RETURN_ON_ERROR(GVNameOrErr.takeError());
  size_t GVSize = VarSym.getSize();
  // Lift each variable as an array of bytes, with a length of GVSize
  // We remove any initializers present in the LCO
  auto Out = new llvm::GlobalVariable(
      M, llvm::ArrayType::get(llvm::Type::getInt8Ty(LLVMContext), GVSize),
      false, llvm::GlobalValue::LinkageTypes::ExternalLinkage, nullptr,
      *GVNameOrErr);
  return *Out;
}

static llvm::Type &
processExplicitKernelArg(const amdgpu::hsamd::Kernel::Arg::Metadata &ArgMD,
                         llvm::LLVMContext &Ctx) {
  llvm::Type *ParamType = llvm::Type::getIntNTy(Ctx, ArgMD.Size * 8);
  // Used when the argument kind is global buffer or dynamic shared pointer
  unsigned int AddressSpace = ArgMD.AddressSpace.has_value()
                                  ? *ArgMD.AddressSpace
                                  : llvm::AMDGPUAS::GLOBAL_ADDRESS;
  unsigned int PointeeAlign =
      ArgMD.PointeeAlign.has_value() ? *ArgMD.PointeeAlign : 0;
  switch (ArgMD.ValueKind) {
  case amdgpu::hsamd::ValueKind::ByValue:
    break;
  case amdgpu::hsamd::ValueKind::GlobalBuffer:
    // Convert the argument to a pointer
    ParamType = llvm::PointerType::get(ParamType, AddressSpace);
    break;
  default:
    llvm_unreachable("Not implemented");
  }
  return *ParamType;
}

static void
processHiddenKernelArg(const amdgpu::hsamd::Kernel::Arg::Metadata &ArgMD,
                       llvm::Function &F, llvm::SIMachineFunctionInfo &MFI,
                       const llvm::SIRegisterInfo &TRI) {
  switch (ArgMD.ValueKind) {
  case amdgpu::hsamd::ValueKind::HiddenGlobalOffsetX:
  case amdgpu::hsamd::ValueKind::HiddenGlobalOffsetY:
  case amdgpu::hsamd::ValueKind::HiddenGlobalOffsetZ:
  case amdgpu::hsamd::ValueKind::HiddenBlockCountX:
  case amdgpu::hsamd::ValueKind::HiddenBlockCountY:
  case amdgpu::hsamd::ValueKind::HiddenBlockCountZ:
  case amdgpu::hsamd::ValueKind::HiddenRemainderX:
  case amdgpu::hsamd::ValueKind::HiddenRemainderY:
  case amdgpu::hsamd::ValueKind::HiddenRemainderZ:
  case amdgpu::hsamd::ValueKind::HiddenNone:
  case amdgpu::hsamd::ValueKind::HiddenGroupSizeX:
  case amdgpu::hsamd::ValueKind::HiddenGroupSizeY:
  case amdgpu::hsamd::ValueKind::HiddenGroupSizeZ:
  case amdgpu::hsamd::ValueKind::HiddenGridDims:
  case amdgpu::hsamd::ValueKind::HiddenPrivateBase:
  case amdgpu::hsamd::ValueKind::HiddenSharedBase:
    break;
  case amdgpu::hsamd::ValueKind::HiddenPrintfBuffer:
    F.getParent()->getOrInsertNamedMetadata("llvm.printf.fmts");
    break;
  case amdgpu::hsamd::ValueKind::HiddenHostcallBuffer:
    F.removeFnAttr("amdgpu-no-hostcall-ptr");
    break;
  case amdgpu::hsamd::ValueKind::HiddenDefaultQueue:
    F.removeFnAttr("amdgpu-no-default-queue");
    break;
  case amdgpu::hsamd::ValueKind::HiddenCompletionAction:
    F.removeFnAttr("amdgpu-no-completion-action");
    break;
  case amdgpu::hsamd::ValueKind::HiddenMultiGridSyncArg:
    F.removeFnAttr("amdgpu-no-multigrid-sync-arg");
    break;
  case amdgpu::hsamd::ValueKind::HiddenHeapV1:
    F.removeFnAttr("amdgpu-no-heap-ptr");
    break;
  case amdgpu::hsamd::ValueKind::HiddenDynamicLDSSize:
    MFI.setUsesDynamicLDS(true);
    break;
  case amdgpu::hsamd::ValueKind::HiddenQueuePtr:
    MFI.addQueuePtr(TRI);
    break;
  default:
    return;
  }
}

static llvm::Expected<llvm::MachineFunction &>
initLiftedKernelEntry(object::AMDGCNKernelDescSymbolRef KernelSym,
                      const amdgpu::hsamd::Kernel::Metadata &KernelMD,
                      llvm::Module &M, llvm::MachineModuleInfo &MMI) {
  llvm::LLVMContext &LLVMContext = M.getContext();
  // Populate the Arguments ==================================================
  llvm::StringRef SymbolName;
  LUTHIER_RETURN_ON_ERROR(KernelSym.getName().moveInto(SymbolName));

  // Kernel's return type is always void
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);

  // Create the Kernel's FunctionType with appropriate kernel Arguments
  // (if any)
  llvm::SmallVector<llvm::Type *> Params;
  unsigned int ExplicitArgsOffset = 0;
  unsigned int ImplicitArgsOffset = 0;
  if (KernelMD.Args.has_value()) {
    // Reserve the number of arguments in the Params vector
    Params.reserve(KernelMD.Args->size());
    // For now, we only rely on required argument metadata
    // This should be updated as new cases are encountered
    for (const auto &ArgMD : *KernelMD.Args) {
      if (ArgMD.ValueKind >= amdgpu::hsamd::ValueKind::HiddenArgKindBegin)
        break;
      else {
        Params.push_back(&processExplicitKernelArg(ArgMD, M.getContext()));
        if (ArgMD.Offset > ExplicitArgsOffset)
          ExplicitArgsOffset = ArgMD.Offset;
      }
    }
  }

  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, Params, false);

  auto *F =
      llvm::Function::Create(FunctionType, llvm::GlobalValue::WeakAnyLinkage,
                             SymbolName.substr(0, SymbolName.rfind(".kd")), M);
  F->setVisibility(llvm::GlobalValue::ProtectedVisibility);

  // Populate the Attributes =================================================

  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

  F->addFnAttr("uniform-work-group-size",
               KernelMD.UniformWorkgroupSize ? "true" : "false");

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated
  llvm::Expected<llvm::ArrayRef<uint8_t>> KDContentsOrErr =
      getContents(KernelSym);
  LUTHIER_RETURN_ON_ERROR(KDContentsOrErr.takeError());
  const auto &KD = *reinterpret_cast<const amdgpu::hsa::KernelDescriptor *>(
      KDContentsOrErr->data());

  F->addFnAttr("amdgpu-lds-size", llvm::to_string(KD.GroupSegmentFixedSize));
  // Kern Arg is determined via analysis usage + args set earlier
  auto Rsrc1 = KD.getRsrc1();
  auto Rsrc2 = KD.getRsrc2();
  auto KCP = KD.getKernelCodeProperties();
  if (KCP.EnableSgprDispatchId == 0) {
    F->addFnAttr("amdgpu-no-dispatch-id");
  }
  if (KCP.EnableSgprDispatchPtr == 0) {
    F->addFnAttr("amdgpu-no-dispatch-ptr");
  }
  if (KCP.EnableSgprQueuePtr == 0) {
    F->addFnAttr("amdgpu-no-queue-ptr");
  }

  F->addFnAttr("amdgpu-ieee", Rsrc1.EnableIeeeMode ? "true" : "false");
  F->addFnAttr("amdgpu-dx10-clamp", Rsrc1.EnableDx10Clamp ? "true" : "false");
  if (Rsrc2.EnableSgprWorkgroupIdX == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-x");
  }
  if (Rsrc2.EnableSgprWorkgroupIdY == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-y");
  }
  if (Rsrc2.EnableSgprWorkgroupIdZ == 0) {
    F->addFnAttr("amdgpu-no-workgroup-id-z");
  }
  switch (Rsrc2.EnableVgprWorkitemId) {
  case 0:
    F->addFnAttr("amdgpu-no-workitem-id-y");
  case 1:
    F->addFnAttr("amdgpu-no-workitem-id-z");
  case 2:
    break;
  default:
    llvm_unreachable("KD's VGPR workitem ID is not valid");
  }

  // TODO: Check the args metadata to set this correctly
  // TODO: Set the rest of the attributes
  //    luthier::outs() << "Preloaded Args: " << (*KDOnHost)->KernArgPreload <<
  //    "\n";
  //  F->addFnAttr("amdgpu-calls");
  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(M.getContext(), "", F);
  new llvm::UnreachableInst(M.getContext(), BB);

  // Populate the MFI ========================================================

  auto &MF = MMI.getOrCreateMachineFunction(*F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));
  auto &TM = MMI.getTarget();

  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(*F)->getRegisterInfo());
  auto MFI = MF.template getInfo<llvm::SIMachineFunctionInfo>();
  if (KCP.EnableSgprDispatchPtr == 1) {
    MFI->addDispatchPtr(*TRI);
  }
  if (KCP.EnableSgprPrivateSegmentBuffer == 1) {
    MFI->addPrivateSegmentBuffer(*TRI);
  }
  if (KCP.EnableSgprKernArgSegmentPtr == 1) {
    MFI->addKernargSegmentPtr(*TRI);
  }
  if (KCP.EnableSgprFlatScratchInit == 1) {
    MFI->addFlatScratchInit(*TRI);
  }
  if (Rsrc2.EnableSgprPrivateSegmentWaveByteOffset == 1) {
    MFI->addPrivateSegmentWaveByteOffset();
  }

  // Process the hidden args now that MFI and MF has been created
  if (KernelMD.Args.has_value()) {
    // Add absence of all hidden arguments; As we iterate over all the
    // hidden arguments, we get rid of them if we detect their presence
    F->addFnAttr("amdgpu-no-hostcall-ptr");
    F->addFnAttr("amdgpu-no-default-queue");
    F->addFnAttr("amdgpu-no-completion-action");
    F->addFnAttr("amdgpu-no-multigrid-sync-arg");
    F->addFnAttr("amdgpu-no-heap-ptr");
    for (const auto &ArgMD : *KernelMD.Args) {
      if (ArgMD.ValueKind >= amdgpu::hsamd::ValueKind::HiddenArgKindBegin &&
          ArgMD.ValueKind <= amdgpu::hsamd::ValueKind::HiddenArgKindEnd) {
        processHiddenKernelArg(
            ArgMD, *F, *MFI,
            *MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo());
        if (ArgMD.Offset > ImplicitArgsOffset)
          ImplicitArgsOffset = ArgMD.Offset;
      }
    }
  }

  // Number of implicit arg bytes is the difference between the last hidden
  // arg offset and the last explicit arg offset
  F->addFnAttr("amdgpu-implicitarg-num-bytes",
               llvm::to_string(ImplicitArgsOffset - ExplicitArgsOffset));

  return MF;
}

static llvm::Expected<llvm::MachineFunction &>
initLiftedDeviceFunctionEntry(object::AMDGCNDeviceFuncSymbolRef FuncSym,
                              llvm::Module &M, llvm::MachineModuleInfo &MMI) {
  llvm::LLVMContext &LLVMContext = M.getContext();

  llvm::Expected<llvm::StringRef> FuncNameOrErr = FuncSym.getName();
  LUTHIER_RETURN_ON_ERROR(FuncNameOrErr.takeError());
  llvm::Type *ReturnType = llvm::Type::getVoidTy(LLVMContext);
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {}, false);

  auto *F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::PrivateLinkage, *FuncNameOrErr, M);
  F->setCallingConv(llvm::CallingConv::C);
  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(LLVMContext, "", F);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      BB != nullptr,
      "Failed to create a dummy IR basic block during code lifting."));
  new llvm::UnreachableInst(LLVMContext, BB);
  auto &MF = MMI.getOrCreateMachineFunction(*F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));
  return MF;
}

llvm::PreservedAnalyses
AMDGPULiftGlobalObjectsPass::run(llvm::Module &TargetModule,
                                 llvm::ModuleAnalysisManager &TargetMAM) {
  llvm::LLVMContext &Ctx = TargetModule.getContext();
  /// Get the AMDGCN object file
  const llvm::object::ObjectFile &ObjFile =
      TargetMAM.getResult<ObjectFileAnalysisPass>(TargetModule).getObject();
  /// Get the MMI
  llvm::MachineModuleInfo &MMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetModule).getMMI();
  /// Get the global value symbols
  auto &GlobalObjectSymbols =
      TargetMAM.getResult<GlobalObjectSymbolsAnalysis>(TargetModule);
  auto &GlobalObjectOffsets =
      TargetMAM.getResult<GlobalObjectOffsetsAnalysis>(TargetModule);

  LUTHIER_EMIT_ERROR_IN_CONTEXT(
      Ctx, LUTHIER_GENERIC_ERROR_CHECK(
               llvm::isa<luthier::object::AMDGCNObjectFile>(ObjFile),
               "Object file is not an amdgcn object"));

  auto &AMDGCNObjFile = llvm::cast<luthier::object::AMDGCNObjectFile>(ObjFile);

  /// Loop over the symbols and initialize them
  for (object::AMDGCNElfSymbolRef Sym : AMDGCNObjFile) {
    const llvm::GlobalObject *GO{nullptr};

    /// Handle global variables
    auto SymAsVarOrErr =
        object::AMDGCNVariableSymbolRef::getAsAMDGCNVariableSymbol(Sym);
    LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, SymAsVarOrErr.takeError());

    if (SymAsVarOrErr->has_value()) {
      llvm::Expected<const llvm::GlobalVariable &> GlobalVarOrErr =
          initGlobalVariable(**SymAsVarOrErr, TargetModule);

      LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, GlobalVarOrErr.takeError());
      GO = &*GlobalVarOrErr;
    }

    /// Handle kernels
    auto SymAsKDOrErr =
        object::AMDGCNKernelDescSymbolRef::getAsAMDGCNKernelDescSymbol(Sym);
    LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, SymAsKDOrErr.takeError());

    if (SymAsKDOrErr->has_value()) {
      /// If the kernel is the same as the one being lifted, create a new
      /// function for it; Otherwise, it will become an external variable
      object::AMDGCNKernelDescSymbolRef LiftedKernelSymbol =
          TargetMAM
              .getResult<AMDGPULiftedKernelSymbolAnalysisPass>(TargetModule)
              .getSymbol();
      llvm::Expected<bool> IsLiftedKernelOrErr = object::areSymbolsEqual(
          AMDGCNObjFile, LiftedKernelSymbol.getRawDataRefImpl(),
          (*SymAsKDOrErr)->getRawDataRefImpl());
      LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, IsLiftedKernelOrErr.takeError());
      if (*IsLiftedKernelOrErr) {
        const auto &KernelMD =
            TargetMAM.getResult<AMDGCNMetadataParserAnalysisPass>(TargetModule)
                .getKernelMetadata();

        llvm::Expected<llvm::MachineFunction &> KernelFuncOrErr =
            initLiftedKernelEntry(**SymAsKDOrErr, KernelMD, TargetModule, MMI);
        LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, KernelFuncOrErr.takeError());

        GO = &KernelFuncOrErr->getFunction();
      } else {
        llvm::Expected<const llvm::GlobalVariable &> GlobalVarOrErr =
            initGlobalVariable(**SymAsKDOrErr, TargetModule);

        LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, GlobalVarOrErr.takeError());
        GO = &*GlobalVarOrErr;
      }
    }
    /// Handle device functions
    auto SymAsDevFuncOrErr =
        object::AMDGCNDeviceFuncSymbolRef::getAsAMDGCNDeviceFuncSymbol(Sym);
    LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, SymAsDevFuncOrErr.takeError());
    if (SymAsDevFuncOrErr->has_value()) {
      llvm::Expected<llvm::MachineFunction &> DevFuncOrErr =
          initLiftedDeviceFunctionEntry(**SymAsDevFuncOrErr, TargetModule, MMI);
      LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, DevFuncOrErr.takeError());
      GO = &DevFuncOrErr->getFunction();
    }
    if (GO) {
      llvm::Expected<std::optional<uint64_t>> SymLoadOffsetOrErr =
          getLoadOffset(Sym);
      LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, SymLoadOffsetOrErr.takeError());
      LUTHIER_EMIT_ERROR_IN_CONTEXT(
          Ctx,
          LUTHIER_GENERIC_ERROR_CHECK(
              SymLoadOffsetOrErr->has_value(),
              llvm::formatv("Global object {0} does not have a loaded offset",
                            GO->getName())));
      GlobalObjectSymbols.insertSymbol(*GO, Sym);
      GlobalObjectOffsets.insertSymbol(**SymLoadOffsetOrErr, *GO);
    }
  }
  /// Force calculate relocations for the lift functions passes
  (void)TargetMAM.getResult<ELFRelocationResolverAnalysisPass>(TargetModule);

  /// None of the analysis at this point relates to the IR/MIR so we preserve
  /// everything
  return llvm::PreservedAnalyses::all();
}
} // namespace luthier