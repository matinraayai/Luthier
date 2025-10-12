
#include "luthier/tooling/CodeDiscoveryPass.h"

#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <luthier/tooling/ExecutableMemorySegmentAccessor.h>
#include <luthier/tooling/MetadataParserAnalysis.h>
#include <unordered_set>

namespace luthier {

static llvm::Type *
processExplicitKernelArg(const amdgpu::hsamd::Kernel::Arg::Metadata &ArgMD,
                         llvm::LLVMContext &Ctx) {
  llvm::Type *ParamType = llvm::Type::getIntNTy(Ctx, ArgMD.Size * 8);
  // Used when the argument kind is global buffer or dynamic shared pointer
  unsigned int AddressSpace = ArgMD.AddressSpace.has_value()
                                  ? *ArgMD.AddressSpace
                                  : llvm::AMDGPUAS::GLOBAL_ADDRESS;
  switch (ArgMD.ValKind) {
  case amdgpu::hsamd::ValueKind::ByValue:
    break;
  case amdgpu::hsamd::ValueKind::GlobalBuffer:
    // Convert the argument to a pointer
    ParamType = llvm::PointerType::get(ParamType, AddressSpace);
    break;
  default:
    llvm_unreachable("Not implemented");
  }
  return ParamType;
}

static void
processHiddenKernelArg(const amdgpu::hsamd::Kernel::Arg::Metadata &ArgMD,
                       llvm::Function &F, llvm::SIMachineFunctionInfo &MFI,
                       const llvm::SIRegisterInfo &TRI) {
  switch (ArgMD.ValKind) {
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

static llvm::Error parseKDRsrc1(const llvm::amdhsa::kernel_descriptor_t &KD,
                                const llvm::GCNTargetMachine &TM,
                                llvm::Function &F) {
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);
  auto Generation = ST.getGeneration();
  /// GRANULATED_WORKITEM_VGPR_COUNT is automatically calculated via the
  /// resource usage analysis pass

  /// GRANULATED_WAVEFRONT_SGPR_COUNT is automatically calculated via the
  /// resouce usage analysis pass

  /// PRIORITY is set by the CP automatically

  /// FLOAT_ROUND_MODE_32, FLOAT_ROUND_MODE_16_64 fields are set to
  /// FP_ROUND_ROUND_TO_NEAREST no matter what, so we will fix it once we
  /// have printed the relocatable of the instrumented object

  /// Lift FLOAT_DENORM_MODE_32 field
  auto Float32Denorm =
      AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                      llvm::amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  std::string Denorm32Val;
  switch (Float32Denorm) {
  case FP_DENORM_FLUSH_IN_FLUSH_OUT:
    Denorm32Val = "preserve-sign,preserve-sign";
    break;
  case FP_DENORM_FLUSH_OUT:
    Denorm32Val = "preserve-sign,";
    break;
  case FP_DENORM_FLUSH_IN:
    Denorm32Val = ",preserve-sign";
    break;
  case FP_DENORM_FLUSH_NONE:
    Denorm32Val = ",";
    break;
  default:
    return llvm::make_error<GenericLuthierError>(
        "Invalid FP 32 denorm field " + llvm::to_string(Float32Denorm) + ".");
  }
  F.addFnAttr("denormal-fp-math-f32", Denorm32Val);
  /// Lift FLOAT_DENORM_MODE_16_64 field
  auto Float1664Denorm =
      AMDHSA_BITS_GET(KD.compute_pgm_rsrc1,
                      llvm::amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  std::string Denorm1664Val;
  switch (Float1664Denorm) {
  case FP_DENORM_FLUSH_IN_FLUSH_OUT:
    Denorm1664Val = "preserve-sign,preserve-sign";
    break;
  case FP_DENORM_FLUSH_OUT:
    Denorm1664Val = "preserve-sign,";
    break;
  case FP_DENORM_FLUSH_IN:
    Denorm1664Val = ",preserve-sign";
    break;
  case FP_DENORM_FLUSH_NONE:
    Denorm1664Val = ",";
    break;
  default:
    return llvm::make_error<GenericLuthierError>(
        "Invalid FP 16/64 denorm field " + llvm::to_string(Float32Denorm) +
        ".");
  }
  F.addFnAttr("denormal-fp-math", Denorm1664Val);
  /// Lift ENABLE_DX10_CLAMP if gfx11-; Otherwise lift WG_RR_EN
  if (ST.hasRrWGMode()) {
    /// WG_RR_EN is set to zero by the backend; must be fixed after the assembly
    /// is printed

  } else {
    F.addFnAttr(
        "amdgpu-dx10-clamp",
        AMDHSA_BITS_GET(
            KD.compute_pgm_rsrc1,
            llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_DX10_CLAMP)
            ? "true"
            : "false");
  }

  /// DEBUG_MODE is set by the CP

  /// Lift ENABLE_IEEE_MODE if gfx11-; DISABLE_PERF is reserved and must be set
  /// to zero
  if (ST.hasIEEEMode()) {
    F.addFnAttr("amdgpu-ieee",
                AMDHSA_BITS_GET(
                    KD.compute_pgm_rsrc1,
                    llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX6_GFX11_ENABLE_IEEE_MODE)
                    ? "true"
                    : "false");
  }

  /// BULKY is set by CP

  /// CDBG_USER is set by CP

  /// FP16_OVFL can be queried on device code, but there's no place to set it
  /// at the MIR level

  /// WGP_MODE is set by the cumode feature in the subtarget

  /// MEM_ORDERED is always set to 1 for GFX10+ in the backend, so we will fix
  /// it once we have printed the relocatable of the instrumented object

  /// FWD_PROGRESS is always set to 1 for GFX10+, so we will fix it once we
  /// have printed the relocatable of the instrumented object

  return llvm::Error::success();
}

static llvm::Error parseKDRsrc2(const llvm::amdhsa::kernel_descriptor_t &KD,
                                const llvm::GCNTargetMachine &TM,
                                llvm::Function &F) {
  /// ENABLE_PRIVATE_SEGMENT is set if the stack size of the kernel is set to
  /// non-zero value

  /// USER_SGPR_COUNT is automatically set based on the user sgprs requested
  /// from the MFI

  /// ENABLE_TRAP_HANDLER is not set in HSA; For other OSes, it should be set
  /// when creating the target

  /// ENABLE_DYNAMIC_VGPR is not seem to have a handle neither in MIR or MC

  /// Lift ENABLE_SGPR_WORKGROUP_ID_X, Y and Z
  if (AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) == 0) {
    F.addFnAttr("amdgpu-no-workgroup-id-x");
  }

  if (AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) == 0) {
    F.addFnAttr("amdgpu-no-workgroup-id-y");
  }
  if (AMDHSA_BITS_GET(
          KD.compute_pgm_rsrc2,
          llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) == 0) {
    F.addFnAttr("amdgpu-no-workgroup-id-z");
  }
  /// ENABLE_SGPR_WORKGROUP_INFO is represented in MFI, but it is always set to
  /// false and there is no way to set it

  /// Lift ENABLE_VGPR_WORKITEM_ID
  switch (AMDHSA_BITS_GET(
      KD.compute_pgm_rsrc2,
      llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID)) {
  case 0:
    F.addFnAttr("amdgpu-no-workitem-id-y");
  case 1:
    F.addFnAttr("amdgpu-no-workitem-id-z");
  case 2:
    break;
  default:
    return llvm::make_error<GenericLuthierError>(
        "KD's VGPR workitem ID is not valid");
  }
  /// ENABLE_EXCEPTION_ADDRESS_WATCH is always set to zero

  /// ENABLE_EXCEPTION_MEMORY is always set to zero

  /// GRANULATED_LDS_SIZE is automatically set via the lds-size attribute

  /// ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION,
  /// ENABLE_EXCEPTION_FP_DENORMAL_SOURCE,
  /// ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO,
  /// ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW,
  /// ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW
  /// ENABLE_EXCEPTION_IEEE_754_FP_INEXACT
  /// ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO are all set to zero; Must be fixed
  /// after printing the assembly

  return llvm::Error::success();
}

static llvm::Expected<llvm::MachineFunction &>
initKernelEntryPointFunction(const llvm::amdhsa::kernel_descriptor_t &KD,
                             const ExecutableMemorySegmentAccessor &SegAccessor,
                             const amdgpu::hsamd::MetadataParser &MDParser,
                             llvm::Module &TargetModule,
                             llvm::MachineModuleInfo &TargetMMI) {
  llvm::LLVMContext &LLVMContext = TargetModule.getContext();
  auto KDLoadAddr = reinterpret_cast<uint64_t>(&KD);
  auto SegmentOrErr = SegAccessor.getSegment(KDLoadAddr);

  uint64_t KDLoadOffset =
      KDLoadAddr -
      reinterpret_cast<uint64_t>(SegmentOrErr->SegmentOnDevice.data());

  /// Locate the KD symbol inside the code object
  llvm::Expected<object::AMDGCNKernelDescSymbolRef> KDSymbolOrErr =
      [&]() -> llvm::Expected<object::AMDGCNKernelDescSymbolRef> {
    llvm::Error Err = llvm::Error::success();
    for (object::AMDGCNKernelDescSymbolRef CurrentKD :
         SegmentOrErr->CodeObjectStorage->kernel_descriptors(Err)) {
      llvm::Expected<uint64_t> CurrentKDLoadAddrOrErr = CurrentKD.getAddress();
      LUTHIER_RETURN_ON_ERROR(CurrentKDLoadAddrOrErr.takeError());
      if (*CurrentKDLoadAddrOrErr == KDLoadAddr) {
        return CurrentKD;
      }
    }
    return llvm::make_error<GenericLuthierError>(llvm::formatv(
        "Failed to get the KD associated with address {0:x}", KDLoadOffset));
  }();

  LUTHIER_RETURN_ON_ERROR(KDSymbolOrErr.takeError());

  /// Get the kernel's name
  llvm::Expected<llvm::StringRef> KernelNameOrErr = KDSymbolOrErr->getName();
  LUTHIER_RETURN_ON_ERROR(KernelNameOrErr.takeError());

  /// Parse the kernel's metadata
  llvm::Expected<std::unique_ptr<llvm::msgpack::Document>> MDDocOrErr =
      SegmentOrErr->CodeObjectStorage->getMetadataDocument();
  LUTHIER_RETURN_ON_ERROR(MDDocOrErr.takeError());
  auto KernelMDOrErr =
      MDParser.parseKernelMetadata(**MDDocOrErr, *KernelNameOrErr);
  LUTHIER_RETURN_ON_ERROR(KernelMDOrErr.takeError());

  // Populate the Arguments
  // ==================================================

  // Kernel's return type is always void
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);

  // Create the Kernel's FunctionType with appropriate kernel Arguments
  // (if any)
  llvm::SmallVector<llvm::Type *> Params;
  unsigned int ExplicitArgsOffset = 0;
  unsigned int ImplicitArgsOffset = 0;
  if ((*KernelMDOrErr)->Args.has_value()) {
    // Reserve the number of arguments in the Params vector
    Params.reserve((*KernelMDOrErr)->Args->size());
    // For now, we only rely on required argument metadata
    // This should be updated as new cases are encountered
    for (const auto &ArgMD : *(*KernelMDOrErr)->Args) {
      if (ArgMD.ValKind >= amdgpu::hsamd::ValueKind::HiddenArgKindBegin)
        break;
      else {
        Params.push_back(processExplicitKernelArg(ArgMD, LLVMContext));
        if (ArgMD.Offset > ExplicitArgsOffset)
          ExplicitArgsOffset = ArgMD.Offset;
      }
    }
  }

  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, Params, false);

  auto *F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::WeakAnyLinkage,
      KernelNameOrErr->substr(0, KernelNameOrErr->rfind(".kd")), TargetModule);
  F->setVisibility(llvm::GlobalValue::ProtectedVisibility);

  // Populate the Attributes =================================================

  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

  F->addFnAttr("uniform-work-group-size",
               KernelMD.UniformWorkgroupSize ? "true" : "false");

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated

  auto &KDOnHost = *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      &SegmentOrErr->SegmentOnHost[KDLoadOffset]);

  F->addFnAttr("amdgpu-lds-size",
               llvm::to_string(KDOnHost.group_segment_fixed_size));
  // Kern Arg is determined via analysis usage + args set earlier

  /// Lift Rsrc1
  const auto &TM =
      *reinterpret_cast<const llvm::GCNTargetMachine *>(&TargetMMI.getTarget());
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc1(KDOnHost, TM, *F));
  auto Rsrc2 = (*KDOnHost)->getRsrc2();
  auto KCP = (*KDOnHost)->getKernelCodeProperties();
  if (KCP.EnableSgprDispatchId == 0) {
    F->addFnAttr("amdgpu-no-dispatch-id");
  }
  if (KCP.EnableSgprDispatchPtr == 0) {
    F->addFnAttr("amdgpu-no-dispatch-ptr");
  }
  if (KCP.EnableSgprQueuePtr == 0) {
    F->addFnAttr("amdgpu-no-queue-ptr");
  }

  // TODO: Check the args metadata to set this correctly
  // TODO: Set the rest of the attributes
  //    luthier::outs() << "Preloaded Args: " << (*KDOnHost)->KernArgPreload <<
  //    "\n";
  //  F->addFnAttr("amdgpu-calls");
  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(TargetModule.getContext(), "", F);
  new llvm::UnreachableInst(TargetModule.getContext(), BB);

  // Populate the MFI ========================================================

  auto &MF = TargetMMI.getOrCreateMachineFunction(*F);

  // TODO: Fix alignment value depending on the function type
  MF.setAlignment(llvm::Align(4096));

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
  if ((*KernelMDOrErr)->Args.has_value()) {
    // Add absence of all hidden arguments; As we iterate over all the
    // hidden arguments, we get rid of them if we detect their presence
    F->addFnAttr("amdgpu-no-hostcall-ptr");
    F->addFnAttr("amdgpu-no-default-queue");
    F->addFnAttr("amdgpu-no-completion-action");
    F->addFnAttr("amdgpu-no-multigrid-sync-arg");
    F->addFnAttr("amdgpu-no-heap-ptr");
    for (const auto &ArgMD : (*KernelMDOrErr)->Args) {
      if (ArgMD.ValKind >= amdgpu::hsamd::ValueKind::HiddenArgKindBegin &&
          ArgMD.ValKind <= amdgpu::hsamd::ValueKind::HiddenArgKindEnd) {
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

llvm::PreservedAnalyses
CodeDiscoveryPass::run(llvm::Module &TargetModule,
                       llvm::ModuleAnalysisManager &TargetMAM) {
  llvm::LLVMContext &Ctx = TargetModule.getContext();
  /// Get the MMI
  llvm::MachineModuleInfo &MMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetModule).getMMI();
  /// Get the device segment accessor
  const ExecutableMemorySegmentAccessor &SegAccessor =
      TargetMAM.getResult<ExecutableMemorySegmentAccessorAnalysis>(TargetModule)
          .getAccessor();

  std::unordered_set<EntryPointType> UnvisitedPointsOfEntry{InitialEntryPoint};
  std::unordered_set<EntryPointType> VisitedPointsOfEntry;

  while (!UnvisitedPointsOfEntry.empty()) {
    EntryPointType CurrentEntryPoint = *UnvisitedPointsOfEntry.begin();

    ExecutableMemorySegmentAccessor::SegmentDescriptor SegDesc;

    uint64_t CurrentInstructionAddrs{0};

    /// Initialize the function handle associated with the entry point
    if (std::holds_alternative<const llvm::amdhsa::kernel_descriptor_t *>(
            CurrentEntryPoint)) {
      const auto &KDOnDevice =
          std::get<const llvm::amdhsa::kernel_descriptor_t *>(
              CurrentEntryPoint);

      const auto &MDParser =
          TargetMAM.getResult<MetadataParserAnalysis>(TargetModule).getParser();

      LUTHIER_EMIT_ERROR_IN_CONTEXT(
          Ctx,
          SegAccessor.getSegment(reinterpret_cast<const uint64_t>(&KDOnDevice))
              .moveInto(SegDesc));

    } else {
    }

    UnvisitedPointsOfEntry.erase(CurrentEntryPoint);
    VisitedPointsOfEntry.insert(CurrentEntryPoint);
  }

  return llvm::PreservedAnalyses::none();
}
} // namespace luthier