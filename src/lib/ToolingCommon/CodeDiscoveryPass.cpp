
#include "luthier/tooling/CodeDiscoveryPass.h"
#include "LuthierRealToPseudoOpcodeMap.hpp"
#include "LuthierRealToPseudoRegEnumMap.hpp"
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/TargetRegistry.h>
#include <luthier/tooling/EntryPointsAnalysis.h>
#include <luthier/tooling/ExecutableMemorySegmentAccessor.h>
#include <luthier/tooling/InstructionTracesAnalysis.h>
#include <luthier/tooling/MetadataParserAnalysis.h>
#include <unordered_set>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-code-discovery"

namespace luthier {

static inline llvm::Type *
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

static inline void
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

static inline llvm::Error
parseKDRsrc1(const llvm::amdhsa::kernel_descriptor_t &KD,
             const llvm::GCNTargetMachine &TM, llvm::Function &F) {
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);
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

static inline llvm::Error
parseKDRsrc2(const llvm::amdhsa::kernel_descriptor_t &KD,
             const llvm::GCNTargetMachine &TM, llvm::Function &F) {
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

// static llvm::Error parseKDRsrc3(const llvm::amdhsa::kernel_descriptor_t &KD,
//                                 const llvm::GCNTargetMachine &TM,
//                                 llvm::Function &F) {
//   auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);
//   auto Generation = ST.getGeneration();
/// Rsrc3 for GFX90A and GFX942 ==============================================

/// ACCUM_OFFSET is automatically calculated based off of vgpr and agpr usage
/// of the MF

/// TG_SPLIT is set by the TM cumode feature

/// Rsrc3 for GFX10 and 11 ===================================================

/// SHARED_VGPR_COUNT is not set by the backend; Must be manually fixed
/// after the assembly is printed

/// INST_PREF_SIZE is automatically calculated according to the size of the
/// code of the kernel

/// TRAP_ON_START and TRAP_ON_END are filled in by the CP

/// IMAGE_OP is not set by the backend; Must be manually fixed after the
/// assembly is printed

/// Rsrc3 for GFX12 ==========================================================

/// INST_PREF_SIZE is automatically calculated according to the size of the
/// code of the kernel

/// GLG_EN is not set either by the backend or MC

/// IMAGE_OP is not set by the backend; Must be manually fixed after the
/// assembly is printed
// }

inline llvm::Error
parseKDKernelCode(const llvm::amdhsa::kernel_descriptor_t &KD,
                  const llvm::GCNTargetMachine &TM, llvm::MachineFunction &MF) {
  llvm::Function &F = MF.getFunction();
  auto MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
  auto TRI = reinterpret_cast<const llvm::SIRegisterInfo *>(
      TM.getSubtargetImpl(F)->getRegisterInfo());
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(F);

  if (!ST.flatScratchIsArchitected()) {
    if (AMDHSA_BITS_GET(
            KD.kernel_code_properties,
            llvm::amdhsa::
                KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) == 1) {
      MFI->addPrivateSegmentBuffer(*TRI);
    }
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) == 1) {
    MFI->addDispatchPtr(*TRI);
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) == 1) {
    MFI->addQueuePtr(*TRI);
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) ==
      1) {
    MFI->addKernargSegmentPtr(*TRI);
  }

  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) == 1) {
    MFI->addDispatchID(*TRI);
  }

  if (ST.flatScratchIsArchitected()) {
    if (AMDHSA_BITS_GET(
            KD.kernel_code_properties,
            llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) ==
        1) {
      MFI->addFlatScratchInit(*TRI);
    }
  }
  if (AMDHSA_BITS_GET(
          KD.kernel_code_properties,
          llvm::amdhsa::
              KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) == 1) {
    MFI->addPrivateSegmentSize(*TRI);
  }

  /// Wavefront32 should be taken care of when creating the target machine

  /// Set the size of the stack based on whether we have a dynamic stack or not
  if (KD.private_segment_fixed_size != 0) {
    llvm::MachineFrameInfo &FrameInfo = MF.getFrameInfo();
    FrameInfo.CreateFixedObject(KD.private_segment_fixed_size, 0, true);
    FrameInfo.setStackSize(KD.private_segment_fixed_size);
    if (AMDHSA_BITS_GET(
            KD.kernel_code_properties,
            llvm::amdhsa::KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK)) {
      FrameInfo.CreateVariableSizedObject(llvm::Align(4), nullptr);
    }
  }

  return llvm::Error::success();
};

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
      LUTHIER_RETURN_ON_ERROR(Err);
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

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated

  auto &KDOnHost = *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      &SegmentOrErr->SegmentOnHost[KDLoadOffset]);

  F->addFnAttr("amdgpu-lds-size",
               llvm::to_string(KDOnHost.group_segment_fixed_size));

  /// Lift Rsrc1-Rsrc2; Rsrc3 is mostly automatically computed so we don't
  /// lift it
  const auto &TM =
      *reinterpret_cast<const llvm::GCNTargetMachine *>(&TargetMMI.getTarget());
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc1(KDOnHost, TM, *F));
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc2(KDOnHost, TM, *F));

  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(TargetModule.getContext(), "", F);
  new llvm::UnreachableInst(TargetModule.getContext(), BB);

  // Populate the MFI ========================================================

  auto &MF = TargetMMI.getOrCreateMachineFunction(*F);

  /// Parse kernel code
  LUTHIER_RETURN_ON_ERROR(parseKDKernelCode(KDOnHost, TM, MF));

  auto MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(*F);

  /// Set pre-loaded kernel argument field for targets that support it
  if (ST.hasKernargPreload()) {
    /// TODO: It seems the AMDGPU backend doesn't support the offset field of
    /// kernarg_preload for now. Fix it once it is added to LLVM upstream
    MFI->getUserSGPRInfo().allocKernargPreloadSGPRs(KDOnHost.kernarg_preload);
  }

  /// Kernel functions are 2^8 byte aligned
  MF.setAlignment(llvm::Align(256));

  // Process the hidden args now that MFI and MF has been created
  if ((*KernelMDOrErr)->Args.has_value()) {
    // Add absence of all hidden arguments; As we iterate over all the
    // hidden arguments, we get rid of them if we detect their presence
    F->addFnAttr("amdgpu-no-hostcall-ptr");
    F->addFnAttr("amdgpu-no-default-queue");
    F->addFnAttr("amdgpu-no-completion-action");
    F->addFnAttr("amdgpu-no-multigrid-sync-arg");
    F->addFnAttr("amdgpu-no-heap-ptr");
    for (const auto &ArgMD : *(*KernelMDOrErr)->Args) {
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

llvm::Expected<llvm::MachineFunction &> initLiftedDeviceFunctionEntry(
    uint64_t DeviceEntryPointAddr,
    const ExecutableMemorySegmentAccessor &SegAccessor,
    llvm::Module &TargetModule, llvm::MachineModuleInfo &TargetMMI) {

  llvm::LLVMContext &LLVMContext = TargetModule.getContext();
  auto SegmentOrErr = SegAccessor.getSegment(DeviceEntryPointAddr);

  uint64_t DevFuncLoadOffset =
      DeviceEntryPointAddr -
      reinterpret_cast<uint64_t>(SegmentOrErr->SegmentOnDevice.data());

  /// Locate the symbol this entry address is part of
  std::optional<llvm::object::ELFSymbolRef> FuncSymRef{std::nullopt};
  uint64_t EntryFromStartOfSymbol{0};

  for (object::AMDGCNElfSymbolRef Symbol :
       SegmentOrErr->CodeObjectStorage->symbols()) {
    if (Symbol.getELFType() == llvm::ELF::STT_FUNC) {
      llvm::Expected<uint64_t> SymbolAddrOrErr = Symbol.getAddress();
      LUTHIER_RETURN_ON_ERROR(SymbolAddrOrErr.takeError());
      uint64_t SymbolSize = Symbol.getSize();
      if (*SymbolAddrOrErr <= DevFuncLoadOffset &&
          DevFuncLoadOffset <= (*SymbolAddrOrErr + SymbolSize)) {
        FuncSymRef = Symbol;
        EntryFromStartOfSymbol = DevFuncLoadOffset - *SymbolAddrOrErr;
      }
    }
  }

  std::string FuncName;
  if (FuncSymRef.has_value()) {
    LUTHIER_RETURN_ON_ERROR(FuncSymRef->getName().moveInto(FuncName));
    FuncName += llvm::formatv("x{0:x}", EntryFromStartOfSymbol);
  } else {
    FuncName = llvm::formatv("x{0:x}", DeviceEntryPointAddr);
  }

  llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {}, false);

  auto *F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::PrivateLinkage, FuncName, TargetModule);
  F->setCallingConv(llvm::CallingConv::C);
  // Add dummy IR instructions ===============================================
  // Very important to have a dummy IR BasicBlock; Otherwise MachinePasses
  // won't run
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(LLVMContext, "", F);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      BB != nullptr,
      "Failed to create a dummy IR basic block during code lifting."));
  new llvm::UnreachableInst(LLVMContext, BB);
  auto &MF = TargetMMI.getOrCreateMachineFunction(*F);

  MF.setAlignment(llvm::Align(4));
  return MF;
}

llvm::Error populateMF(const InstructionTracesAnalysis::Result &MFTrace,
                       uint64_t EntryAddr, llvm::MachineFunction &MF,
                       llvm::DenseMap<const llvm::MachineInstr *,
                                      const TraceInstr *> &MIToTraceInstrMap) {

  llvm::MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();

  const auto &TM =
      *reinterpret_cast<const llvm::GCNTargetMachine *>(&MF.getTarget());

  MF.push_back(MBB);
  auto MBBEntry = MBB;

  const llvm::MCRegisterInfo &MRI = *TM.getMCRegisterInfo();
  auto IP =
      std::unique_ptr<llvm::MCInstPrinter>(TM.getTarget().createMCInstPrinter(
          TM.getTargetTriple(), TM.getMCAsmInfo()->getAssemblerDialect(),
          *TM.getMCAsmInfo(), *TM.getMCInstrInfo(), MRI));

  llvm::MCContext &MCContext = MF.getContext();

  const llvm::MCInstrInfo &MCInstInfo = *TM.getMCInstrInfo();
  std::unique_ptr<llvm::MCInstrAnalysis> MIA(
      TM.getTarget().createMCInstrAnalysis(&MCInstInfo));

  llvm::DenseMap<uint64_t,
                 llvm::SmallVector<llvm::MachineInstr *>>
      UnresolvedBranchMIs; // < Set of branch instructions located at a
                           // device address waiting for their
                           // target to be resolved after MBBs and MIs
                           // are created
  llvm::DenseMap<luthier::address_t, llvm::MachineBasicBlock *>
      BranchTargetMBBs; // < Set of MBBs that will be the target of the
                        // UnresolvedBranchMIs

  llvm::SmallVector<llvm::MachineBasicBlock *, 4> MBBs;
  MBBs.push_back(MBB);

  const auto &BranchTargets = MFTrace.getBranchTargets();
  for (const auto &[TraceStartEndAddr, Trace] : MFTrace.getTraceGroup()) {
    uint64_t CurrentInstrAddr = TraceStartEndAddr.first;
    uint64_t LastInstrAddr = TraceStartEndAddr.second;
    while (CurrentInstrAddr <= LastInstrAddr) {
      const TraceInstr &Inst = Trace.at(CurrentInstrAddr);
      LLVM_DEBUG(llvm::dbgs()
                     << "+++++++++++++++++++++++++++++++++++++++++++++++"
                        "+++++++++++++++++++++++++\n";);
      auto MCInst = Inst.getMCInst();
      const unsigned Opcode = getPseudoOpcodeFromReal(MCInst.getOpcode());
      const llvm::MCInstrDesc &MCID = MCInstInfo.get(Opcode);
      bool IsDirectBranch = MCID.isBranch() && !MCID.isIndirectBranch();
      bool IsDirectBranchTarget =
          BranchTargets.contains(Inst.getLoadedDeviceAddress());
      LLVM_DEBUG(llvm::dbgs() << "Lifting and adding MC Inst: ";
                 MCInst.dump_pretty(llvm::dbgs(), IP.get(), " ", &MCContext);
                 llvm::dbgs() << "\n";
                 llvm::dbgs()
                 << llvm::formatv("Loaded address of the instruction: {0:x}\n",
                                  Inst.getLoadedDeviceAddress());
                 llvm::dbgs()
                 << llvm::formatv("Is branch? {0}\n", MCID.isBranch());
                 llvm::dbgs() << llvm::formatv("Is indirect branch? {0}\n",
                                               MCID.isIndirectBranch()););

      if (IsDirectBranchTarget) {
        LLVM_DEBUG(llvm::dbgs() << "Instruction is a branch target.\n";);
        if (!MBB->empty()) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "Current MBB is not empty; Creating a new basic block\n");
          auto OldMBB = MBB;
          MBB = MF.CreateMachineBasicBlock();
          MF.push_back(MBB);
          MBBs.push_back(MBB);
          OldMBB->addSuccessor(MBB);
          // Branch targets mark the beginning of an MBB
          LLVM_DEBUG(llvm::dbgs()
                     << "*********************************************"
                        "***************************\n");
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Current MBB is empty; No new block created "
                        "for the branch target.\n");
        }
        BranchTargetMBBs.insert({Inst.getLoadedDeviceAddress(), MBB});
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "Address {0:x} marks the beginning of MBB idx {1}.\n",
                       Inst.getLoadedDeviceAddress(), MBB->getNumber()););
      }
      llvm::MachineInstrBuilder Builder =
          llvm::BuildMI(MBB, llvm::DebugLoc(), MCID);
      MIToTraceInstrMap.insert({Builder.getInstr(), &Inst});

      LLVM_DEBUG(llvm::dbgs() << "Number of operands according to MCID: "
                              << MCID.operands().size() << "\n";
                 llvm::dbgs() << "Populating operands\n";);
      for (unsigned OpIndex = 0, E = MCInst.getNumOperands(); OpIndex < E;
           ++OpIndex) {
        const llvm::MCOperand &Op = MCInst.getOperand(OpIndex);
        if (Op.isReg()) {
          LLVM_DEBUG(llvm::dbgs() << "Resolving reg operand.\n");
          unsigned RegNum = RealToPseudoRegisterMapTable(Op.getReg());
          const bool IsDef = OpIndex < MCID.getNumDefs();
          unsigned Flags = 0;
          const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[OpIndex];
          if (IsDef && !OpInfo.isOptionalDef()) {
            Flags |= llvm::RegState::Define;
          }
          LLVM_DEBUG(llvm::dbgs()
                         << "Adding register "
                         << llvm::printReg(RegNum,
                                           MF.getSubtarget().getRegisterInfo())
                         << " with flags " << Flags << "\n";);
          (void)Builder.addReg(RegNum, Flags);
        } else if (Op.isImm()) {
          LLVM_DEBUG(llvm::dbgs() << "Resolving an immediate operand.\n");
          // TODO: Resolve immediate load/store operands if they don't have
          // relocations associated with them (e.g. when they happen in the
          // text section)
          luthier::address_t InstAddr = Inst.getLoadedDeviceAddress();
          size_t InstSize = Inst.getSize();
          if (!IsDirectBranch) {
            LLVM_DEBUG(
                llvm::dbgs()
                    << "Relocation was not applied for the "
                       "immediate operand, and it is not a direct branch.\n";
                llvm::dbgs() << "Adding the immediate operand directly to the "
                                "instruction\n");
            if (llvm::SIInstrInfo::isSOPK(*Builder)) {
              LLVM_DEBUG(llvm::dbgs() << "Instruction is in SOPK format\n");
              if (llvm::SIInstrInfo::sopkIsZext(Opcode)) {
                auto Imm = static_cast<uint16_t>(Op.getImm());
                LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                               "Adding truncated imm value: {0}\n", Imm));
                (void)Builder.addImm(Imm);
              } else {
                auto Imm = static_cast<int16_t>(Op.getImm());
                LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                               "Adding truncated imm value: {0}\n", Imm));
                (void)Builder.addImm(Imm);
              }
            } else {
              LLVM_DEBUG(llvm::dbgs()
                         << llvm::formatv("Adding Imm: {0}\n", Op.getImm()));
              (void)Builder.addImm(Op.getImm());
            }
          }

        } else
          llvm_unreachable("Unexpected operand type");
      }
      // Create a (fake) memory operand to keep the machine verifier happy
      // when encountering image instructions
      if (llvm::SIInstrInfo::isImage(*Builder)) {
        llvm::MachinePointerInfo PtrInfo =
            llvm::MachinePointerInfo::getConstantPool(MF);
        auto *MMO = MF.getMachineMemOperand(
            PtrInfo,
            MCInstInfo.get(Builder->getOpcode()).mayLoad()
                ? llvm::MachineMemOperand::MOLoad
                : llvm::MachineMemOperand::MOStore,
            16, llvm::Align(8));
        Builder->addMemOperand(MF, MMO);
      }

      if (MCInst.getNumOperands() < MCID.NumOperands) {
        LLVM_DEBUG(llvm::dbgs() << "Must fixup instruction ";
                   Builder->print(llvm::dbgs()); llvm::dbgs() << "\n";
                   llvm::dbgs() << "Num explicit operands added so far: "
                                << MCInst.getNumOperands() << "\n";
                   llvm::dbgs() << "Num explicit operands according to MCID: "
                                << MCID.NumOperands << "\n";);
        // Loop over missing explicit operands (if any) and fixup any missing
        for (unsigned int MissingExpOpIdx = MCInst.getNumOperands();
             MissingExpOpIdx < MCID.NumOperands; MissingExpOpIdx++) {
          LLVM_DEBUG(llvm::dbgs() << "Fixing up operand no " << MissingExpOpIdx
                                  << "\n";);
          auto OpType = MCID.operands()[MissingExpOpIdx].OperandType;
          if (OpType == llvm::MCOI::OPERAND_IMMEDIATE ||
              OpType == llvm::AMDGPU::OPERAND_KIMM32) {
            LLVM_DEBUG(llvm::dbgs() << "Added a 0-immediate operand.\n";);
            Builder.addImm(0);
          }
        }
      }

      LUTHIER_RETURN_ON_ERROR(fixupBitsetInst(*Builder));
      LUTHIER_RETURN_ON_ERROR(verifyInstruction(Builder));
      LLVM_DEBUG(llvm::dbgs() << "Final form of the instruction (not final if "
                                 "it's a direct branch): ";
                 Builder->print(llvm::dbgs()); llvm::dbgs() << "\n");
      // Basic Block resolving
      if (MCID.isTerminator()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Instruction is a terminator; Finishing basic block.\n");
        if (IsDirectBranch) {
          LLVM_DEBUG(llvm::dbgs() << "The terminator is a direct branch.\n");
          luthier::address_t BranchTarget;
          if (MIA->evaluateBranch(MCInst, Inst.getLoadedDeviceAddress(), 4,
                                  BranchTarget)) {
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                           "Address was resolved to {0:x}\n", BranchTarget));
            if (!UnresolvedBranchMIs.contains(BranchTarget)) {
              UnresolvedBranchMIs.insert({BranchTarget, {Builder.getInstr()}});
            } else {
              UnresolvedBranchMIs[BranchTarget].push_back(Builder.getInstr());
            }
          } else {
            LLVM_DEBUG(llvm::dbgs()
                       << "Error resolving the target address of the branch\n");
          }
        }
        // if this is the last instruction in the stream, no need for creating
        // a new basic block
        if (CurrentInstrAddr == LastInstrAddr) {
          LLVM_DEBUG(llvm::dbgs() << "Creating a new basic block.\n");
          auto OldMBB = MBB;
          MBB = MF.CreateMachineBasicBlock();
          MBBs.push_back(MBB);
          MF.push_back(MBB);
          // Don't add the next block to the list of successors if the
          // terminator is an unconditional branch
          if (!MCID.isUnconditionalBranch())
            OldMBB->addSuccessor(MBB);
          LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                         "Address {0:x} marks the beginning of MBB idx {1}.\n",
                         Inst.getLoadedDeviceAddress(), MBB->getNumber()););
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "*********************************************"
                      "***************************\n");
      }
    }
    // Resolve the branch and target MIs/MBBs
    LLVM_DEBUG(llvm::dbgs() << "Resolving direct branch MIs\n");
    for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "Resolving MIs jumping to target address {0:x}.\n",
                     TargetAddress));
      MBB = BranchTargetMBBs[TargetAddress];
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          MBB != nullptr,
          llvm::formatv("Failed to find the MachineBasicBlock associated with "
                        "the branch target address {0:x}.",
                        TargetAddress)));
      for (auto &MI : BranchMIs) {
        LLVM_DEBUG(llvm::dbgs() << "Resolving branch for the instruction ";
                   MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
        MI->addOperand(llvm::MachineOperand::CreateMBB(MBB));
        MI->getParent()->addSuccessor(MBB);
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "MBB {0:x} {1} was set as the target of the branch.\n",
                       MBB, MBB->getName()));
        LLVM_DEBUG(llvm::dbgs() << "Final branch instruction: ";
                   MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "*********************************************"
                               "***************************\n");

    MF.getRegInfo().freezeReservedRegs();

    // Populate the properties of MF
    llvm::MachineFunctionProperties &Properties = MF.getProperties();
    Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
    Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
    Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
    Properties.set(llvm::MachineFunctionProperties::Property::TracksLiveness);
    Properties.set(llvm::MachineFunctionProperties::Property::Selected);

    LLVM_DEBUG(llvm::dbgs() << "Final form of the Machine function:\n";
               MF.print(llvm::dbgs());
               llvm::dbgs() << "\n"
                            << "*********************************************"
                               "***************************\n";);
  }

  // Resolve the branch and target MIs/MBBs
  LLVM_DEBUG(llvm::dbgs() << "Resolving direct branch MIs\n");
  for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
    LLVM_DEBUG(
        llvm::dbgs() << llvm::formatv(
            "Resolving MIs jumping to target address {0:x}.\n", TargetAddress));
    MBB = BranchTargetMBBs[TargetAddress];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MBB != nullptr,
        llvm::formatv("Failed to find the MachineBasicBlock associated with "
                      "the branch target address {0:x}.",
                      TargetAddress)));
    for (auto &MI : BranchMIs) {
      LLVM_DEBUG(llvm::dbgs() << "Resolving branch for the instruction ";
                 MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
      MI->addOperand(llvm::MachineOperand::CreateMBB(MBB));
      MI->getParent()->addSuccessor(MBB);
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "MBB {0:x} {1} was set as the target of the branch.\n",
                     MBB, MBB->getName()));
      LLVM_DEBUG(llvm::dbgs() << "Final branch instruction: ";
                 MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "*********************************************"
                             "***************************\n");

  MF.getRegInfo().freezeReservedRegs();

  // Populate the properties of MF
  llvm::MachineFunctionProperties &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Properties.set(llvm::MachineFunctionProperties::Property::TracksLiveness);
  Properties.set(llvm::MachineFunctionProperties::Property::Selected);

  LLVM_DEBUG(llvm::dbgs() << "Final form of the Machine function:\n";
             MF.print(llvm::dbgs());
             llvm::dbgs() << "\n"
                          << "*********************************************"
                             "***************************\n";);
  return llvm::Error::success();
}

llvm::PreservedAnalyses
CodeDiscoveryPass::run(llvm::Module &TargetModule,
                       llvm::ModuleAnalysisManager &TargetMAM) {
  llvm::LLVMContext &Ctx = TargetModule.getContext();

  llvm::MachineModuleInfo &MMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetModule).getMMI();
  auto &TM =
      *reinterpret_cast<const llvm::GCNTargetMachine *>(&MMI.getTarget());

  const ExecutableMemorySegmentAccessor &SegAccessor =
      TargetMAM.getResult<ExecutableMemorySegmentAccessorAnalysis>(TargetModule)
          .getAccessor();

  auto &EntryPoints = TargetMAM.getResult<EntryPointsAnalysis>(TargetModule);

  auto &MFAM = TargetMAM
                   .getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(
                       TargetModule)
                   .getManager();

  std::unordered_set<EntryPointType> UnvisitedPointsOfEntry{InitialEntryPoint};
  std::unordered_set<EntryPointType> VisitedPointsOfEntry;

  while (!UnvisitedPointsOfEntry.empty()) {
    EntryPointType CurrentEntryPoint = *UnvisitedPointsOfEntry.begin();

    llvm::MachineFunction *MF{nullptr};

    /// Initialize the function handle associated with the entry point
    if (std::holds_alternative<const llvm::amdhsa::kernel_descriptor_t *>(
            CurrentEntryPoint)) {
      const auto &KDOnDevice =
          std::get<const llvm::amdhsa::kernel_descriptor_t *>(
              CurrentEntryPoint);

      const auto &MDParser =
          TargetMAM.getResult<MetadataParserAnalysis>(TargetModule).getParser();

      llvm::Expected<llvm::MachineFunction &> MFOrErr =
          initKernelEntryPointFunction(*KDOnDevice, SegAccessor, MDParser,
                                       TargetModule, MMI);
      LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, MFOrErr.takeError());

      MF = &*MFOrErr;
    } else {
      llvm::Expected<llvm::MachineFunction &> MFOrErr =
          initLiftedDeviceFunctionEntry(std::get<uint64_t>(CurrentEntryPoint),
                                        SegAccessor, TargetModule, MMI);
      LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, MFOrErr.takeError());
      MF = &*MFOrErr;
    }
    /// Add the newly created MF's entry point
    EntryPoints.insert(*MF, CurrentEntryPoint);
    /// Ask for the trace of the instructions
    auto TraceResults = MFAM.getResult<InstructionTracesAnalysis>(*MF);

    UnvisitedPointsOfEntry.erase(CurrentEntryPoint);
    VisitedPointsOfEntry.insert(CurrentEntryPoint);
  }

  return llvm::PreservedAnalyses::none();
}
} // namespace luthier