
#include "luthier/Tooling/CodeDiscoveryPass.h"
#include "AMDGPUTargetMachine.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Tooling/EntryPoint.h"
#include "luthier/Tooling/FunctionAnnotations.h"
// #include "luthier/Tooling/IPVectorRegLiveness.h"
#include "../../../include/luthier/InstSemantics/PseudoOpcodeAnRegMapper.h"
#include "luthier/Tooling/InitialEntryPointAnalysis.h"
#include "luthier/Tooling/InstructionTracesAnalysis.h"
#include "luthier/Tooling/MemoryAllocationAccessor.h"
#include "luthier/Tooling/MetadataParserAnalysis.h"
#include "luthier/Tooling/TargetMachineInstrMDNode.h"
#include <MCTargetDesc/AMDGPUMCExpr.h>
#include <SIMachineFunctionInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/ReachingDefAnalysis.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/TargetRegistry.h>
// #include <luthier/Tooling/IPVectorRegLiveness.h>
// #include <luthier/Tooling/IndirectBranchResolverAnalysis.h>
// #include <luthier/Tooling/MachineFunctionEntryPoint.h>
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
    ParamType = llvm::PointerType::get(Ctx, AddressSpace);
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

  /// ENABLE_DYNAMIC_VGPR does not seem to have a handle neither in MIR or MC

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

/// Initializes the \c llvm::Function and \c llvm::MachineFunction for the
/// entry point for the \p KD
static llvm::Expected<std::pair<llvm::MachineFunction &,
                                std::optional<object::AMDGCNElfSymbolRef>>>
initKernelEntryPointFunction(const llvm::amdhsa::kernel_descriptor_t &KD,
                             const MemoryAllocationAccessor &SegAccessor,
                             const amdgpu::hsamd::MetadataParser &MDParser,
                             llvm::Module &TargetModule,
                             const llvm::GCNTargetMachine &TM,
                             llvm::FunctionAnalysisManager &FAM) {
  llvm::LLVMContext &LLVMContext = TargetModule.getContext();
  /// Get the memory allocation associated with the kernel descriptor
  auto KDLoadAddr = reinterpret_cast<uint64_t>(&KD);
  auto SegmentOrErr = SegAccessor.getAllocationDescriptor(KDLoadAddr);
  LUTHIER_RETURN_ON_ERROR(SegmentOrErr.takeError());

  if (SegmentOrErr->empty())
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Failed to find the memory allocation descriptor "
                      "associated with KD {0:x}.",
                      KDLoadAddr));

  uint64_t KDLoadOffset =
      KDLoadAddr -
      reinterpret_cast<uint64_t>(SegmentOrErr->getDeviceAllocation().data());

  std::optional<object::AMDGCNKernelDescSymbolRef> KDSymbolIfPresent{
      std::nullopt};

  auto *ObjFile = SegmentOrErr->getAllocationCodeObject();

  /// If the KD allocation has a code object associated with it,
  /// locate the KD's symbol
  if (ObjFile) {
    /// Lambda is used instead of a plain loop for easier break/error checking
    llvm::Expected<object::AMDGCNKernelDescSymbolRef> KDSymbolOrErr =
        [&]() -> llvm::Expected<object::AMDGCNKernelDescSymbolRef> {
      llvm::Error Err = llvm::Error::success();
      for (object::AMDGCNKernelDescSymbolRef CurrentKD :
           ObjFile->kernel_descriptors(Err)) {
        LUTHIER_RETURN_ON_ERROR(Err);
        llvm::Expected<uint64_t> CurrentKDLoadAddrOrErr =
            CurrentKD.getAddress();
        LUTHIER_RETURN_ON_ERROR(CurrentKDLoadAddrOrErr.takeError());
        if (*CurrentKDLoadAddrOrErr == KDLoadOffset) {
          return CurrentKD;
          LUTHIER_RETURN_ON_ERROR(Err);
        }
      }
      LUTHIER_RETURN_ON_ERROR(Err);
      return llvm::make_error<GenericLuthierError>(llvm::formatv(
          "Failed to get the KD associated with address {0:x}", KDLoadOffset));
    }();
    LUTHIER_RETURN_ON_ERROR(KDSymbolOrErr.takeError());

    KDSymbolIfPresent = *KDSymbolOrErr;
  }

  /// If we have an object symbol associated with the kernel descriptor, then
  /// we find its name from its kernel symbol; Otherwise, give it a name
  /// based on its load address
  std::string KernelName;
  if (KDSymbolIfPresent.has_value()) {
    llvm::Expected<llvm::StringRef> KernelNameOrErr =
        KDSymbolIfPresent->getName();
    LUTHIER_RETURN_ON_ERROR(KernelNameOrErr.takeError());
    KernelName = KernelNameOrErr->substr(0, KernelNameOrErr->rfind(".kd"));
  } else {
    KernelName = llvm::formatv("kernel-{0:x}", KDLoadAddr);
  }

  /// Kernel's return type is always void
  /// We also do not attempt to recover the kernel's prototype from the metadata
  /// because the raised LLVM IR version of the kernel has to translate
  /// low-level argument buffer loading instructions not present in the
  /// high-level prototype of the original kernel (assuming the kernel was
  /// written in a high-level language); For example, if the kernarg buffer is
  /// passed in s[4:5], the raised IR instruction of s_load_dword s[4:5], s[4:5]
  /// 0x0 will be raised to a load, not "read from the first kernel function's
  /// argument"
  /// Also there is no guarantee if a kernel written in assembly tries
  /// to conform with the LLVM IR convention of "explicit args first, implicit
  /// arg after" i.e. it might mix implicit and explicit arguments together
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(LLVMContext);

  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {}, false);

  llvm::Function *F =
      llvm::Function::Create(FunctionType, llvm::GlobalValue::WeakAnyLinkage,
                             KernelName, TargetModule);
  F->setVisibility(llvm::GlobalValue::ProtectedVisibility);

  // Populate the Attributes =================================================

  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);

  // Construct the attributes of the Function, which will result in the MF
  // attributes getting populated

  auto &KDOnHost = *reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      &SegmentOrErr->getHostAllocation()[KDLoadOffset]);

  F->addFnAttr("amdgpu-lds-size",
               llvm::to_string(KDOnHost.group_segment_fixed_size));

  /// Lift Rsrc1-Rsrc2; Rsrc3 is mostly automatically computed so we don't
  /// lift it
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc1(KDOnHost, TM, *F));
  LUTHIER_RETURN_ON_ERROR(parseKDRsrc2(KDOnHost, TM, *F));

  // Populate the MFI ==========================================================

  llvm::MachineFunction &MF =
      FAM.getResult<llvm::MachineFunctionAnalysis>(*F).getMF();

  /// Parse the kernel code field of the kernel
  LUTHIER_RETURN_ON_ERROR(parseKDKernelCode(KDOnHost, TM, MF));

  auto *MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();
  auto &ST = TM.getSubtarget<llvm::GCNSubtarget>(*F);

  /// Set pre-loaded kernel argument field for targets that support it
  if (ST.kernargPreload()) {
    /// TODO: It seems the AMDGPU backend doesn't support the offset field of
    /// kernarg_preload for now. Fix it once it is added to LLVM upstream
    MFI->getUserSGPRInfo().allocKernargPreloadSGPRs(KDOnHost.kernarg_preload);
  }

  /// Kernel functions are 2^8 byte aligned
  MF.setAlignment(llvm::Align(256));

  /// We do not define any implicit arguments to be present since at the binary
  /// level we don't care whether an argument is implicit or explicit and we
  /// treat them the same
  F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");

  return std::make_pair(std::ref(MF), KDSymbolIfPresent);
}

llvm::Expected<std::pair<llvm::MachineFunction &,
                         std::optional<object::AMDGCNElfSymbolRef>>>
initLiftedDeviceFunctionEntry(uint64_t DeviceEntryPointAddr,
                              const MemoryAllocationAccessor &SegAccessor,
                              llvm::Module &TargetModule,
                              llvm::FunctionAnalysisManager &FAM) {

  llvm::LLVMContext &LLVMContext = TargetModule.getContext();
  auto SegmentOrErr = SegAccessor.getAllocationDescriptor(DeviceEntryPointAddr);
  LUTHIER_RETURN_ON_ERROR(SegmentOrErr.takeError());
  if (SegmentOrErr->empty())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Failed to find the allocation associated with address {0:x}",
        DeviceEntryPointAddr));

  uint64_t DevFuncLoadOffset =
      DeviceEntryPointAddr -
      reinterpret_cast<uint64_t>(SegmentOrErr->getDeviceAllocation().data());

  /// Locate the symbol this entry address is part of
  std::optional<object::AMDGCNElfSymbolRef> FuncSymRef{std::nullopt};
  uint64_t EntryFromStartOfSymbol{0};

  if (auto *ObjFile = SegmentOrErr->getAllocationCodeObject()) {
    for (object::AMDGCNElfSymbolRef Symbol : ObjFile->symbols()) {
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
  }

  std::string FuncName;
  if (FuncSymRef.has_value()) {
    LUTHIER_RETURN_ON_ERROR(FuncSymRef->getName().moveInto(FuncName));
    FuncName += llvm::formatv("x{0:x}", EntryFromStartOfSymbol);
  } else {
    FuncName = llvm::formatv("x{0:x}", DeviceEntryPointAddr);
  }

  llvm::Type *ReturnType = llvm::Type::getVoidTy(LLVMContext);
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {}, false);

  llvm::Function *F = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::PrivateLinkage, FuncName, TargetModule);
  F->setCallingConv(llvm::CallingConv::C);

  llvm::MachineFunction &MF =
      FAM.getResult<llvm::MachineFunctionAnalysis>(*F).getMF();

  MF.setAlignment(llvm::Align(4));
  return std::make_pair(std::ref(MF), FuncSymRef);
}

static bool shouldImplicitReadExec(const llvm::MachineInstr &MI) {
  if (llvm::SIInstrInfo::isVALU(MI)) {
    switch (MI.getOpcode()) {
    case llvm::AMDGPU::V_READLANE_B32:
    case llvm::AMDGPU::SI_RESTORE_S32_FROM_VGPR:
    case llvm::AMDGPU::V_WRITELANE_B32:
    case llvm::AMDGPU::SI_SPILL_S32_TO_VGPR:
      return false;
    default:
      return true;
    }
  }

  if (llvm::SIInstrInfo::isSALU(MI) || llvm::SIInstrInfo::isSMRD(MI))
    return false;

  return true;
}

/// Walks over the \p MCOperands and converts them to \c llvm::MachineOperand
/// instances before adding them to the \c llvm::MachineInstr managed
/// by the \p MIBuilder
/// \note Does not convert direct branch target immediate operands to a machine
/// basic block
/// \return \c llvm::Error indicating the success or failure of the operation
static llvm::Error
convertAndAddMCOperandsToMI(llvm::ArrayRef<llvm::MCOperand> MCOperands,
                            llvm::MachineInstrBuilder &MIBuilder) {
  const unsigned Opcode = MIBuilder->getOpcode();
  const llvm::MCInstrDesc &MCID = MIBuilder->getDesc();
  llvm::MachineBasicBlock *MBB = MIBuilder->getParent();
  assert(MBB && "MI is not part of a machine basic block");
  llvm::MachineFunction *MF = MBB->getParent();
  assert(MF && "MI is not part of a machine function");
  const bool IsDirectBranch =
      MIBuilder->isBranch() && !MIBuilder->isIndirectBranch();
  for (auto [MCOpIdx, MCOp] : llvm::enumerate(MCOperands)) {
    if (MCOp.isReg()) {
      LLVM_DEBUG(llvm::dbgs() << "Converting MC register operand.\n");
      unsigned RegNum = RealToPseudoRegisterMapTable(MCOp.getReg());
      const bool IsDef = MCOpIdx < MCID.getNumDefs();
      auto Flags = llvm::RegState::NoFlags;
      const llvm::MCOperandInfo &OpInfo = MCID.operands().begin()[MCOpIdx];
      if (IsDef && !OpInfo.isOptionalDef()) {
        Flags |= llvm::RegState::Define;
      }
      LLVM_DEBUG(llvm::dbgs()
                     << "Adding register "
                     << llvm::printReg(RegNum,
                                       MF->getSubtarget().getRegisterInfo())
                     << " with flags " << Flags << "\n";);
      (void)MIBuilder.addReg(RegNum, Flags);
    } else if (MCOp.isImm()) {
      LLVM_DEBUG(llvm::dbgs() << "Resolving an immediate operand.\n");

      if (!IsDirectBranch) {
        LLVM_DEBUG(llvm::dbgs()
                       << "Relocation was not applied for the "
                          "immediate operand, and it is not a direct branch.\n";
                   llvm::dbgs()
                   << "Adding the immediate operand directly to the "
                      "instruction\n");
        /// SOPK needs some special attention to be converted correctly
        if (llvm::SIInstrInfo::isSOPK(*MIBuilder)) {
          LLVM_DEBUG(llvm::dbgs() << "Instruction is in SOPK format\n");
          if (llvm::SIInstrInfo::sopkIsZext(MIBuilder->getOpcode())) {
            auto Imm = static_cast<uint16_t>(MCOp.getImm());
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                           "Adding truncated imm value: {0}\n", Imm));
            (void)MIBuilder.addImm(Imm);
          } else {
            auto Imm = static_cast<int16_t>(MCOp.getImm());
            LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                           "Adding truncated imm value: {0}\n", Imm));
            (void)MIBuilder.addImm(Imm);
          }
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << llvm::formatv("Adding Imm: {0}\n", MCOp.getImm()));
          (void)MIBuilder.addImm(MCOp.getImm());
        }
      }
    } else if (MCOp.isExpr() && llvm::isa<llvm::AMDGPUMCExpr>(MCOp.getExpr())) {
      const auto *TargetExpr = llvm::cast<llvm::AMDGPUMCExpr>(MCOp.getExpr());

      switch (TargetExpr->getKind()) {
      case llvm::AMDGPUMCExpr::AGVK_Lit:
      case llvm::AMDGPUMCExpr::AGVK_Lit64:
        LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
            TargetExpr->getArgs().size() == 1 &&
                llvm::isa<llvm::MCConstantExpr>(TargetExpr->getSubExpr(0)),
            "Literal expr operands should only have one constant sub "
            "expression"));
        (void)MIBuilder.addImm(
            llvm::cast<llvm::MCConstantExpr>(TargetExpr->getSubExpr(0))
                ->getValue());
        break;
      default:
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "Unexpected expression type: {0}", TargetExpr->getKind()));
      }
    } else {
      return LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Unexpected MC operand: ", MCOp));
    }
  }
  // Create a (fake) memory operand to keep the machine verifier happy
  // when encountering image instructions
  if (llvm::SIInstrInfo::isImage(*MIBuilder)) {
    llvm::MachinePointerInfo PtrInfo =
        llvm::MachinePointerInfo::getConstantPool(*MF);
    llvm::MachineMemOperand *MMO = MF->getMachineMemOperand(
        PtrInfo,
        MCID.mayLoad() ? llvm::MachineMemOperand::MOLoad
                       : llvm::MachineMemOperand::MOStore,
        16, llvm::Align(8));
    MIBuilder->addMemOperand(*MF, MMO);
  }

  if (size_t NumMCOps = MCOperands.size(); NumMCOps < MCID.NumOperands) {
    LLVM_DEBUG(llvm::dbgs() << "Must fixup instruction ";
               MIBuilder->print(llvm::dbgs()); llvm::dbgs() << "\n";
               llvm::dbgs()
               << "Num explicit operands added so far: " << NumMCOps << "\n";
               llvm::dbgs()
               << "Num explicit operands according to Machine Instr Info: "
               << MCID.NumOperands << "\n";);
    // Loop over missing explicit operands (if any) and fixup any missing
    // immediate operands; We ignore any other missing operand kinds here and
    // fix it later
    for (unsigned int MissingExpOpIdx = NumMCOps;
         MissingExpOpIdx < MCID.NumOperands; MissingExpOpIdx++) {
      LLVM_DEBUG(llvm::dbgs()
                     << "Fixing up operand no " << MissingExpOpIdx << "\n";);
      auto OpType = MCID.operands()[MissingExpOpIdx].OperandType;
      if (OpType == llvm::MCOI::OPERAND_IMMEDIATE ||
          OpType == llvm::AMDGPU::OPERAND_KIMM32) {
        LLVM_DEBUG(llvm::dbgs() << "Added a 0-immediate operand.\n";);
        (void)MIBuilder.addImm(0);
      }
    }
  }

  if (Opcode == llvm::AMDGPU::S_BITSET0_B32 ||
      Opcode == llvm::AMDGPU::S_BITSET0_B64 ||
      Opcode == llvm::AMDGPU::S_BITSET1_B32 ||
      Opcode == llvm::AMDGPU::S_BITSET1_B64) {
    // bitset instructions have a tied def/use that is not reflected in the
    // MC version
    if (MIBuilder->getNumOperands() < MIBuilder->getNumExplicitOperands()) {
      const llvm::MachineOperand &DefMachineOp = MIBuilder->getOperand(0);
      // Check if the first operand is a register
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          DefMachineOp.isReg(),
          "The first operand of a bitset instruction is not a register."));
      // Add the output reg also as the first input, and tie the first and
      // second operands together
      MIBuilder->addOperand(
          llvm::MachineOperand::CreateReg(DefMachineOp.getReg(), false));
      MIBuilder->tieOperands(0, 2);
    }
  }

  /// Add implicit use of the execute mask if it's not already reflected in
  /// the machine instruction
  if (MCID.hasImplicitUseOfPhysReg(llvm::AMDGPU::EXEC) &&
      !MIBuilder->hasRegisterImplicitUseOperand(llvm::AMDGPU::EXEC)) {
    MIBuilder->addOperand(
        llvm::MachineOperand::CreateReg(llvm::AMDGPU::EXEC, false, true));
  }

  return llvm::Error::success();
}

static llvm::Error populateMF(const InstructionTraces &MFTrace,
                              llvm::MachineFunction &MF) {
  llvm::LLVMContext &Ctx = MF.getFunction().getContext();
  llvm::MachineBasicBlock *CurrentMBB = MF.CreateMachineBasicBlock();

  const auto &TM =
      *reinterpret_cast<const llvm::GCNTargetMachine *>(&MF.getTarget());

  MF.push_back(CurrentMBB);

  const llvm::MCRegisterInfo &MRI = *TM.getMCRegisterInfo();

  std::unique_ptr<llvm::MCInstPrinter> IP(TM.getTarget().createMCInstPrinter(
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
  llvm::DenseMap<uint64_t, llvm::MachineBasicBlock *>
      BranchTargetMBBs; // < Set of MBBs that will be the target of the
                        // UnresolvedBranchMIs

  /// The start address of the first trace
  const uint64_t FirstTraceStartAddr = MFTrace.traces_begin()->first.first;
  /// The end address of the last trace
  const uint64_t LastTraceEndAddr = MFTrace.traces_rbegin()->first.second;

  const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  llvm::MachineInstr *EntryInst{nullptr};

  for (const auto &[TraceStartEndAddr, Trace] : MFTrace.traces()) {
    uint64_t CurrentInstrAddr = TraceStartEndAddr.first;
    uint64_t LastTraceInstrAddr = TraceStartEndAddr.second;
    while (CurrentInstrAddr <= LastTraceInstrAddr) {
      const TraceInstr &Inst = Trace->at(CurrentInstrAddr);
      LLVM_DEBUG(llvm::dbgs()
                     << "+++++++++++++++++++++++++++++++++++++++++++++++"
                        "+++++++++++++++++++++++++\n";);
      auto MCInst = Inst.getMCInst();
      const unsigned Opcode = getPseudoOpcodeFromReal(MCInst.getOpcode());
      const llvm::MCInstrDesc &MCID = MCInstInfo.get(Opcode);
      bool IsIndirectBranch = MCID.isIndirectBranch();
      bool IsDirectBranch = MCID.isBranch() && !IsIndirectBranch;
      bool IsDirectBranchTarget =
          MFTrace.isAddressDirectBranchTarget(Inst.getLoadedDeviceAddress());
      LLVM_DEBUG(llvm::dbgs() << "Processing MCInst: ";
                 MCInst.dump_pretty(llvm::dbgs(), IP.get(), " ", &MCContext);
                 llvm::dbgs() << "\n";
                 llvm::dbgs()
                 << llvm::formatv("Device address of the instruction: {0:x}\n",
                                  Inst.getLoadedDeviceAddress());
                 llvm::dbgs()
                 << llvm::formatv("Is branch? {0}\n", MCID.isBranch());
                 llvm::dbgs() << llvm::formatv("Is indirect branch? {0}\n",
                                               IsIndirectBranch););

      if (IsDirectBranchTarget) {
        LLVM_DEBUG(llvm::dbgs() << "Instruction is a branch target.\n";);
        if (!CurrentMBB->empty()) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "Current MBB is not empty; Creating a new basic block\n");
          llvm::MachineBasicBlock *OldMBB = CurrentMBB;
          CurrentMBB = MF.CreateMachineBasicBlock();
          MF.push_back(CurrentMBB);
          OldMBB->addSuccessor(CurrentMBB);
          // Branch targets mark the beginning of an MBB
          LLVM_DEBUG(llvm::dbgs()
                     << "*********************************************"
                        "***************************\n");
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Current MBB is empty; No new block created "
                        "for the branch target.\n");
        }
        BranchTargetMBBs.insert({Inst.getLoadedDeviceAddress(), CurrentMBB});
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "Address {0:x} marks the beginning of MBB idx {1}.\n",
                       Inst.getLoadedDeviceAddress(),
                       CurrentMBB->getNumber()););
      }
      llvm::MachineInstrBuilder Builder =
          llvm::BuildMI(CurrentMBB, llvm::DebugLoc(), MCID);
      auto MDNodeOrErr = TargetMachineInstrMDNode::initializeMDNode(*Builder);
      LUTHIER_RETURN_ON_ERROR(MDNodeOrErr.takeError());
      MDNodeOrErr->setTraceInstrAddress(Ctx, CurrentInstrAddr);

      LLVM_DEBUG(llvm::dbgs() << "Number of operands according to MCID: "
                              << MCID.operands().size() << "\n";
                 llvm::dbgs() << "Populating operands\n";);
      LUTHIER_RETURN_ON_ERROR(
          convertAndAddMCOperandsToMI(MCInst.getOperands(), Builder));

      LLVM_DEBUG(llvm::dbgs() << "Final form of the instruction (not final if "
                                 "it's a direct branch): ";
                 Builder->print(llvm::dbgs()); llvm::dbgs() << "\n");
      if (Inst.getLoadedDeviceAddress() ==
          MFTrace.getInitialEntryPoint().getEntryPointAddress()) {
        EntryInst = Builder.getInstr();
      }
      // Basic Block resolving; We also split blocks further down to "vector"
      // and scalar block to make it easier to deal with predication calculation
      if (MCID.isTerminator()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Instruction is a terminator; Finishing basic block.\n");
        if (IsDirectBranch) {
          LLVM_DEBUG(llvm::dbgs() << "The terminator is a direct branch.\n");
          uint64_t BranchTarget;
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
        // if this is the last instruction in the trace group, no need for
        // creating a new basic block
        if (CurrentInstrAddr != LastTraceEndAddr) {
          LLVM_DEBUG(llvm::dbgs() << "Creating a new basic block.\n");
          llvm::MachineBasicBlock *OldMBB = CurrentMBB;
          CurrentMBB = MF.CreateMachineBasicBlock();
          MF.push_back(CurrentMBB);
          // Don't add the next block to the list of successors if the
          // terminator is an unconditional branch
          if (!MCID.isUnconditionalBranch())
            OldMBB->addSuccessor(CurrentMBB);
          LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                         "Address {0:x} marks the beginning of MBB idx {1}.\n",
                         Inst.getLoadedDeviceAddress(),
                         CurrentMBB->getNumber()););
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "*********************************************"
                      "***************************\n");
      } else if (llvm::MachineInstr *PrevMI = Builder->getPrevNode()) {
        bool IsCurrentMIVector = shouldImplicitReadExec(*Builder);
        bool IsFormerMIVector = shouldImplicitReadExec(*PrevMI);
        bool CurrentMIWritesExecMask =
            Builder->modifiesRegister(llvm::AMDGPU::EXEC, TRI);
        /// TODO: WQM instructions
        bool ShouldSplitCurrentMBB =
            CurrentMIWritesExecMask || IsCurrentMIVector ^ IsFormerMIVector;
        if (ShouldSplitCurrentMBB) {
          llvm::MachineBasicBlock *OldMBB = CurrentMBB;
          CurrentMBB = OldMBB->splitAt(*PrevMI);
          LLVM_DEBUG(llvm::dbgs()
                     << "*********************************************"
                        "***************************\n");
        }
      }
      CurrentInstrAddr += Inst.getSize();
    }
  }
  // Resolve the direct branch and target MIs/MBBs
  LLVM_DEBUG(llvm::dbgs() << "Resolving direct branch MIs\n");
  for (auto &[TargetAddress, BranchMIs] : UnresolvedBranchMIs) {
    LLVM_DEBUG(
        llvm::dbgs() << llvm::formatv(
            "Resolving MIs jumping to target address {0:x}.\n", TargetAddress));
    CurrentMBB = BranchTargetMBBs[TargetAddress];
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        CurrentMBB != nullptr,
        llvm::formatv("Failed to find the MachineBasicBlock associated with "
                      "the branch target address {0:x}.",
                      TargetAddress)));
    for (auto &MI : BranchMIs) {
      LLVM_DEBUG(llvm::dbgs() << "Resolving branch for the instruction ";
                 MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
      MI->addOperand(llvm::MachineOperand::CreateMBB(CurrentMBB));
      MI->getParent()->addSuccessor(CurrentMBB);
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "MBB {0:x} {1} was set as the target of the branch.\n",
                     CurrentMBB, CurrentMBB->getName()));
      LLVM_DEBUG(llvm::dbgs() << "Final branch instruction: ";
                 MI->print(llvm::dbgs()); llvm::dbgs() << "\n");
    }
  }

  /// If the entry address of the trace group is not equal to the address of the
  /// first trace then add a new basic block in the beginning of the MF,
  /// put a jump to the entry instruction
  if (MFTrace.getInitialEntryPoint().getEntryPointAddress() !=
      FirstTraceStartAddr) {
    assert(EntryInst != nullptr && "the entry instruction is nullptr");
    llvm::MachineBasicBlock *EntryMBB = MF.CreateMachineBasicBlock();
    MF.push_front(EntryMBB);

    llvm::MachineInstrBuilder Builder = llvm::BuildMI(
        EntryMBB, llvm::DebugLoc(), MCInstInfo.get(llvm::AMDGPU::S_BRANCH));
    auto MDNodeOrErr = TargetMachineInstrMDNode::initializeMDNode(*Builder);
    LUTHIER_RETURN_ON_ERROR(MDNodeOrErr.takeError());
    MDNodeOrErr->setCanRelaxDirectBranch(Ctx, true);
    Builder->addOperand(
        llvm::MachineOperand::CreateMBB(EntryInst->getParent()));
    EntryMBB->addSuccessor(EntryInst->getParent());
  }

  LLVM_DEBUG(llvm::dbgs() << "*********************************************"
                             "***************************\n");

  /// Freeze the set of reserved register because we will not do any register
  /// allocations here
  MF.getRegInfo().freezeReservedRegs();

  /// Populate the properties of MF
  llvm::MachineFunctionProperties &Properties = MF.getProperties();
  Properties.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Properties.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Properties.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Properties.reset(llvm::MachineFunctionProperties::Property::TracksLiveness);
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

  llvm::MachineModuleInfo &TargetMMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetModule).getMMI();
  auto &TM =
      *reinterpret_cast<const llvm::GCNTargetMachine *>(&TargetMMI.getTarget());

  const MemoryAllocationAccessor &SegAccessor =
      TargetMAM.getResult<MemoryAllocationAnalysis>(TargetModule).getAccessor();

  llvm::MachineFunctionAnalysisManager &MFAM =
      TargetMAM
          .getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(
              TargetModule)
          .getManager();
  llvm::FunctionAnalysisManager &FAM =
      TargetMAM
          .getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();

  EntryPoint InitialEntryPoint =
      TargetMAM.getResult<InitialEntryPointAnalysis>(TargetModule)
          .getInitialEntryPoint();

  llvm::SmallDenseSet<EntryPoint> UnvisitedPointsOfEntry{InitialEntryPoint};

  llvm::SmallDenseSet<EntryPoint> VisitedPointsOfEntry{};

  while (!UnvisitedPointsOfEntry.empty()) {
    EntryPoint CurrentEntryPoint = *UnvisitedPointsOfEntry.begin();

    auto [MF, FuncSymRef] =
        [&]() -> std::pair<llvm::MachineFunction &,
                           std::optional<object::AMDGCNElfSymbolRef>> {
      /// Initialize the function handle associated with the entry point
      if (const auto *KDOnDevice = CurrentEntryPoint.getKernelDescriptor()) {
        const auto &MDParser =
            TargetMAM.getResult<MetadataParserAnalysis>(TargetModule)
                .getParser();

        auto MFOrAndSymOrErr = initKernelEntryPointFunction(
            *KDOnDevice, SegAccessor, MDParser, TargetModule, TM, FAM);
        LUTHIER_CTX_EMIT_ON_ERROR(Ctx, MFOrAndSymOrErr.takeError());

        return *MFOrAndSymOrErr;
      } else {
        auto MFOrAndSymOrErr = initLiftedDeviceFunctionEntry(
            CurrentEntryPoint.getEntryPointAddress(), SegAccessor, TargetModule,
            FAM);
        LUTHIER_CTX_EMIT_ON_ERROR(Ctx, MFOrAndSymOrErr.takeError());
        return *MFOrAndSymOrErr;
      }
    }();
    /// Set the function's entry point as an attribute
    setFunctionEntryPoint(MF.getFunction(), CurrentEntryPoint);
    if (CurrentEntryPoint == InitialEntryPoint) {
      MF.getFunction().addFnAttr(InitialEntryPointAttr);
    }

    /// Add the newly created MF's entry point
    /// Ask for the trace of the instructions for this machine function
    auto &TraceResults = MFAM.getResult<InstructionTracesAnalysis>(MF);
    if (!TraceResults.getTraces()) {
      LUTHIER_CTX_EMIT_ON_ERROR(
          Ctx,
          LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
              "Failed to get trace results for function {0}", MF.getName())));
    }
    /// Populate the current machine function using the trace info we just
    /// obtained
    LUTHIER_CTX_EMIT_ON_ERROR(Ctx, populateMF(*TraceResults.getTraces(), MF));

    /// Invalidate all module-level analysis not related to Functions and
    /// Machine Functions proxies because we just added a new machine function
    /// and they are now stale
    llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
    PA.preserve<llvm::MachineFunctionAnalysisManagerModuleProxy>();
    PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
    TargetMAM.invalidate(TargetModule, PA);

    /// Check if there are call instruction in return blocks of the
    /// machine function; If so, check if there is a function symbol associated
    /// with it. If so, see if the size of the function indicates there is more
    /// code after each call instruction, and add them to the list of
    /// entry points to be visited

    /// Form the Predicated CFG for the current Machine Function and
    /// raise the discovered code to LLVM IR

    /// Re-calculate IP-liveness and IP-reaching definitions
    // (void)TargetMAM.getResult<IndirectBranchResolverAnalysis>(TargetModule);

    UnvisitedPointsOfEntry.erase(CurrentEntryPoint);
    VisitedPointsOfEntry.insert(CurrentEntryPoint);
  }

  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<llvm::MachineFunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
  return PA;
}
} // namespace luthier