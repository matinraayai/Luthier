#include "tooling_common/InstrumentationStateValueManager.hpp"
#include "common/Error.hpp"
#include <SIMachineFunctionInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/Function.h>
#include <luthier/LiftedRepresentation.h>

namespace luthier {

/// A set of \c LiftedKernelArgumentManager::KernelArgumentType that have
/// a \c llvm::AMDGPUFunctionArgInfo::PreloadedValue equivalent (i.e. can/must
/// be preloaded into register values)
static const llvm::SmallDenseMap<KernelArgumentType,
                                 llvm::AMDGPUFunctionArgInfo::PreloadedValue>
    ToLLVMRegArgMap{
        {PRIVATE_SEGMENT_BUFFER,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::PRIVATE_SEGMENT_BUFFER},
        {DISPATCH_PTR,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::DISPATCH_PTR},
        {QUEUE_PTR, llvm::AMDGPUFunctionArgInfo::PreloadedValue::QUEUE_PTR},
        {KERNARG_SEGMENT_PTR,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::KERNARG_SEGMENT_PTR},
        {DISPATCH_ID, llvm::AMDGPUFunctionArgInfo::PreloadedValue::DISPATCH_ID},
        {FLAT_SCRATCH_INIT,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::FLAT_SCRATCH_INIT},
        {PRIVATE_SEGMENT_SIZE,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::PRIVATE_SEGMENT_SIZE},
        {PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::
             PRIVATE_SEGMENT_WAVE_BYTE_OFFSET},
        {WORK_ITEM_X,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_X},
        {WORK_ITEM_Y,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_Y},
        {WORK_ITEM_Z,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_Z},
    };

/// A set of \c LiftedKernelArgumentManager::KernelArgumentType that have a
/// hidden argument type equivalent
static const llvm::SmallDenseMap<hsa::md::ValueKind, KernelArgumentType>
    ToHiddenArgMap{
        {hsa::md::ValueKind::HiddenQueuePtr, QUEUE_PTR},
        {hsa::md::ValueKind::HiddenGlobalOffsetX, GLOBAL_OFFSET_X},
        {hsa::md::ValueKind::HiddenGlobalOffsetY, GLOBAL_OFFSET_Y},
        {hsa::md::ValueKind::HiddenGlobalOffsetZ, GLOBAL_OFFSET_Z},
        {hsa::md::ValueKind::HiddenPrintfBuffer, PRINT_BUFFER},
        {hsa::md::ValueKind::HiddenHostcallBuffer, HOSTCALL_BUFFER},
        {hsa::md::ValueKind::HiddenDefaultQueue, DEFAULT_QUEUE},
        {hsa::md::ValueKind::HiddenCompletionAction, COMPLETION_ACTION},
        {hsa::md::ValueKind::HiddenMultiGridSyncArg, MULTIGRID_SYNC},
        {hsa::md::ValueKind::HiddenBlockCountX, BLOCK_COUNT_X},
        {hsa::md::ValueKind::HiddenBlockCountY, BLOCK_COUNT_Y},
        {hsa::md::ValueKind::HiddenBlockCountZ, BLOCK_COUNT_Z},
        {hsa::md::ValueKind::HiddenGroupSizeX, GROUP_SIZE_X},
        {hsa::md::ValueKind::HiddenGroupSizeY, GROUP_SIZE_Y},
        {hsa::md::ValueKind::HiddenGroupSizeZ, GROUP_SIZE_Z},
        {hsa::md::ValueKind::HiddenRemainderX, REMAINDER_X},
        {hsa::md::ValueKind::HiddenRemainderY, REMAINDER_Y},
        {hsa::md::ValueKind::HiddenRemainderZ, REMAINDER_Z},
        {hsa::md::ValueKind::HiddenGridDims, GRID_DIMS},
        {hsa::md::ValueKind::HiddenHeapV1, HEAP_V1},
        {hsa::md::ValueKind::HiddenDynamicLDSSize, DYNAMIC_LDS_SIZE},
        {hsa::md::ValueKind::HiddenPrivateBase, PRIVATE_BASE},
        {hsa::md::ValueKind::HiddenSharedBase, SHARED_BASE}};

//static llvm::SmallDenseMap<KernelArgumentType, uint8_t> HiddenArgToLaneIDValues{
//    {}
//    // Hidden Args:
//    {static_cast<const uint8_t>(
//         luthier::hsa::md::ValueKind::HiddenGlobalOffsetY),
//     13},
//    {static_cast<const uint8_t>(
//         luthier::hsa::md::ValueKind::HiddenGlobalOffsetZ),
//     14},
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenNone),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenPrintfBuffer),
//    static_cast<const uint8_t>(
//        luthier::hsa::md::ValueKind::HiddenHostcallBuffer),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenDefaultQueue),
//    static_cast<const uint8_t>(
//        luthier::hsa::md::ValueKind::HiddenCompletionAction),
//    static_cast<const uint8_t>(
//        luthier::hsa::md::ValueKind::HiddenMultiGridSyncArg),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenBlockCountX),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenBlockCountY),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenBlockCountZ),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenGroupSizeX),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenGroupSizeY),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenGroupSizeZ),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenRemainderX),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenRemainderY),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenRemainderZ),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenGridDims),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenHeapV1),
//    static_cast<const uint8_t>(
//        luthier::hsa::md::ValueKind::HiddenDynamicLDSSize),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenPrivateBase),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenSharedBase),
//    static_cast<const uint8_t>(luthier::hsa::md::ValueKind::HiddenQueuePtr),
//    //
//};

InstrumentationStateValueManager::InstrumentationStateValueManager(
    LiftedRepresentation &LR,
    llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap,
    llvm::Module &InstrumentationModule,
    llvm::MachineModuleInfo &InstrumentationMMI)
    : LR(LR), Module(InstrumentationModule), MMI(InstrumentationMMI),
      MIToHookMap(MIToHookMap) {
  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    if (auto *KernelSymbol =
            llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(FuncSymbol)) {
      Kernel = std::make_pair(KernelSymbol, MF);
      const auto &MD = KernelSymbol->getKernelMetadata();
      if (MD.Args.has_value()) {
        for (const auto &Arg : *MD.Args) {
          if (hsa::md::ValueKind::HiddenArgKindBegin <= Arg.ValueKind &&
              hsa::md::ValueKind::HiddenArgKindEnd >= Arg.ValueKind) {
            HiddenKernArgsToOffsetMap.insert(
                {ToHiddenArgMap.at(Arg.ValueKind), Arg.Offset});
          }
        }
      }

    }
  }
}

bool InstrumentationStateValueManager::hasAccessToArg(
    KernelArgumentType Arg) const {
  bool ContainsArg{false};
  if (ToLLVMRegArgMap.contains(Arg))
    ContainsArg |=
        Kernel.second->getInfo<llvm::SIMachineFunctionInfo>()->getPreloadedReg(
            ToLLVMRegArgMap.at(Arg)) != llvm::MCRegister();
  if (HiddenKernArgsToOffsetMap.contains(Arg))
    ContainsArg = true;
  return ContainsArg;
}

llvm::MCRegister
InstrumentationStateValueManager::getArgReg(KernelArgumentType ArgReg) {
  return Kernel.second->getInfo<llvm::SIMachineFunctionInfo>()->getPreloadedReg(
      ToLLVMRegArgMap.at(ArgReg));
}

static llvm::MachineFunction &
createEmptyMachineFunction(llvm::Module &M, llvm::MachineModuleInfo &MMI,
                           llvm::StringRef Name) {
  llvm::Type *const ReturnType = llvm::Type::getVoidTy(M.getContext());
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(ReturnType, {}, false);
  auto *PrologueFunction = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::ExternalLinkage, Name, M);
  return MMI.getOrCreateMachineFunction(*PrologueFunction);
}


void LoadSpilledSGPRToVGPR(llvm::MachineBasicBlock::iterator InsertionPoint,
                           llvm::MCRegister DestSGPR, llvm::MCRegister SrcVGPR,
                           llvm::SIRegisterInfo *TRI,
                           llvm::TargetInstrInfo *TII,
                           llvm::MachineRegisterInfo &MRI,
                           unsigned int LaneID) {
  auto SGPRRegClass = TRI->getPhysRegBaseClass(DestSGPR);
  // First copy the SrcSGPR to a virtual register to copy the values into
  auto IntermediateVirtualSGPR = MRI.createVirtualRegister(SGPRRegClass);
  // Get 32-bit sub registers of this SGPR
  auto SplitRegs = TRI->getRegSplitParts(SGPRRegClass, 32);
  llvm::SmallVector<llvm::Register, 4> SubRegs;
  for (auto [i, SubReg] : llvm::enumerate(SplitRegs)) {
    auto IntermediateVirtualSGPR32 =
        MRI.createVirtualRegister(&llvm::AMDGPU::SGPR_32RegClass);
    llvm::BuildMI(*InsertionPoint->getParent(), InsertionPoint,
                  llvm::DebugLoc(), TII->get(llvm::AMDGPU::V_READLANE_B32),
                  IntermediateVirtualSGPR32)
        .addReg(SrcVGPR, 0)
        .addImm(LaneID + i);
    SubRegs.push_back(IntermediateVirtualSGPR32);
  }
  // Create a reg sequence instruction to merge the values together
  auto MIB = llvm::BuildMI(
      *InsertionPoint->getParent(), InsertionPoint, llvm::DebugLoc(),
      TII->get(llvm::AMDGPU::REG_SEQUENCE), IntermediateVirtualSGPR);
  for (auto [SplitReg, SubReg] : llvm::zip(SplitRegs, SubRegs)) {
    MIB.addReg(SubReg, llvm::RegState::Kill, SplitReg).addImm(SubReg);
  }
  // Finally, copy the merged values to the dest SGPR
  llvm::BuildMI(*InsertionPoint->getParent(), InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::COPY), DestSGPR)
      .addReg(IntermediateVirtualSGPR, llvm::RegState::Kill);
}

void SpillSGPRToVGPR(llvm::MachineBasicBlock::iterator InsertionPoint,
                     llvm::MCRegister SrcSGPR, llvm::MCRegister DestVGPR,
                     llvm::SIRegisterInfo *TRI, llvm::TargetInstrInfo *TII,
                     llvm::MachineRegisterInfo &MRI, unsigned long LaneID) {
  auto SGPRRegClass = TRI->getPhysRegBaseClass(SrcSGPR);
  // First copy the SrcSGPR to a virtual register
  auto IntermediateVirtualSGPR = MRI.createVirtualRegister(SGPRRegClass);
  llvm::BuildMI(*InsertionPoint->getParent(), InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::COPY), IntermediateVirtualSGPR)
      .addReg(SrcSGPR, llvm::RegState::Kill);

  auto SplitRegs = TRI->getRegSplitParts(SGPRRegClass, 32);

  for (auto [i, SplitReg] : llvm::enumerate(SplitRegs)) {
    auto MIB = llvm::BuildMI(*InsertionPoint->getParent(), InsertionPoint,
                             llvm::DebugLoc(),
                             TII->get(llvm::AMDGPU::V_WRITELANE_B32), DestVGPR);
    MIB.addReg(IntermediateVirtualSGPR, 0, SplitReg)
        .addImm(LaneID + i)
        .addReg(DestVGPR);
  }
}

void SpillSGPRToVGPR(llvm::MachineBasicBlock::iterator InsertionPoint,
                     llvm::Register SrcSGPR, llvm::MCRegister DestVGPR,
                     const llvm::SIRegisterInfo *TRI,
                     const llvm::TargetInstrInfo *TII,
                     llvm::MachineRegisterInfo &MRI, unsigned long LaneID) {
  auto SGPRRegClass = TRI->getRegClassForReg(MRI, SrcSGPR);

  auto SplitRegs = TRI->getRegSplitParts(SGPRRegClass, 32);

  for (auto [i, SplitReg] : llvm::enumerate(SplitRegs)) {
    auto MIB = llvm::BuildMI(*InsertionPoint->getParent(), InsertionPoint,
                             llvm::DebugLoc(),
                             TII->get(llvm::AMDGPU::V_WRITELANE_B32), DestVGPR);
    MIB.addReg(SrcSGPR, 0, SplitReg).addImm(LaneID + i).addReg(DestVGPR);
  }
}

/// Generates a prologue code for reading information regarding the kernel
/// arguments and storing it in the state value
/// \param [in] KernArgSGPRPair Where the address of the kernel argument buffer
/// is stored
/// \param [in] InsertionPoint The MI iterator pointing to the beginning of
/// where the kernel argument loading code will be emitted
/// \param [in] OriginalKernelArgs The original arguments metadata of the kernel
/// \param [in] EnsuredArgs The set of arguments that needs to be ensured by the
/// kernel prologue
/// \param [in] AdditionalUserArgs A list of additional arguments the tool
/// writer wants to pass to the kernel, added after the original kernel's
/// arguments
/// \param [out] FinalLuthierArgsMetadata the Luthier argument list needed
/// to load the arguments after the kernel is instrumented
static void insertKernArgSavePrologue(
    llvm::MCRegister KernArgSGPRPair, llvm::MCRegister ValueRegVGPR,
    llvm::MachineBasicBlock::iterator InsertionPoint,
    llvm::ArrayRef<hsa::md::Kernel::Arg::Metadata> OriginalKernelArgs,
    const llvm::SmallDenseSet<KernelArgumentType, 4> &EnsuredArgs,
    const llvm::SmallVectorImpl<llvm::Type *> &AdditionalUserArgs,
    llvm::SmallVectorImpl<hsa::md::Kernel::Metadata>
        &FinalLuthierArgsMetadata) {
  auto *ParentMF = InsertionPoint->getParent()->getParent();
  auto *TII = ParentMF->getSubtarget().getInstrInfo();
  auto &MRI = ParentMF->getRegInfo();
  auto *TRI = ParentMF->getSubtarget().getRegisterInfo();
  auto GetLoadMCID = [&](unsigned int Size) -> llvm::MCInstrDesc {
    switch (Size) {
    case 2:
    case 4:
      return TII->get(llvm::AMDGPU::S_LOAD_DWORD_IMM);
    case 8:
      return TII->get(llvm::AMDGPU::S_LOAD_DWORDX2_IMM);
    case 16:
      return TII->get(llvm::AMDGPU::S_LOAD_DWORDX8_IMM);
    default:
      llvm_unreachable("Invalid size");
    }
  };
  auto getSGPRClass = [&](unsigned int Size) {
    switch (Size) {
    case 2:
    case 4:
      return &llvm::AMDGPU::SGPR_32RegClass;
    case 8:
      return &llvm::AMDGPU::SGPR_64RegClass;
    case 16:
      return &llvm::AMDGPU::SGPR_128RegClass;
    default:
      llvm_unreachable("Invalid size");
    }
  };

  auto *MBB = InsertionPoint->getParent();
  // We first start by looking at the original kernel's hidden arguments to
  // see if we can reuse any of it
  for (const auto &OriginalKernelArg : OriginalKernelArgs) {
    if (hsa::md::ValueKind::HiddenArgKindBegin <= OriginalKernelArg.ValueKind &&
        OriginalKernelArg.ValueKind <= hsa::md::ValueKind::HiddenArgKindEnd) {
      auto ArgKind = ToHiddenArgMap.at(OriginalKernelArg.ValueKind);
      // If any of the intrinsics requested this argument, then load it and
      // store it in the value register
      if (EnsuredArgs.contains(ArgKind)) {
        auto VirtSReg =
            MRI.createVirtualRegister(getSGPRClass(OriginalKernelArg.Size));
        llvm::BuildMI(*MBB, InsertionPoint, llvm::DebugLoc(),
                      GetLoadMCID(OriginalKernelArg.Size))
            .addReg(VirtSReg)
            .addReg(KernArgSGPRPair)
            .addImm(OriginalKernelArg.Offset);
      }
    }
  }
  // After that, we move on to loading the Luthier hidden arguments

  //

  // Find out where the last argument is in the kernel
  auto OriginalKernArgSize =
      OriginalKernelArgs.back().Offset + OriginalKernelArgs.back().Size;

  // The first hidden Luthier argument will be the unswizzled offset of the
  // Luthier stack
}

llvm::Error InstrumentationStateValueManager::insertPrologue() {
  // Record the original place of the reg arguments for creating a
  // pre-kernel later down the line
  llvm::SmallDenseMap<KernelArgumentType, llvm::MCRegister, 16>
      OriginalSGPRArgsLocations;
  for (std::underlying_type<KernelArgumentType>::type ArgKind =
           ALWAYS_IN_SGPR_BEGIN;
       ArgKind <= EITHER_IN_SGPR_OR_HIDDEN_END; ArgKind++) {
    if (auto Reg = getArgReg(KernelArgumentType(ArgKind))) {
      OriginalSGPRArgsLocations.insert({KernelArgumentType(ArgKind), Reg});
    }
  }
  //============================================================================
  // Enable the SGPR args in the Machine Function Info that were require to
  // be enabled, either because Luthier needs them or because the user
  // requested them
  // TODO: This step might need to happen later down the line in case we don't
  // need to enable the private segment buffer
  auto &MFI = *Kernel.second->getInfo<llvm::SIMachineFunctionInfo>();
  auto TRI = *reinterpret_cast<const llvm::SIRegisterInfo *>(
      Kernel.second->getSubtarget().getRegisterInfo());

  // For now always enable access to the private segment buffer
  if (!hasAccessToArg(PRIVATE_SEGMENT_BUFFER))
    MFI.addPrivateSegmentBuffer(TRI);

  // Always ensure access to the kernel argument buffer
  if (!hasAccessToArg(KERNARG_SEGMENT_PTR))
    MFI.addKernargSegmentPtr(TRI);

  // Enable access to kernel dispatch ID if the user has requested it
  if (EnsuredArgs.contains(DISPATCH_ID) && !hasAccessToArg(DISPATCH_ID))
    MFI.addDispatchID(TRI);

  // For now, always ensure access to flat scratch init register
  if (!hasAccessToArg(FLAT_SCRATCH_INIT))
    MFI.addFlatScratchInit(TRI);

  // For now, always ensure access to private segment wave offset
  if (!hasAccessToArg(PRIVATE_SEGMENT_WAVE_BYTE_OFFSET))
    MFI.addPrivateSegmentWaveByteOffset();

  // The other arguments will have to be passed on a hidden buffer if not
  // already enabled
  //============================================================================
  // Prologue/Epilogue code generation +
  // register move code generation (if necessary)

  auto *TII = Kernel.second->getSubtarget().getInstrInfo();
  auto &CopyMCID = TII->get(llvm::AMDGPU::COPY);

  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    auto IsKernel = llvm::isa<hsa::LoadedCodeObjectKernel>(FuncSymbol);
    auto &ValueAndFSRegLocations = ValueStateRegAndFlatScratchIntervals.at(MF);
    auto &PrologueMachineFunction = createEmptyMachineFunction(
        Module, MMI, llvm::formatv("PrologueFor{0}", MF->getName()).str());
    auto *EntryValSeg = getValueSegmentForInstr(*MF, *MF->begin()->begin());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(EntryValSeg != nullptr));
    // The VGPR we will store the value register in the first instruction
    llvm::MCRegister ValRegisterLocationOnEntry =
        EntryValSeg->getValueRegisterLocation();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ValRegisterLocationOnEntry != 0));
    PrologueAndRegStateMoveCode.insert(
        {&(*MF->begin()->begin()), &PrologueMachineFunction});
    // Generate prologue for the kernel
    // First MBB of the function
    auto EntryMBB = MF->begin();
    if (IsKernel) {
      // Generate code for first reading the SGPR kernel argument
      // Load the Private segment buffer
      auto PSBVirtualReg =
          PrologueMachineFunction.getRegInfo().createVirtualRegister(
              &llvm::AMDGPU::SGPR_128RegClass);
      llvm::BuildMI(*EntryMBB, EntryMBB->begin(), llvm::DebugLoc(), CopyMCID,
                    PSBVirtualReg)
          .addReg(getArgReg(PRIVATE_SEGMENT_BUFFER), 0);
      // Load the private wave offset
      auto PWOVirtualReg =
          PrologueMachineFunction.getRegInfo().createVirtualRegister(
              &llvm::AMDGPU::SGPR_32RegClass);
      llvm::BuildMI(*EntryMBB, EntryMBB->begin(), llvm::DebugLoc(), CopyMCID,
                    PSBVirtualReg)
          .addReg(getArgReg(PRIVATE_SEGMENT_WAVE_BYTE_OFFSET), 0);
      // Add PSBWO to the S0 of PSB, S1 for the carry add

      auto ScratchRsrcSub0 = TRI.getSubReg(PSBVirtualReg, llvm::AMDGPU::sub0);
      auto ScratchRsrcSub1 = TRI.getSubReg(PSBVirtualReg, llvm::AMDGPU::sub1);

      llvm::BuildMI(*EntryMBB, EntryMBB->begin(), llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::S_ADD_U32), ScratchRsrcSub0)
          .addReg(ScratchRsrcSub0)
          .addReg(PWOVirtualReg)
          .addReg(PSBVirtualReg, llvm::RegState::ImplicitDefine);
      llvm::BuildMI(*EntryMBB, EntryMBB->begin(), llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::S_ADDC_U32), ScratchRsrcSub1)
          .addReg(ScratchRsrcSub1)
          .addImm(0)
          .addReg(PSBVirtualReg, llvm::RegState::ImplicitDefine)
          ->getOperand(3).setIsDead(); // Mark SCC as dead.
      // Store the PSB in lane 7
      SpillSGPRToVGPR(EntryMBB->begin(), PSBVirtualReg,
                      ValRegisterLocationOnEntry, &TRI, TII, MF->getRegInfo(),
                      7);

//      // Load the
//
//      for (std::underlying_type<KernelArgumentType>::type ArgKind =
//               ALWAYS_IN_SGPR_BEGIN;
//           ArgKind <= EITHER_IN_SGPR_OR_HIDDEN_END; ArgKind++) {
//        // If an argument is not enabled in the final kernel, then we don't
//        // worry about it
//        if (auto SGPRArgReg = getArgReg(KernelArgumentType(ArgKind))) {
//          // Generate a copy from the current location to a virtual one
//          auto VirtualSReg = (TRI.getPhysRegBaseClass(SGPRArgReg));
//          llvm::BuildMI(*EntryMBB, EntryMBB->begin(), llvm::DebugLoc(),
//                        CopyMCID)
//              .addReg(VirtualSReg, llvm::RegState::Define)
//              .addReg(SGPRArgReg, llvm::RegState::Kill);
//          // Store the arg in the value register
//        }
//      }

    }
    // Generate prologue for machine functions if necessary
    else if (!OnlyKernelNeedsPrologue) {
    }

    // Generate logic to move around the value state register and the
    // instrumentation flat scratch
  }
  return llvm::Error::success();
}

void InstrumentationStateValueManager::ensureAccessToArg(
    KernelArgumentType Arg) {
  EnsuredArgs.insert(Arg);
}

} // namespace luthier