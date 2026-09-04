//===-- RefKernelSupport.cpp ------------------------------------*- C++ -*-===//
//
// Definitions of the shared reference-kernel scaffold (setupScaffold,
// finalizeMF). The register reservation and MIR emit primitives are inline in
// RefKernelSupport.h.
//
//===----------------------------------------------------------------------===//
#include "RefKernelSupport.h"

#include <GCNSubtarget.h>
#include <SIMachineFunctionInfo.h>

#include <llvm/IR/Function.h>

namespace luthier::test {

unsigned waveSize(const llvm::TargetMachine &TM) {
  // Reflects the subtarget's wavefront feature (target-cpu default, or an
  // explicit +wavefrontsize32/64 in the TM feature string).
  return TM.getMCSubtargetInfo().hasFeature(llvm::AMDGPU::FeatureWavefrontSize32)
             ? 32u
             : 64u;
}

uint32_t vSharpWord3(const llvm::MCSubtargetInfo &STI) {
  /// DST_SEL_{X,Y,Z,W} in bits [11:0], identity swizzle (X,Y,Z,W = 4,5,6,7).
  /// Same position in every generation, and ignored by the raw (non-format)
  /// buffer ops; it only matters for the *_FORMAT_* / MTBUF paths.
  constexpr uint32_t DstSelIdentity = 4u | (5u << 3) | (6u << 6) | (7u << 9);

  if (llvm::AMDGPU::isGFX10Plus(STI)) {
    /// GFX10+: one 7-bit unified FORMAT at [18:12] replaces GFX9's
    /// num_format/data_format pair, plus RESOURCE_LEVEL (bit 24) and
    /// OOB_SELECT ([29:28]). OOB_SELECT = 3 means "raw buffer: out of bounds
    /// iff offset >= num_records", which is the rule these reference kernels
    /// rely on; leaving it at 0 selects the structured-buffer rule and makes
    /// the hardware treat every access as out of bounds.
    return DstSelIdentity |
           (static_cast<uint32_t>(llvm::AMDGPU::UfmtGFX10::UFMT_32_FLOAT)
            << 12) |
           (1u << 24) | // RESOURCE_LEVEL = 1
           (3u << 28);  // OOB_SELECT = 3 (raw)
  }
  /// GFX9 and earlier: NUM_FORMAT = 7 (float) at [14:12], DATA_FORMAT = 4
  /// (32-bit) at [18:15]. Together with the identity swizzle this is the
  /// familiar 0x00027FAC raw-buffer descriptor word.
  return DstSelIdentity | (7u << 12) | (4u << 15);
}

int64_t mtbufFormat(const llvm::MCSubtargetInfo &STI, unsigned Dfmt,
                    unsigned Nfmt) {
  if (llvm::AMDGPU::isGFX10Plus(STI))
    return llvm::AMDGPU::MTBUFFormat::convertDfmtNfmt2Ufmt(Dfmt, Nfmt, STI);
  return static_cast<int64_t>(Dfmt) | (static_cast<int64_t>(Nfmt) << 4);
}

llvm::Expected<Scaffold> setupScaffold(llvm::TargetMachine &TM,
                                       KernelMFContext &KCtx,
                                       unsigned NumI32Inputs,
                                       unsigned NumPtrArgs,
                                       bool EnableFlatScratch,
                                       unsigned FlatWorkGroupSize) {
  const bool PerLane = FlatWorkGroupSize > 1;
  KCtx.Ctx = std::make_unique<llvm::LLVMContext>();
  llvm::LLVMContext &Ctx = *KCtx.Ctx;
  KCtx.Mod = std::make_unique<llvm::Module>("ref_kernel", Ctx);
  KCtx.Mod->setTargetTriple(llvm::Triple("amdgcn-amd-amdhsa"));
  KCtx.Mod->setDataLayout(TM.createDataLayout());

  llvm::SmallVector<llvm::Type *, 6> Params;
  for (unsigned I = 0; I < std::max(NumPtrArgs, 1u); ++I)
    Params.push_back(llvm::PointerType::get(Ctx, /*AddressSpace=*/1));
  for (unsigned I = 0; I < NumI32Inputs; ++I)
    Params.push_back(llvm::Type::getInt32Ty(Ctx));
  auto *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), Params, false);
  auto *F = llvm::Function::Create(FTy, llvm::GlobalValue::ExternalLinkage,
                                   KCtx.KernelName, KCtx.Mod.get());
  F->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  F->addFnAttr("amdgpu-flat-work-group-size",
               PerLane ? (llvm::Twine(FlatWorkGroupSize) + "," +
                          llvm::Twine(FlatWorkGroupSize)).str()
                       : std::string("1,1"));
  F->addFnAttr("uniform-work-group-size", "true");
  F->addFnAttr("amdgpu-implicitarg-num-bytes", "0");
  // Register-file sizes, read by TraceFunctionTranslator::initRegFileLayouts when the
  // translation path raises this kernel back to LLVM IR (must be non-zero). The
  // reference kernels stay well within these counts.
  //
  // On GFX10+ the SGPR count must also cover VCC, not just the SGPRs the kernel
  // body names. The translator sizes its SGPR register file as `amdgpu-num-sgpr`
  // DWORDs and resolves a register to a file by hardware encoding index. GFX9
  // and earlier re-home VCC into the top of the declared allocation, so any
  // reasonable count works there; on GFX10+ that aliasing is gone and VCC keeps
  // its real index (VCC_LO=106, VCC_HI=107). With a smaller count VCC lands past
  // the end of the SGPR file and matches no file at all, so getRegFileKey falls
  // through to an llvm_unreachable -- which in a Release/NDEBUG build is UB, and
  // in practice silently loses every VCC write (e.g. a V_CMP's implicit-def,
  // leaving a later `$vcc_lo` read to fold to 0). 108 is the smallest count that
  // spans through VCC_HI.
  //
  // This is deliberately not applied on GFX9: 108 exceeds that generation's 102
  // addressable SGPRs (GFX10+ allows 106), and asking for more than the target
  // has is not something the reference kernels should do when the smaller,
  // long-standing count is already correct there.
  F->addFnAttr("amdgpu-num-sgpr",
               llvm::AMDGPU::isGFX10Plus(TM.getMCSubtargetInfo()) ? "108" : "64");
  F->addFnAttr("amdgpu-num-vgpr", "64");
  for (const char *Attr :
       {"amdgpu-no-dispatch-ptr", "amdgpu-no-queue-ptr", "amdgpu-no-dispatch-id",
        "amdgpu-no-implicitarg-ptr", "amdgpu-no-workgroup-id-x",
        "amdgpu-no-workgroup-id-y", "amdgpu-no-workgroup-id-z",
        "amdgpu-no-workitem-id-x", "amdgpu-no-workitem-id-y",
        "amdgpu-no-workitem-id-z", "amdgpu-no-heap-ptr",
        "amdgpu-no-hostcall-ptr", "amdgpu-no-lds-kernel-id",
        "amdgpu-no-multigrid-sync-arg", "amdgpu-no-completion-action",
        "amdgpu-no-default-queue"}) {
    // A per-lane kernel reads the workitem id (x) from v0, so leave that use
    // enabled (do not suppress it).
    if (PerLane && llvm::StringRef(Attr) == "amdgpu-no-workitem-id-x")
      continue;
    F->addFnAttr(Attr);
  }
  // Flat-scratch init is required for SCRATCH_* (flat-scratch) instructions;
  // suppressed otherwise so the kernarg pointer lands right after the private
  // segment buffer. The +enable-flat-scratch subtarget feature switches the
  // scratch ABI from the private-segment *buffer* (s[0:3]) to flat scratch:
  // no private_segment_buffer user SGPR, a flat_scratch_init user SGPR pair, and
  // a PEI-emitted `FLAT_SCR = flat_scratch_init + wave_offset` prologue. Getting
  // this ABI right (vs. emitting SCRATCH_* under the buffer ABI) is what makes
  // HSA supply flat_scratch_init at dispatch instead of page-faulting.
  if (EnableFlatScratch) {
    // Setting target-features overrides the subtarget default, so re-assert the
    // wavefront size here to keep this kernel in the same wave mode as the rest.
    std::string TF = "+enable-flat-scratch";
    TF += waveSize(TM) == 32 ? ",+wavefrontsize32,-wavefrontsize64"
                             : ",+wavefrontsize64,-wavefrontsize32";
    F->addFnAttr("target-features", TF);
  } else {
    F->addFnAttr("amdgpu-no-flat-scratch-init");
  }
  llvm::ReturnInst::Create(Ctx, llvm::BasicBlock::Create(Ctx, "entry", F));

  KCtx.MMIWP = std::make_unique<llvm::MachineModuleInfoWrapperPass>(&TM);
  llvm::MachineFunction &MF =
      KCtx.MMIWP->getMMI().getOrCreateMachineFunction(*F);
  KCtx.MF = &MF;

  const auto &ST = MF.getSubtarget<llvm::GCNSubtarget>();
  const llvm::SIInstrInfo &TII = *ST.getInstrInfo();
  const llvm::SIRegisterInfo &TRI = *ST.getRegisterInfo();
  auto *MFI = MF.getInfo<llvm::SIMachineFunctionInfo>();

  const auto &U = MFI->getUserSGPRInfo();
  if (U.hasImplicitBufferPtr())
    return makeError("implicit buffer pointer ABI unsupported");
  if (U.hasPrivateSegmentBuffer())
    MFI->addPrivateSegmentBuffer(TRI);
  if (U.hasDispatchPtr())
    MFI->addDispatchPtr(TRI);
  if (U.hasQueuePtr())
    MFI->addQueuePtr(TRI);
  if (!U.hasKernargSegmentPtr())
    return makeError("kernel has no kernarg segment pointer");
  llvm::Register KernargReg = MFI->addKernargSegmentPtr(TRI);

  // Flat-scratch init: a user SGPR pair (must be allocated before any system
  // SGPR) plus the private-segment wave byte offset (a system SGPR). PEI emits
  // FLAT_SCR = flat_scratch_init + wave_offset when FLAT_SCR is used.
  llvm::Register FlatScrInitReg, WaveOffReg;
  if (EnableFlatScratch) {
    if (!U.hasFlatScratchInit())
      return makeError("flat scratch init requested but not enabled by the ABI");
    FlatScrInitReg = MFI->addFlatScratchInit(TRI);
    if (MFI->hasPrivateSegmentWaveByteOffset())
      WaveOffReg = MFI->addPrivateSegmentWaveByteOffset();
  }

  MF.getRegInfo().addLiveIn(KernargReg);
  if (FlatScrInitReg)
    MF.getRegInfo().addLiveIn(FlatScrInitReg);
  if (WaveOffReg)
    MF.getRegInfo().addLiveIn(WaveOffReg);
  // The VGPR workitem id (x) lives in v0 at entry for a per-lane dispatch.
  // Registering it in ArgInfo is what instruction selection would normally do
  // (allocateSpecialInputVGPRs); a live-in alone records that v0 is defined on
  // entry but not *what* it holds. Consumers that ask the ABI where the
  // workitem id lives -- notably TraceFunctionTranslator, which seeds it from
  // llvm.amdgcn.workitem.id.x -- find nothing and fall back to poison, which
  // turns every per-lane address computation into a wild pointer.
  if (PerLane) {
    MF.getRegInfo().addLiveIn(llvm::AMDGPU::VGPR0);
    MFI->setWorkItemIDX(
        llvm::ArgDescriptor::createRegister(llvm::AMDGPU::VGPR0));
  }
  auto *BB = MF.CreateMachineBasicBlock();
  MF.push_back(BB);
  BB->addLiveIn(KernargReg);
  if (FlatScrInitReg)
    BB->addLiveIn(FlatScrInitReg);
  if (WaveOffReg)
    BB->addLiveIn(WaveOffReg);
  if (PerLane)
    BB->addLiveIn(llvm::AMDGPU::VGPR0);

  llvm::MCRegister OutPtrReg = TRI.getMatchingSuperReg(
      sgpr32(OutPtrSGPRIdx), llvm::AMDGPU::sub0, &llvm::AMDGPU::SGPR_64RegClass);
  if (!OutPtrReg)
    return makeError("failed to form the output pointer SGPR pair");

  return Scaffold{&MF, &TII, &TRI, BB, KernargReg, OutPtrReg};
}

void finalizeMF(llvm::MachineFunction &MF) {
  auto &Props = MF.getProperties();
  Props.reset(llvm::MachineFunctionProperties::Property::IsSSA);
  Props.set(llvm::MachineFunctionProperties::Property::NoPHIs);
  Props.set(llvm::MachineFunctionProperties::Property::NoVRegs);
  Props.set(llvm::MachineFunctionProperties::Property::Selected);
  Props.set(llvm::MachineFunctionProperties::Property::TracksLiveness);
  MF.getRegInfo().freezeReservedRegs();
}

} // namespace luthier::test
