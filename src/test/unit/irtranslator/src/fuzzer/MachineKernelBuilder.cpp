//===-- MachineKernelBuilder.cpp --------------------------------*- C++ -*-===//
//
// Reference path front end: dispatch an opcode to the matching instruction-
// class builder (instructions/VOP.cpp, instructions/DS.cpp) and emit the
// resulting MachineFunction to a loadable code object.
//
// The per-class prolog/epilog logic lives in instructions/*.cpp; the shared
// scaffold and MIR emit primitives live in RefKernelSupport.{h,cpp}.
//
//===----------------------------------------------------------------------===//
#include "MachineKernelBuilder.h"

#include "InstructionBuilders.h"
#include "RefKernelSupport.h" // makeError

#include "luthier/Linker/Linker.h"
#include "luthier/ToolCodeGen/TraceFunctionTranslator.h"

#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

namespace luthier::test {

//===----------------------------------------------------------------------===//
// Instruction-class builder table (first match wins).
//===----------------------------------------------------------------------===//
static bool matchesDS(const llvm::MCInstrDesc &Desc, const InstrProfile &) {
  return Desc.TSFlags & llvm::SIInstrFlags::DS;
}
static bool matchesSOP(const llvm::MCInstrDesc &Desc, const InstrProfile &) {
  // Scalar ALU encodings. Checked before VOP because these are also non-memory
  // ops with an output and would otherwise be grabbed by matchesVOP.
  return Desc.TSFlags &
         (llvm::SIInstrFlags::SOP1 | llvm::SIInstrFlags::SOP2 |
          llvm::SIInstrFlags::SOPK | llvm::SIInstrFlags::SOPC);
}
static bool matchesScratch(const llvm::MCInstrDesc &Desc, const InstrProfile &) {
  return Desc.TSFlags & llvm::SIInstrFlags::FlatScratch;
}
static bool matchesFLAT(const llvm::MCInstrDesc &Desc, const InstrProfile &) {
  return Desc.TSFlags & llvm::SIInstrFlags::FLAT; // FLAT / GLOBAL (scratch above)
}
static bool matchesSMEM(const llvm::MCInstrDesc &Desc, const InstrProfile &) {
  // Scalar memory (SMRD/SMEM). Checked before VOP because scalar loads have an
  // sdst output and are not flagged as memory in the VOP matcher's view.
  return Desc.TSFlags & llvm::SIInstrFlags::SMRD;
}
static bool matchesMUBUF(const llvm::MCInstrDesc &Desc, const InstrProfile &) {
  // Untyped (MUBUF) and typed (MTBUF) buffer ops; both handled by buildMUBUF.
  return Desc.TSFlags &
         (llvm::SIInstrFlags::MUBUF | llvm::SIInstrFlags::MTBUF);
}
static bool matchesVOP(const llvm::MCInstrDesc &Desc, const InstrProfile &P) {
  // Non-memory VALU that produces an observable result: an explicit output
  // (VOP1/VOP2 and simple ALU), or a VOPC compare whose only result is the
  // implicit VCC mask.
  if (P.Mem.MemKind != MemAccessInfo::None)
    return false;
  return !P.Outputs.empty() || (Desc.TSFlags & llvm::SIInstrFlags::VOPC);
}

llvm::ArrayRef<InstrClassBuilder> getInstrClassBuilders() {
  static const InstrClassBuilder Builders[] = {
      {"DS", matchesDS, buildDS},
      {"SOP", matchesSOP, buildSOP},
      {"SCRATCH", matchesScratch, buildScratch},
      {"FLAT", matchesFLAT, buildFLAT},
      {"SMEM", matchesSMEM, buildSMEM},
      {"MUBUF", matchesMUBUF, buildMUBUF},
      {"VOP", matchesVOP, buildVOP},
  };
  return Builders;
}

//===----------------------------------------------------------------------===//
// build() dispatcher
//===----------------------------------------------------------------------===//
llvm::Expected<KernelMFContext>
MachineKernelBuilder::build(const InstrProfile &Profile, KernargLayout &Layout) {
  const llvm::MCInstrDesc &Desc = TM.getMCInstrInfo()->get(Profile.Opcode);
  for (const InstrClassBuilder &B : getInstrClassBuilders())
    if (B.Matches(Desc, Profile))
      return B.Build(TM, Profile, Layout);
  return makeError(Profile.Name +
                   ": no reference-kernel builder handles this instruction "
                   "class yet");
}

//===----------------------------------------------------------------------===//
// emitToELF()
//===----------------------------------------------------------------------===//
namespace {

/// Starts the codegen pipeline at a named pass for as long as it is alive. The
/// instruction under test is already selected, so the pipeline must start after
/// instruction selection; restored on destruction (option is process-wide).
class ScopedStartBeforePass {
public:
  static llvm::Expected<ScopedStartBeforePass> create(llvm::StringRef PassName) {
    auto &Opts = llvm::cl::getRegisteredOptions();
    auto It = Opts.find("start-before");
    if (It == Opts.end())
      return makeError("the 'start-before' codegen option is not registered; "
                       "llvm::initializeCodeGen() must be called first");
    auto *Opt = static_cast<llvm::cl::opt<std::string> *>(It->second);
    return ScopedStartBeforePass(Opt, PassName);
  }
  ~ScopedStartBeforePass() {
    if (Opt)
      Opt->setValue(Saved);
  }
  ScopedStartBeforePass(ScopedStartBeforePass &&O) noexcept
      : Opt(O.Opt), Saved(std::move(O.Saved)) {
    O.Opt = nullptr;
  }
  ScopedStartBeforePass(const ScopedStartBeforePass &) = delete;
  ScopedStartBeforePass &operator=(const ScopedStartBeforePass &) = delete;

private:
  ScopedStartBeforePass(llvm::cl::opt<std::string> *Opt, llvm::StringRef Name)
      : Opt(Opt), Saved(Opt->getValue()) {
    Opt->setValue(Name.str());
  }
  llvm::cl::opt<std::string> *Opt;
  std::string Saved;
};

} // namespace

llvm::Expected<llvm::SmallVector<char, 0>>
MachineKernelBuilder::emitToELF(KernelMFContext &KCtx) {
  if (!KCtx.MMIWP)
    return makeError("emitToELF may only be called once per kernel context");
  auto StartBeforeOrErr = ScopedStartBeforePass::create("prologepilog");
  if (!StartBeforeOrErr)
    return StartBeforeOrErr.takeError();

  llvm::SmallVector<char, 0> Obj;
  llvm::raw_svector_ostream OS(Obj);
  llvm::legacy::PassManager PM;
  llvm::MachineModuleInfoWrapperPass *MMIWP = KCtx.MMIWP.release();
  if (TM.addPassesToEmitFile(PM, OS, /*DwoOut=*/nullptr,
                             llvm::CodeGenFileType::ObjectFile,
                             /*DisableVerify=*/false, MMIWP))
    return makeError("the AMDGPU target does not support object emission");
  PM.run(*KCtx.Mod);

  llvm::SmallVector<char, 0> Exec;
  if (auto Err = luthier::linker::linkRelocatableToExecutable(Obj, Exec))
    return std::move(Err);
  return Exec;
}

//===----------------------------------------------------------------------===//
// emitTranslatedToELF()
//===----------------------------------------------------------------------===//

/// Record the parts of \p MF's shape that live outside its IR function.
///
/// The translated kernel is regenerated from IR alone (a fresh
/// MachineModuleInfo, so a brand-new MachineFunction), which means anything a
/// builder poked directly into the reference MachineFunction is dropped unless
/// the raised IR happens to re-encode it. The clearest example is a private
/// segment: the scratch builders call MachineFrameInfo::CreateStackObject to
/// force a non-zero private_segment_fixed_size in the kernel descriptor, and
/// that stack object has no IR representation at all.
static void captureFrameFacts(const llvm::MachineFunction &MF,
                              std::string &Out) {
  const llvm::MachineFrameInfo &MFI = MF.getFrameInfo();
  llvm::raw_string_ostream OS(Out);
  OS << "stack objects        : " << MFI.getNumObjects() << "\n"
     << "stack size           : " << MFI.getStackSize() << "\n"
     << "has stack objects    : " << (MFI.hasStackObjects() ? "yes" : "no")
     << "\n";
  // These attributes shape the kernarg/scratch ABI and, unlike frame objects,
  // do travel with the IR function.
  for (llvm::StringRef Attr :
       {"target-features", "amdgpu-flat-work-group-size", "amdgpu-num-sgpr",
        "amdgpu-num-vgpr"}) {
    if (MF.getFunction().hasFnAttribute(Attr))
      OS << Attr << " : "
         << MF.getFunction().getFnAttribute(Attr).getValueAsString() << "\n";
  }
}

/// Give the translated kernel the same private segment as the reference.
///
/// The scratch builders establish their private segment directly on the
/// reference MachineFunction (MachineFrameInfo::CreateStackObject). That is
/// what makes the emitted descriptor's private_segment_fixed_size non-zero,
/// and therefore what makes HSA back the kernel with scratch memory at all.
///
/// A stack object has no IR representation, and the translated kernel is
/// rebuilt from the raised IR alone — so without this the descriptor reports a
/// zero-sized private segment, HSA allocates no scratch, the flat-scratch base
/// resolves to null, and the first private access faults at address (nil).
/// That is a property of the kernel scaffold, not of the instruction semantic
/// under test, so reproducing it here keeps the two paths comparable.
///
/// The slot is anchored by an empty inline-asm use: it has no other users, so
/// AMDGPUPromoteAlloca or plain DCE would otherwise be free to delete it.
static void materializePrivateSegment(const llvm::MachineFunction &MF,
                                      llvm::Function &F) {
  const llvm::MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t Bytes = 0;
  llvm::Align MaxAlign(4);
  for (int I = 0, E = MFI.getObjectIndexEnd(); I < E; ++I) {
    if (MFI.isDeadObjectIndex(I))
      continue;
    int64_t Size = MFI.getObjectSize(I);
    if (Size <= 0)
      continue;
    Bytes += static_cast<uint64_t>(Size);
    MaxAlign = std::max(MaxAlign, MFI.getObjectAlign(I));
  }
  if (Bytes == 0 || F.empty())
    return;

  llvm::IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  auto *Slot = B.CreateAlloca(llvm::ArrayType::get(B.getInt8Ty(), Bytes),
                              llvm::AMDGPUAS::PRIVATE_ADDRESS, nullptr,
                              "scratch.backing");
  Slot->setAlignment(MaxAlign);
  auto *AsmTy =
      llvm::FunctionType::get(B.getVoidTy(), {Slot->getType()}, false);
  B.CreateCall(llvm::InlineAsm::get(AsmTy, /*AsmString=*/"", /*Constraints=*/"v",
                                    /*hasSideEffects=*/true),
               {Slot});
}

llvm::Expected<llvm::SmallVector<char, 0>>
MachineKernelBuilder::emitTranslatedToELF(KernelMFContext &KCtx,
                                          TranslationArtifacts *Artifacts) {
  if (!KCtx.MF)
    return makeError("emitTranslatedToELF: no MachineFunction");

  llvm::Function &F = const_cast<llvm::Function &>(KCtx.MF->getFunction());

  // Snapshot the MIR and the IR shell before translate() rewrites the body.
  if (Artifacts) {
    Artifacts->ReferenceMIR.clear();
    Artifacts->ReferenceIR.clear();
    Artifacts->ReferenceFrameFacts.clear();
    llvm::raw_string_ostream MIROS(Artifacts->ReferenceMIR);
    KCtx.MF->print(MIROS);
    llvm::raw_string_ostream(Artifacts->ReferenceIR) << F;
    captureFrameFacts(*KCtx.MF, Artifacts->ReferenceFrameFacts);
  }

  // Raise the reference-kernel MIR back to LLVM IR, in place in the kernel's
  // IR function body.
  llvm::Error Err = llvm::Error::success();
  luthier::TraceFunctionTranslator Translator(*KCtx.MF, Err);
  if (Err)
    return std::move(Err);
  Translator.translate();

  // The translator emits llvm.ssa.copy identity copies for register-value
  // tracking; replace each with its source so the module is plain, codegen-
  // ready IR.
  llvm::SmallVector<llvm::CallInst *, 32> ToErase;
  for (llvm::BasicBlock &BB : F)
    for (llvm::Instruction &I : BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (const llvm::Function *Callee = CI->getCalledFunction())
          if (Callee->getIntrinsicID() == llvm::Intrinsic::ssa_copy) {
            CI->replaceAllUsesWith(CI->getArgOperand(0));
            ToErase.push_back(CI);
          }
  for (llvm::CallInst *CI : ToErase)
    CI->eraseFromParent();

  // Restore the private segment, which is held on the reference
  // MachineFunction and does not survive the round trip through IR.
  materializePrivateSegment(*KCtx.MF, F);

  if (Artifacts) {
    Artifacts->RaisedIR.clear();
    llvm::raw_string_ostream(Artifacts->RaisedIR) << F;
  }

  // Codegen the raised module from IR (fresh pipeline: full ISel, not the
  // start-before=prologepilog reference shortcut). The reference path may have
  // set the global start-before option, so force it clear for this compile.
  auto StartBeforeOrErr = ScopedStartBeforePass::create("");
  if (!StartBeforeOrErr)
    return StartBeforeOrErr.takeError();

  llvm::SmallVector<char, 0> Obj;
  llvm::raw_svector_ostream OS(Obj);
  llvm::legacy::PassManager PM;
  if (TM.addPassesToEmitFile(PM, OS, /*DwoOut=*/nullptr,
                             llvm::CodeGenFileType::ObjectFile,
                             /*DisableVerify=*/true, /*MMIWP=*/nullptr))
    return makeError("the AMDGPU target does not support object emission");
  PM.run(*KCtx.Mod);

  llvm::SmallVector<char, 0> Exec;
  if (auto E = luthier::linker::linkRelocatableToExecutable(Obj, Exec))
    return std::move(E);
  return Exec;
}

} // namespace luthier::test
