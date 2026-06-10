//===-- IndirectBranchResolverPass.hip ------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// The indirect-branch resolution feature in a single HIP translation unit:
///
///  * **Device side** — the resolver \c __luthier_resolve_indirect, compiled
///    into this tool's Luthier offload bundle (per-arch LLVM bitcode). It
///    translates an instrumented kernel's computed branch/call target (a
///    *trace* address) to the runtime *load* address of the instrumented copy:
///    scan the device-resident trace->load map on the fast path; on a miss, ask
///    the host via a device-libs \c SERVICE_FUNCTION_CALL hostcall.
///
///  * **Host side** — this TU is also the bundle's host translation unit, and
///    the *only* TU that completes \c DeviceToolCodeFatBinaryParser<Impl> (so
///    \c LoadHIPFATBinaryInfoPass populates the embedded fat-binary slots). It
///    defines \c Impl (the parser holding the resolver bundle) and every member
///    of \c luthier::IndirectBranchResolverPass.
///
/// The map pointer and host-resolver-fn are plain \c __device__ globals (no
/// longer host-published externs); the map is seeded by a device constructor
/// (deferred). The hostcall buffer is obtained from the kernel's implicit
/// arguments by \c __ockl_hostcall_preview itself (the normal device-libs API),
/// so the resolver needs no explicit buffer global; the Luthier tool IR pass
/// repoints the implicit-arg pointer to Luthier's implicit-arg block.
/// \c __ockl_hostcall_preview is wave-collective — it coalesces the active
/// lanes into one packet and elects a leader to submit it, so divergent
/// per-lane targets each get resolved without any hand-rolled wave
/// coordination.
//===----------------------------------------------------------------------===//
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSATooling/DeviceToolCodeFatBinaryParser.h"
#include "luthier/Intrinsic/IntrinsicCalls.h"
#include "luthier/ToolCodeGen/CustomKernargLayout.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/MIToIRMappingAnalysis.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <luthier/ToolCodeGen/IndirectBranchResolverPass.h>
#include <luthier/ToolCodeGen/TraceCallGraph.h>
#include <optional>

/// device-libs hostcall entry (defined in ockl.bc, linked at compile time).
/// The normal hostcall API: it fetches the per-wave hostcall buffer from the
/// kernel's implicit arguments itself (via \c __builtin_amdgcn_implicitarg_ptr,
/// which the Luthier tool IR pass repoints to Luthier's implicit-arg block),
/// pushes one packet carrying \p Service + \p A0..A7, blocks the wave until the
/// host responds, and returns the first two output ulongs.
extern "C" __device__ unsigned long __attribute__((ext_vector_type(2)))
__ockl_hostcall_preview(unsigned int Service, unsigned long A0,
                        unsigned long A1, unsigned long A2, unsigned long A3,
                        unsigned long A4, unsigned long A5, unsigned long A6,
                        unsigned long A7);

namespace {

/// One resolved indirect-target mapping: the compile-time *trace* address of a
/// callee's entry point (\c getFunctionEntryPoint(F)->getRawAddress()) and the
/// runtime *load* address of its instrumented copy.
struct MapEntry {
  uint64_t Trace;
  uint64_t Load;
};

/// Header of the device-resident trace->load table. \c Count entries follow
/// contiguously after this header; the device-side constructor (deferred — see
/// the pass docs) seeds the compile-time-known entries and the resolver scans
/// \c [0, Count) on the fast path. \c Capacity bounds growth.
struct ResolveMap {
  uint64_t Count;
  uint64_t Capacity;
  MapEntry Entries[1]; // flexible array; real length is Capacity
};

/// Device global (a \c ResolveMap*) — the trace->load table. Defined (null) in
/// the resolver; populated by the deferred device constructor rather than the
/// host loader.
inline constexpr const char *IndirectMapGlobalName = "__luthier_indirect_map";

/// Device global (a \c uint64_t) — the host address of the resolver function,
/// invoked via a \c SERVICE_FUNCTION_CALL hostcall on the slow path.
inline constexpr const char *HostResolveFnGlobalName =
    "__luthier_host_resolve_fn";

/// Name of the device-side resolver entry the per-site dispatcher calls:
/// \c uint64_t __luthier_resolve_indirect(uint64_t traceAddr).
inline constexpr const char *ResolveFnName = "__luthier_resolve_indirect";

} // namespace

/// Runtime inputs. Defined (not extern) device globals: the trace->load map is
/// seeded by a device constructor (deferred); the host-resolver-fn pointer is
/// filled by the deferred loader/ctor wiring. (The hostcall buffer is no longer
/// a global — \c __ockl_hostcall_preview reads it from the implicit args.)
extern "C" __device__ ResolveMap *__luthier_indirect_map = nullptr;
extern "C" __device__ unsigned long __luthier_host_resolve_fn = 0;

// __attribute__((used)) keeps the resolver alive through the HIP device
// internalize/DCE pass (it has no in-TU kernel caller — its callers are the
// per-site dispatchers injected later, after this bitcode is linked into the
// IModule). noinline keeps it a single linkable definition.
extern "C" __device__ __attribute__((used, noinline)) unsigned long
__luthier_resolve_indirect(unsigned long TraceAddr) {
  // Fast path: scan the already-resolved table (acquire-loaded count pairs with
  // the writer's release-store on append).
  if (ResolveMap *Map = __luthier_indirect_map) {
    unsigned long N = __atomic_load_n(&Map->Count, __ATOMIC_ACQUIRE);
    for (unsigned long I = 0; I < N; ++I)
      if (Map->Entries[I].Trace == TraceAddr)
        return Map->Entries[I].Load;
  }
  // Slow path: SERVICE_FUNCTION_CALL hostcall to the host resolver. Argument
  // layout for the function-call service: a0 = host function pointer, the rest
  // become its inputs (in[0] = TraceAddr); the host writes out[0] = load
  // address, returned here as the .x lane of the result.
  unsigned long __attribute__((ext_vector_type(2))) R =
      __ockl_hostcall_preview(ServiceFunctionCall, __luthier_host_resolve_fn,
                              TraceAddr, 0, 0, 0, 0, 0, 0);
  return R.x;
}

//===----------------------------------------------------------------------===//
// Host side — the bundle's host TU (guarded out of device compilation). This is
// the only TU that completes DeviceToolCodeFatBinaryParser<Impl>.
//===----------------------------------------------------------------------===//

namespace luthier {

/// The parser holding this pass's embedded resolver fat binary. Defined here
/// and nowhere else, so \c DeviceToolCodeFatBinaryParser<Impl> (and its
/// \c [[gnu::used]] fat-binary slots) is instantiated in this single TU.
struct IndirectBranchResolverPass::Impl
    : DeviceToolCodeFatBinaryParser<IndirectBranchResolverPass::Impl> {
  using DeviceToolCodeFatBinaryParser::DeviceToolCodeFatBinaryParser;
};

namespace {

/// The computed-target register: the first explicit register *use* operand of
/// the transfer (the 64-bit SGPR pair holding the PC for S_SETPC_B64 /
/// S_SWAPPC_B64).
std::optional<llvm::MCRegister> getTargetReg(const llvm::MachineInstr &MI) {
  for (const llvm::MachineOperand &MO : MI.explicit_uses())
    if (MO.isReg() && MO.getReg())
      return MO.getReg().asMCReg();
  return std::nullopt;
}

/// Emit a \c luthier::writeReg intrinsic call \c writeReg(Reg, Val). The
/// register enum is the first (constant) argument; the value the second. The
/// intrinsic is keyed by the \c luthier.intrinsic function attribute, not the
/// mangled name (mirrors how \c IntrinsicCalls.h builds \c readReg). Built by
/// hand because the variadic \c insertCallToIntrinsic cannot take mixed
/// scalar + \c Value arguments.
void buildWriteReg(llvm::Module &IModule, llvm::IRBuilderBase &Builder,
                   llvm::MCRegister Reg, llvm::Value *Val) {
  llvm::LLVMContext &Ctx = IModule.getContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *ValTy = Val->getType();
  auto *FTy = llvm::FunctionType::get(ValTy, {I32, ValTy}, /*isVarArg=*/false);
  llvm::FunctionCallee Callee = IModule.getOrInsertFunction(
      "luthier::writeReg.i64", FTy,
      llvm::AttributeList().addFnAttribute(Ctx, IntrinsicAttribute,
                                           "luthier::writeReg"));
  Builder.CreateCall(Callee, {llvm::ConstantInt::get(I32, Reg.id()), Val});
}

/// Drop the \c .luthier.uses_hostcall marker into \p TargetModule (idempotent)
/// so the launcher provisions a hostcall buffer for the resolver slow path.
void emitHostcallMarker(llvm::Module &TargetModule) {
  if (TargetModule.getNamedGlobal(HostcallMarkerGlobalName))
    return;
  llvm::Type *I8 = llvm::Type::getInt8Ty(TargetModule.getContext());
  auto *GV = new llvm::GlobalVariable(
      TargetModule, I8, /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
      llvm::ConstantInt::get(I8, 1), HostcallMarkerGlobalName);
  GV->setSection(HostcallMarkerSectionName);
  GV->setVisibility(llvm::GlobalValue::ProtectedVisibility);
  llvm::appendToUsed(TargetModule, {GV});
}

/// Rewrite a call \p MI (it saves a return address into its sdst \p RetReg and
/// jumps to the computed target \p TargetReg) into, at the target-module MIR
/// level:
///   \code
///     RetReg = <inst_addr> + 4          ; two S_MOV_B32 (lo/hi)
///     S_SETPC_B64 TargetReg             ; implicit use of RetReg
///   \endcode
/// where \c <inst_addr> is the call's own *trace* address read from its
/// PC-sections metadata. The hardware return-address save is replaced by an
/// explicit constant move because S_SETPC_B64 does not save one; the return
/// goes to the trace-domain next-instruction address and is itself resolved at
/// the return site. \p RetReg is kept live across the jump via an implicit use.
/// The new jump inherits \p MI 's PC-sections so \c TargetModulePatcherPass
/// still maps the injected dispatcher payload onto this site; \p MI is erased.
llvm::Error rewriteCallToJump(llvm::MachineInstr &MI,
                              llvm::MCRegister TargetReg,
                              llvm::MCRegister RetReg) {
  llvm::MachineFunction &MF = *MI.getMF();
  const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  TargetMachineInstrMDNode *MD =
      TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
  std::optional<uint64_t> InstAddr =
      MD ? MD->getTraceInstrAddress() : std::nullopt;
  if (!InstAddr)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Call site has no trace address in its PC-sections metadata; cannot "
        "compute the return address.");
  // S_SWAPPC_B64 / S_CALL_B64 are 4-byte instructions, so the return point is
  // the trace address immediately after the call.
  const uint64_t RetAddr = *InstAddr + 4;

  llvm::MDNode *PCS = MI.getPCSections();
  llvm::MachineBasicBlock &MBB = *MI.getParent();
  const llvm::DebugLoc &DL = MI.getDebugLoc();

  // Materialize the (trace-domain) return address into the call's return reg:
  // RetReg.sub0 = lo32, RetReg.sub1 = hi32.
  llvm::BuildMI(MBB, MI, DL, TII->get(llvm::AMDGPU::S_MOV_B32),
                TRI->getSubReg(RetReg, llvm::AMDGPU::sub0))
      .addImm(static_cast<int64_t>(RetAddr & 0xffffffffULL));
  llvm::BuildMI(MBB, MI, DL, TII->get(llvm::AMDGPU::S_MOV_B32),
                TRI->getSubReg(RetReg, llvm::AMDGPU::sub1))
      .addImm(static_cast<int64_t>(RetAddr >> 32));

  // Replace the call with a plain indirect jump to the (dispatcher-rewritten)
  // target, keeping RetReg live across it.
  llvm::MachineInstrBuilder Jump =
      llvm::BuildMI(MBB, MI, DL, TII->get(llvm::AMDGPU::S_SETPC_B64))
          .addReg(TargetReg)
          .addReg(RetReg, llvm::RegState::Implicit);
  if (PCS)
    Jump->setPCSections(MF, PCS);

  MI.eraseFromParent();
  return llvm::Error::success();
}

} // namespace

IndirectBranchResolverPass::IndirectBranchResolverPass(llvm::Error &Err)
    : PImpl(std::make_unique<Impl>(Err)) {}

IndirectBranchResolverPass::IndirectBranchResolverPass(
    IndirectBranchResolverPass &&) noexcept = default;

IndirectBranchResolverPass::~IndirectBranchResolverPass() = default;

llvm::Error IndirectBranchResolverPass::ensureResolverLinked(
    llvm::Module &IModule, const llvm::TargetMachine &TM) {
  if (Linked)
    return llvm::Error::success();

  llvm::Triple T = TM.getTargetTriple();
  llvm::SubtargetFeatures Features(TM.getTargetFeatureString());

  auto ModOrErr =
      PImpl->parseModule(T, TM.getTargetCPU(), Features, IModule.getContext());
  if (!ModOrErr)
    return ModOrErr.takeError();

  if (llvm::Linker::linkModules(IModule, std::move(*ModOrErr)))
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to link the indirect-branch resolver module into the IModule.");

  ResolverFn = IModule.getFunction(ResolveFnName);
  if (ResolverFn == nullptr)
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::Twine("Linked resolver module is missing the '") + ResolveFnName +
        "' function.");

  Linked = true;
  return llvm::Error::success();
}

llvm::Error IndirectBranchResolverPass::injectDispatcher(
    llvm::Module &IModule, llvm::MachineInstr &MI,
    const llvm::TargetMachine &TM, llvm::ArrayRef<llvm::Function *> CallTargets,
    bool IsIncomplete) {
  std::optional<llvm::MCRegister> TargetReg = getTargetReg(MI);
  if (!TargetReg)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Indirect transfer has no register target operand to resolve.");

  // For a call (it saves a return address into its sdst def operand and jumps),
  // capture that return-address register before the call is rewritten into a
  // plain jump below, so the payload can restore it.
  const bool IsCall = MI.isCall();
  std::optional<llvm::MCRegister> RetAddrReg;
  if (IsCall && MI.getNumOperands() > 0 && MI.getOperand(0).isReg() &&
      MI.getOperand(0).isDef())
    RetAddrReg = MI.getOperand(0).getReg().asMCReg();

  auto FnOrErr = createInjectedPayload(
      IModule, MI, [&](llvm::IRBuilderBase &Builder) -> llvm::Error {
        llvm::LLVMContext &Ctx = IModule.getContext();
        llvm::Type *I64 = Builder.getInt64Ty();
        llvm::Function *PayloadFn = Builder.GetInsertBlock()->getParent();

        // The computed *trace* address currently held in the target register.
        llvm::CallInst *Trace = insertCallToIntrinsic(
            IModule, Builder, "luthier::readReg", *I64, TargetReg->id());

        // Fast path: an if/else chain over the statically-known callees. On a
        // match, overwrite <target> with that callee's (instrumented) address
        // and return from the payload.
        llvm::BasicBlock *Cur = Builder.GetInsertBlock();
        for (llvm::Function *Callee : CallTargets) {
          std::optional<EntryPoint> EP = getFunctionEntryPoint(*Callee);
          if (!EP)
            continue;
          auto *Match = llvm::BasicBlock::Create(Ctx, "match", PayloadFn);
          auto *Next = llvm::BasicBlock::Create(Ctx, "next", PayloadFn);
          Builder.SetInsertPoint(Cur);
          llvm::Value *Eq = Builder.CreateICmpEQ(
              Trace, Builder.getInt64(EP->getRawAddress()));
          Builder.CreateCondBr(Eq, Match, Next);

          Builder.SetInsertPoint(Match);
          // Reference the callee by name in the IModule (a trivial declaration
          // — only its address matters); the patcher/loader binds it to the
          // loaded instrumented copy. A canonical void() type in the IModule's
          // context avoids reusing the callee's cross-module FunctionType.
          auto *VoidFnTy =
              llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
          llvm::FunctionCallee CalleeRef =
              IModule.getOrInsertFunction(Callee->getName(), VoidFnTy);
          llvm::Value *LoadAddr =
              Builder.CreatePtrToInt(CalleeRef.getCallee(), I64);
          buildWriteReg(IModule, Builder, *TargetReg, LoadAddr);
          Builder.CreateRetVoid();

          Cur = Next;
        }

        // Default arm: no static match. Resolve at runtime only when the site
        // is incomplete (has unresolved targets); a fully-resolved site never
        // reaches here. Leave the builder in this block so
        // createInjectedPayload appends the terminating `ret void`.
        Builder.SetInsertPoint(Cur);
        if (IsIncomplete) {
          llvm::Value *Load = Builder.CreateCall(ResolverFn->getFunctionType(),
                                                 ResolverFn, {Trace});
          buildWriteReg(IModule, Builder, *TargetReg, Load);
        }
        return llvm::Error::success();
      });
  if (!FnOrErr)
    return FnOrErr.takeError();

  // Turn the call into a plain indirect jump now that the dispatcher owns
  // <target>, saving the trace-domain return address into the call's sdst
  // register; the new jump inherits the call's PC-sections so the patcher still
  // maps this payload onto it.
  if (IsCall) {
    if (!RetAddrReg)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Call site has no return-address (sdst) register to preserve.");
    if (llvm::Error Err = rewriteCallToJump(MI, *TargetReg, *RetAddrReg))
      return Err;
  }

  // Match the payload's subtarget to the target it is injected into.
  llvm::Function *PayloadFn = *FnOrErr;
  if (!TM.getTargetCPU().empty())
    PayloadFn->addFnAttr("target-cpu", TM.getTargetCPU());
  if (!TM.getTargetFeatureString().empty())
    PayloadFn->addFnAttr("target-features", TM.getTargetFeatureString());
  return llvm::Error::success();
}

InstrumentationPreservedAnalyses
IndirectBranchResolverPass::runInstrumentationPass(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &IMAM,
    llvm::Module &TargetModule, llvm::ModuleAnalysisManager &TargetMAM) {
  TraceCallGraph &CG =
      TargetMAM.getResult<TraceCallGraphAnalysis>(TargetModule);
  // If there are no calls in the call graph, there's nothing to patch.
  if (CG.call_targets_empty())
    return {llvm::PreservedAnalyses::all(), llvm::PreservedAnalyses::all()};

  llvm::LLVMContext &Ctx = IModule.getContext();

  // We have call targets to deal with, so link the resolver device module into
  // the IModule now (once). The ISA tuple comes from the target module's MMI.
  llvm::MachineModuleInfo &MMI =
      TargetMAM.getResult<llvm::MachineModuleAnalysis>(TargetModule).getMMI();
  const llvm::TargetMachine &TM = MMI.getTarget();
  if (llvm::Error Err = ensureResolverLinked(IModule, TM)) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return {llvm::PreservedAnalyses::all(), llvm::PreservedAnalyses::all()};
  }

  llvm::FunctionAnalysisManager &FAM =
      TargetMAM
          .getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();
  llvm::MachineFunctionAnalysisManager &MFAM =
      TargetMAM
          .getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(
              TargetModule)
          .getManager();

  for (auto &[CI, CallTargets] : CG.call_targets()) {
    // Map the trace CallInst back to the MachineInstr it was lifted from.
    llvm::Function *CIFunc = CI->getFunction();
    llvm::MachineFunction &MF =
        FAM.getResult<llvm::MachineFunctionAnalysis>(*CIFunc).getMF();
    const MIToIRMapping &Mapping = MFAM.getResult<MIToIRMappingAnalysis>(MF);
    llvm::MachineInstr *MI = Mapping.getMachineInstr(*CI);

    // Only patch call sites backed by a real lifted *trace* instruction; skip
    // synthesized / non-trace calls (e.g. instructions with no source MI, or
    // whose MI carries no trace address).
    if (MI == nullptr)
      continue;
    TargetMachineInstrMDNode *MD =
        TargetMachineInstrMDNode::getInstrMDNodeIfExists(*MI);
    if (MD == nullptr || !MD->isTraceInstr())
      continue;

    // Inject the dispatcher: an if/else chain over the known targets plus, for
    // incomplete sites, a fall-through to the device resolver. injectDispatcher
    // also rewrites a call MI into a plain indirect jump.
    if (llvm::Error Err = injectDispatcher(IModule, *MI, TM, CallTargets,
                                           CG.containsIncompleteCallSite(CI))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return {llvm::PreservedAnalyses::all(), llvm::PreservedAnalyses::all()};
    }
  }

  emitHostcallMarker(TargetModule);

  // We linked into the IModule and rewrote/instrumented target call sites, so
  // neither side's analyses are preserved.
  return {llvm::PreservedAnalyses::none(), llvm::PreservedAnalyses::none()};
}

} // namespace luthier
