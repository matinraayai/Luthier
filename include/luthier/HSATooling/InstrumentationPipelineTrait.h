//===-- InstrumentationPipelineTrait.h --------------------------*- C++ -*-===//
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
/// \file
/// CRTP trait for creating and running the instrumentation pipeline used by
/// every HSA tool.
///
/// The trait forwards a set of optional, plugin-style callbacks to
/// the pipeline builder. Each callback is detected on \c Derived via a
/// \c requires-expression; if \c Derived does not define a given method, the
/// corresponding driver callback is a no-op. The detected customization
/// points (all optional) are:
///   - \c createInstrumentationModule(llvm::LLVMContext &)
///   - \c preIROptimizationPasses(llvm::ModulePassManager &)
///   - \c registerInstrumentationAnalyses(llvm::ModuleAnalysisManager &,
///        llvm::MachineFunctionAnalysisManager &)
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INSTRUMENTATION_PIPELINE_TRAIT_H
#define LUTHIER_TOOLING_INSTRUMENTATION_PIPELINE_TRAIT_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/CodeDiscoveryPass.h"
#include "luthier/ToolCodeGen/EntryPoint.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/InitialExecutionPointAnalysis.h"
#include "luthier/ToolCodeGen/InstructionTracesAnalysis.h"
#include "luthier/ToolCodeGen/InstrumentationPassBuilder.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorsAnalysis.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/NewPMAsmPrinter.h"
#include "luthier/ToolCodeGen/ParentPrototypeAnalysis.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "luthier/ToolCodeGen/PrototypeCallGraph.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeParser.h"
#include "luthier/ToolCodeGen/TraceFunctionTranslationAnalysis.h"
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/CGPassBuilderOption.h>
#include <llvm/Target/TargetMachine.h>
#include <memory>
#include <string>

namespace luthier {

/// \brief CRTP trait that runs Luthier's per-dispatch instrumentation pipeline.
///
/// \tparam Derived the concrete tool. It must provide
/// \c buildTargetMachineForKD, \c createMemoryAllocationAccessor,
/// \c parseModule, \c getIntrinsicProcessorRegistry, and be an
/// \c InstrumentationPass for the payload-injection adapter cast to succeed —
/// all of which \c HSATool and \c KFDTool supply.
///
/// \note Despite living under \c HSATooling, nothing in this trait is specific
/// to HSA. The one thing that was — constructing the memory allocation accessor
/// — is now \c createMemoryAllocationAccessor on the tool, because that is
/// precisely what differs between a tool attached to an HSA application and one
/// attached to an application that drives the driver itself.
/// \tparam TargetUnitT the instrumentation target unit (matches \c HSATool's).
template <typename Derived, typename TargetUnitT = llvm::MachineFunction>
class InstrumentationPipelineTrait {
  Derived &derived() { return static_cast<Derived &>(*this); }

  /// Thin Prototype-pass adapter that forwards into the tool's own
  /// \c run(Prototype &, PrototypeAnalysisManager &) so the pipeline can hook
  /// the tool's payload-injection logic in without copying the singleton tool
  /// object.
  ///
  /// Payload creation is a Prototype-level pass now: it reads the target
  /// module's MIR and writes into the instrumentation module, so it needs both
  /// halves of the prototype rather than a single module.
  struct InjectPayloadsAdapter
      : public llvm::PassInfoMixin<InjectPayloadsAdapter> {
    Derived *T;
    explicit InjectPayloadsAdapter(Derived *T) : T(T) {}
    llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM) {
      return T->run(P, PAM);
    }
    static bool isRequired() { return true; }
  };

public:
  /// Register the common set of instrumentation analyses on \p MAM / \p MFAM
  /// for the kernel described by \p KD. \p MMI and \p MDParser must outlive the
  /// pass run that consumes them. After the common analyses are registered, the
  /// tool's optional \c registerInstrumentationAnalyses(MAM, MFAM) hook (if
  /// present) is invoked so a tool can add its own.
  void
  registerInstrumentationAnalyses(llvm::MachineModuleInfo &MMI,
                                  llvm::ModuleAnalysisManager &MAM,
                                  llvm::MachineFunctionAnalysisManager &MFAM) {
    Derived &D = derived();

    MAM.registerPass([&] { return llvm::MachineModuleAnalysis(MMI); });
    MFAM.registerPass([] { return luthier::InstructionTracesAnalysis(); });
    MFAM.registerPass(
        [] { return luthier::TraceFunctionTranslationAnalysis(); });
    MAM.registerPass([] { return luthier::InitialEntryPointAnalysis(); });
    MAM.registerPass([] { return luthier::InitialExecutionPointAnalysis(); });
    // The accessor comes from the tool, because what can answer "which
    // allocation holds this address" is exactly what differs between a tool
    // attached to an HSA application and one attached to an application that
    // drives the driver itself. Nothing else in this pipeline differs between
    // the two, which is why this is the only hook.
    MAM.registerPass([&] {
      return luthier::MemoryAllocationAnalysis(
          D.createMemoryAllocationAccessor());
    });

    // PrototypeCallGraphAnalysis, IPPredCFGAnalysis and
    // FunctionPreambleDescriptorAnalysis are Prototype analyses; they are
    // registered on the PrototypeAnalysisManager (see
    // registerPrototypeAnalyses on InstrumentationPassBuilder), not here.
    // Registering a Prototype analysis on a ModuleAnalysisManager compiles but
    // can never resolve at run time.

    if constexpr (requires(Derived &Tool) {
                    Tool.registerInstrumentationAnalyses(MAM, MFAM);
                  })
      D.registerInstrumentationAnalyses(MAM, MFAM);
  }

  /// Assemble and run the standard instrumentation pipeline for the kernel
  /// referenced by \p KD, returning the resulting relocatable object-file
  /// bytes.
  ///
  /// The pipeline itself comes from
  /// \c InstrumentationPassBuilder::buildInstrumentationPipeline: code
  /// discovery, the tool's payload injection, IModule optimization and
  /// intrinsic lowering, AMDGPU codegen, and finally the target-module patch
  /// plus asm printing. \p Level selects the optimization level used for the
  /// instrumentation module's IR pipeline.
protected:
  /// \brief Everything one dispatch's lifting stands on, handed to a body that
  /// runs while it is all still alive.
  ///
  /// References throughout: every one of these lives on \c withLiftedDispatch's
  /// frame and dies when it returns, which is why a body has to finish its work
  /// rather than stash any of this.
  struct LiftedDispatch {
    luthier::Prototype &IP;
    luthier::PrototypeAnalysisManager &IPAM;
    luthier::InstrumentationPassBuilder &PB;
    llvm::TargetMachine &TM;
    /// The two module analysis managers, needed by name because
    /// \c ParentPrototypeAnalysis is consumed through \c getCachedResult and so
    /// has to be materialized in each one before a pipeline runs.
    llvm::ModuleAnalysisManager &TargetMAM;
    llvm::ModuleAnalysisManager &IMAM;
  };

private:
  /// Stand up everything one dispatch's lifting needs, then hand it to \p Body.
  ///
  /// Extracted so the two entry points below cannot drift apart. That matters
  /// more than it usually would: the declaration order of the analysis managers
  /// here is load-bearing (see the comment on them), and a second copy of that
  /// ordering would be a second chance to get it subtly wrong -- with the
  /// symptom appearing as lifted MIR vanishing mid-pipeline rather than as
  /// anything that looks like a lifetime bug.
  ///
  /// \param Body receives the prototype, its analysis manager, the pass builder
  /// and the target machine, all fully registered and cross-proxied. Anything it
  /// wants to keep must be copied out: every object here dies when this returns.
  template <typename BodyT>
  llvm::Error withLiftedDispatch(const llvm::amdhsa::kernel_descriptor_t &KD,
                                 llvm::OptimizationLevel Level,
                                 llvm::PassInstrumentationCallbacks &PIC,
                                 BodyT Body) {
    Derived &D = derived();

    std::unique_ptr<llvm::TargetMachine> TM;
    LUTHIER_RETURN_ON_ERROR(D.buildTargetMachineForKD(&KD).moveInto(TM));

    llvm::LLVMContext Ctx;
    auto TargetM = std::make_unique<llvm::Module>("luthier.target", Ctx);
    TargetM->setTargetTriple(TM->getTargetTriple());
    TargetM->setDataLayout(TM->createDataLayout());

    // The prototype owns both modules for the whole run. The instrumentation
    // module holds the tool's hooks: either the tool builds it, or its embedded
    // device-side bitcode is parsed here. It is populated up front rather than
    // materialized mid-pipeline, because every pass from payload injection
    // onwards expects both halves of the prototype to exist.
    llvm::Triple ToolTriple = TM->getTargetTriple();
    std::string ToolCPU(TM->getTargetCPU());
    llvm::SubtargetFeatures ToolFeatures(TM->getTargetFeatureString());

    std::unique_ptr<llvm::Module> IModuleM;
    if constexpr (requires(Derived &Tool) {
                    Tool.createInstrumentationModule(Ctx);
                  }) {
      IModuleM = D.createInstrumentationModule(Ctx);
    } else if constexpr (requires(Derived &Tool) {
                           Tool.parseModule(ToolTriple, ToolCPU, ToolFeatures,
                                            Ctx);
                         }) {
      LUTHIER_RETURN_ON_ERROR(
          D.parseModule(ToolTriple, ToolCPU, ToolFeatures, Ctx)
              .moveInto(IModuleM));
    } else {
      // A tool with no device code of its own -- an analysis-only tool. The
      // prototype still owns two modules, because everything downstream expects
      // both halves to exist, but nothing will ever put anything in this one.
      // Reaching payload injection from here would fail on an empty module, and
      // that is the right outcome: a tool with no payload has no business in the
      // instrumentation pipeline, only in runCodeDiscoveryForDispatch.
      IModuleM = std::make_unique<llvm::Module>("luthier.instrumentation", Ctx);
    }
    IModuleM->setTargetTriple(ToolTriple);
    IModuleM->setDataLayout(TM->createDataLayout());

    // Record the dispatch's entry/execution point on the target module so the
    // corresponding analyses can read them without knowing where a kernel
    // descriptor comes from.
    luthier::setInitialEntryPoint(*TargetM, luthier::EntryPoint(KD));
    luthier::setInitialExecutionPoint(*TargetM, KD);

    luthier::Prototype IP(std::move(TargetM), std::move(IModuleM));

    llvm::MachineModuleInfo MMI(TM.get());

    // Each of the prototype's two modules gets its own set of managers. They
    // must not be shared: LLVM reaches a module's inner managers through
    // proxies whose invalidation hook clears the inner manager wholesale
    // (FunctionAnalysisManagerModuleProxy::Result::invalidate in
    // PassManager.cpp), and a nested llvm::ModulePassManager re-invalidates
    // after every pass it runs. With one shared set, the first pass of the
    // instrumentation module's IR pipeline to report PreservedAnalyses::none()
    // therefore destroys the target module's cached MachineFunctionAnalysis
    // results -- and with them the lifted target MIR those results own.
    //
    // Declaration order matters: these are destroyed in reverse, and an inner
    // analysis-manager proxy's destructor clears the manager it proxies. So an
    // outer manager must be declared after everything it proxies -- innermost
    // first, the Prototype manager (which proxies all six) last.
    llvm::LoopAnalysisManager TargetLAM, ILAM;
    llvm::FunctionAnalysisManager TargetFAM, IFAM;
    llvm::CGSCCAnalysisManager TargetCGAM, ICGAM;
    llvm::MachineFunctionAnalysisManager TargetMFAM, IMFAM;
    llvm::ModuleAnalysisManager TargetMAM, IMAM;
    luthier::PrototypeAnalysisManager IPAM;

    const luthier::InstrumentationPassBuilder::ModuleAnalysisManagers TargetAMs{
        TargetMAM, TargetCGAM, TargetFAM, TargetLAM, TargetMFAM};
    const luthier::InstrumentationPassBuilder::ModuleAnalysisManagers IAMs{
        IMAM, ICGAM, IFAM, ILAM, IMFAM};

    // SI must outlive the pipeline run. StandardInstrumentations reads
    // --print-after-all / --print-before-all / --print-changed / -time-passes
    // and registers the corresponding PassInstrumentationCallbacks. PIC is the
    // caller's, because it must outlive this frame: the pass manager the body
    // builds holds on to it.
    llvm::StandardInstrumentations SI(Ctx, /*DebugLogging=*/false);

    luthier::InstrumentationPassBuilder PB(*TM, llvm::PipelineTuningOptions(),
                                           std::nullopt, &PIC);
    PB.registerPrototypeAnalyses(IPAM);
    PB.registerAnalyses(TargetAMs);
    PB.registerAnalyses(IAMs);
    PB.crossRegisterProxies(IPAM, TargetAMs, IAMs);

    // StandardInstrumentations takes a single ModuleAnalysisManager, used only
    // by its optional debug instrumentation (-print-changed, CFG checking).
    // The instrumentation module is where all the heavy pipelines run, so it is
    // the one worth wiring up.
    SI.registerCallbacks(PIC, &IMAM);

    // Both modules need the common analyses: a module analysis is resolved out
    // of the manager belonging to the module it is asked about. The single MMI
    // is deliberately shared -- it owns the MCContext the MachineFunctions of
    // both modules are created against, and TargetModulePatcherPass moves MIR
    // between them.
    registerInstrumentationAnalyses(MMI, TargetMAM, TargetMFAM);
    registerInstrumentationAnalyses(MMI, IMAM, IMFAM);

    // Intrinsic lowering resolves each luthier:: intrinsic through this
    // registry, which the tool owns.
    for (llvm::ModuleAnalysisManager *M : {&TargetMAM, &IMAM})
      M->registerPass([&D] {
        return luthier::IntrinsicsProcessorsAnalysis(
            D.getIntrinsicProcessorRegistry());
      });

    // The machine-function passes InjectedPayloadPEIPass and SVAPhysVGPRPinPass
    // resolve their module back to the owning prototype through this map, and
    // read the analysis with getCachedResult -- so it has to be registered and
    // materialized before the pipeline runs.
    luthier::ModuleToPrototypeMap ParentMap;
    ParentMap.registerPrototype(IP);
    for (llvm::ModuleAnalysisManager *M : {&TargetMAM, &IMAM})
      M->registerPass(
          [&ParentMap] { return luthier::ParentPrototypeAnalysis(ParentMap); });

    return Body(LiftedDispatch{IP, IPAM, PB, *TM, TargetMAM, IMAM});
  }

public:
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  runInstrumentationPipelineForDispatch(
      const llvm::amdhsa::kernel_descriptor_t &KD,
      llvm::OptimizationLevel Level = llvm::OptimizationLevel::O3) {
    llvm::PassInstrumentationCallbacks PIC;
    std::unique_ptr<llvm::MemoryBuffer> Result;
    LUTHIER_RETURN_ON_ERROR(withLiftedDispatch(
        KD, Level, PIC,
        [&](LiftedDispatch L) -> llvm::Error {
          Derived &D = derived();
          luthier::Prototype &IP = L.IP;
          luthier::PrototypeAnalysisManager &IPAM = L.IPAM;
          luthier::InstrumentationPassBuilder &PB = L.PB;
          llvm::ModuleAnalysisManager &TargetMAM = L.TargetMAM;
          llvm::ModuleAnalysisManager &IMAM = L.IMAM;
      llvm::SmallVector<char, 0> ObjBuf;
      llvm::raw_svector_ostream ObjOS(ObjBuf);

      llvm::CGPassBuilderOption CGPBO = llvm::getCGPassBuilderOption();

      luthier::PrototypePassManager IPPM;
      LUTHIER_RETURN_ON_ERROR(PB.buildInstrumentationPipeline(
          IPPM,
          // The instrumentation stage: the tool's own payload injection, then any
          // extra IR-level passes it asks for.
          [&D](luthier::PrototypePassManager &PPM, llvm::OptimizationLevel) {
            PPM.addPass(InjectPayloadsAdapter(&D));
            if constexpr (requires(Derived &Tool) {
                            Tool.preIROptimizationPasses(PPM);
                          })
              D.preIROptimizationPasses(PPM);
          },
          Level, llvm::CodeGenFileType::ObjectFile, CGPBO, &ObjOS, &PIC));

      // ParentPrototypeAnalysis is consumed via getCachedResult, so materialize
      // it for both modules up front, each in its own manager.
      (void)IMAM.getResult<luthier::ParentPrototypeAnalysis>(
          IP.getInstrumentationModule());
      (void)TargetMAM.getResult<luthier::ParentPrototypeAnalysis>(
          IP.getTargetModule());

      IPPM.run(IP, IPAM);

      Result = std::make_unique<llvm::SmallVectorMemoryBuffer>(
          std::move(ObjBuf), "luthier.instrumented",
          /*RequiresNullTerminator=*/false);
      return llvm::Error::success();
        }));
    return Result;
  }

  //===--------------------------------------------------------------------===//
  // Payload injection helpers
  //===--------------------------------------------------------------------===//
  //
  // These used to live on HSATool. Nothing in them is specific to HSA: they
  // resolve a host shadow handle to a device function through the tool's own
  // offload parser and hand it to the Prototype. Keeping them there meant a tool
  // below the runtime could observe dispatches and lift kernels but not inject
  // anything into them -- which is the whole of instrumentation.

  /// Resolve a payload function's host shadow handle to the corresponding
  /// \c llvm::Function inside the given instrumentation module.
  ///
  /// \c lookupHandleName returns the device-side mangled name as
  /// recorded by \c ToolDeviceCodeOffloadParserPass. For \c __global__ kernels
  /// (HIP \c __hipRegisterFunction path) the recorded name is the
  /// kernel's natural Itanium mangling; for tagged \c __device__
  /// functions (CXX-plugin export-handle path) the IR pass already
  /// demangles the synthesized host sibling and stores the original
  /// device function's Itanium-mangled name. In both cases a single
  /// \c Module::getFunction lookup against the IModule resolves the
  /// payload.
  ///
  /// The handle is taken as a typed pointer so callers can pass
  /// \c &MyTool::myHook directly; \c lookupHandleName does the cast to the
  /// opaque key internally.
  template <typename T>
  llvm::Expected<llvm::Function *>
  resolvePayloadHandle(T *HostHandle, llvm::Module &InstrumentationModule) {
    auto &Self = static_cast<Derived &>(*this);
    auto NameOrErr = Self.lookupHandleName(HostHandle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());

    if (llvm::Function *F = InstrumentationModule.getFunction(*NameOrErr))
      return F;

    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Payload function '{0}' not present in the instrumentation module.",
        *NameOrErr));
  }

  /// Convenience overload that takes a HIP host-shadow handle (the
  /// pointer used by HIP to reference a \c __device__ function from the
  /// host side, e.g. \c &MyTool::myHook) instead of a pre-resolved
  /// \c llvm::Function. Resolves the handle via \c resolvePayloadHandle,
  /// pulls the instrumentation module's \c FunctionAnalysisManager off
  /// \p PAM, then forwards to \c Prototype::createInjectedPayload. The
  /// handle is taken as a typed pointer so callers need not cast to
  /// \c void*.
  template <typename T>
  llvm::Error createInjectedPayload(T *HostHandle, Prototype &P,
                                    PrototypeAnalysisManager &PAM,
                                    llvm::MachineInstr &TargetMI,
                                    llvm::ArrayRef<PayloadArg> Args = {}) {
    auto FnOrErr = resolvePayloadHandle(HostHandle, P.getInstrumentationModule());
    LUTHIER_RETURN_ON_ERROR(FnOrErr.takeError());
    llvm::FunctionAnalysisManager &IFAM =
        PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
            .getManager();
    return P.createInjectedPayload(**FnOrErr, TargetMI, IFAM, Args).takeError();
  }

  /// Lambda-taking companion of the above: resolves \p HostHandle to a
  /// \c llvm::Function, hands both the resolved hook and an \c IRBuilder
  /// pointing into the freshly-created payload body to \p Build. Use this
  /// when the payload needs arguments that must be materialized inside the
  /// payload's own function (e.g. \c luthier::readReg intrinsic calls),
  /// which cannot be prepared before the payload's BB exists.
  template <typename T>
  llvm::Error createInjectedPayload(
      T *HostHandle, Prototype &P, PrototypeAnalysisManager &PAM,
      llvm::MachineInstr &TargetMI,
      llvm::function_ref<llvm::Error(llvm::Function &, llvm::IRBuilderBase &)>
          Build) {
    auto FnOrErr = resolvePayloadHandle(HostHandle, P.getInstrumentationModule());
    LUTHIER_RETURN_ON_ERROR(FnOrErr.takeError());
    llvm::FunctionAnalysisManager &IFAM =
        PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
            .getManager();
    llvm::Function &Hook = **FnOrErr;
    return P.createInjectedPayload(
                 TargetMI, IFAM,
                 [&](llvm::IRBuilderBase &Builder) -> llvm::Error {
                   return Build(Hook, Builder);
                 })
        .takeError();
  }


  /// \brief Lift the kernel described by \p KD and hand the result to \p Visit,
  /// without instrumenting anything.
  ///
  /// \par Why this exists next to the full pipeline
  /// \c buildInstrumentationPipeline always continues past code discovery into
  /// payload injection, the instrumentation module's IR pipeline, AMDGPU codegen
  /// and asm printing. A tool that only analyses has no payload to inject, so all
  /// of that is work whose output it discards -- and it is work that assumes an
  /// instrumentation module with something in it.
  ///
  /// This runs code discovery and stops. What \p Visit receives is the lifted
  /// target MIR and every analysis over it, which is what an analysis tool wants
  /// and the whole of what a tool below the runtime can currently do: loading a
  /// modified kernel back needs the HSA-based loader, which has no equivalent
  /// below HSA yet.
  ///
  /// \par Why the target module's analysis manager is handed over too
  /// The lifted code is \c MachineFunction s, and reaching them from an
  /// \c llvm::Function needs the \c MachineModuleInfo, which lives in that
  /// manager. Without it a caller can see that discovery returned no error but
  /// cannot look at what it produced -- and those are very different statements:
  /// \c CodeDiscoveryPass reports an unresolvable address through the LLVM error
  /// handler and returns normally, so "no error" is satisfied by a run that
  /// produced nothing at all.
  ///
  /// \param Visit runs with everything alive. It must copy out anything it keeps.
  llvm::Error runCodeDiscoveryForDispatch(
      const llvm::amdhsa::kernel_descriptor_t &KD,
      llvm::function_ref<llvm::Error(luthier::Prototype &,
                                     luthier::PrototypeAnalysisManager &,
                                     llvm::ModuleAnalysisManager &)>
          Visit) {
    llvm::PassInstrumentationCallbacks PIC;
    return withLiftedDispatch(
        KD, llvm::OptimizationLevel::O0, PIC,
        [&](LiftedDispatch L) -> llvm::Error {
          luthier::PrototypePassManager PPM;
          PPM.addPass(luthier::CodeDiscoveryPass());
          // ParentPrototypeAnalysis is read with getCachedResult, so it has to
          // exist before any pass asks for it.
          (void)L.IMAM.template getResult<luthier::ParentPrototypeAnalysis>(
              L.IP.getInstrumentationModule());
          (void)L.TargetMAM.template getResult<luthier::ParentPrototypeAnalysis>(
              L.IP.getTargetModule());
          PPM.run(L.IP, L.IPAM);
          return Visit(L.IP, L.IPAM, L.TargetMAM);
        });
  }
};

} // namespace luthier

#endif // LUTHIER_TOOLING_INSTRUMENTATION_PIPELINE_TRAIT_H
