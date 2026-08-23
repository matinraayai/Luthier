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
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"
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
#include "luthier/ToolCodeGen/PrePostAmbleEmitter.h"
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
/// \tparam Derived the concrete tool (an \c HSATool subclass). It must provide
/// \c buildTargetMachineForKD, \c parseModule,
/// \c getIntrinsicProcessorRegistry, and be an \c InstrumentationPass for the
/// payload-injection adapter cast to succeed — all of which \c HSATool already
/// supplies.
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
    MAM.registerPass([&] {
      return luthier::MemoryAllocationAnalysis(
          std::make_unique<luthier::HsaMemoryAllocationAccessor>(
              static_cast<const LoadedCodeObjectCache &>(D),
              D.getCoreApiTableSnapshot(), D.getAmdExtTableSnapshot(),
              D.getLoaderTableSnapshot().getTable()));
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
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  runInstrumentationPipelineForDispatch(
      const llvm::amdhsa::kernel_descriptor_t &KD,
      llvm::OptimizationLevel Level = llvm::OptimizationLevel::O2) {
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
    } else {
      LUTHIER_RETURN_ON_ERROR(
          D.parseModule(ToolTriple, ToolCPU, ToolFeatures, Ctx)
              .moveInto(IModuleM));
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

    // PIC + SI must outlive the pipeline run. StandardInstrumentations reads
    // --print-after-all / --print-before-all / --print-changed / -time-passes
    // and registers the corresponding PassInstrumentationCallbacks.
    llvm::PassInstrumentationCallbacks PIC;
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

    return std::make_unique<llvm::SmallVectorMemoryBuffer>(
        std::move(ObjBuf), "luthier.instrumented",
        /*RequiresNullTerminator=*/false);
  }
};

} // namespace luthier

#endif // LUTHIER_TOOLING_INSTRUMENTATION_PIPELINE_TRAIT_H
