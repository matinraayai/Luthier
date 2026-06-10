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
/// \file InstrumentationPipelineTrait.h
/// CRTP trait that owns the per-dispatch instrumentation pipeline shared by
/// every HSA tool: it builds the target \c Module / analysis managers,
/// registers the common instrumentation analyses, and drives
/// \c CodeDiscoveryPass -> \c InstrumentationPMDriver -> \c NewPMAsmPrinter to
/// produce an instrumented relocatable for a dispatched kernel.
///
/// The trait forwards a set of optional, plugin-style callbacks to the
/// \c InstrumentationPMDriver. Each callback is detected on \c Derived via a
/// \c requires-expression; if \c Derived does not define a given method, the
/// corresponding driver callback is a no-op. The detected customization
/// points (all optional) are:
///   - \c onBeginDispatchInstrumentation()
///   - \c createInstrumentationModule(llvm::LLVMContext &)
///   - \c augmentInstrumentationPassBuilder(llvm::PassBuilder &)
///   - \c preIROptimizationPasses(llvm::ModulePassManager &)
///   - \c preLuthierIRIntrinsicLoweringPasses(llvm::ModulePassManager &)
///   - \c postLuthierIRIntrinsicLoweringPasses(llvm::ModulePassManager &)
///   - \c augmentTargetPassConfig(llvm::PassRegistry &,
///        llvm::TargetPassConfig &, llvm::TargetMachine &)
///   - \c registerInstrumentationAnalyses(llvm::ModuleAnalysisManager &,
///        llvm::MachineFunctionAnalysisManager &)
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INSTRUMENTATION_PIPELINE_TRAIT_H
#define LUTHIER_TOOLING_INSTRUMENTATION_PIPELINE_TRAIT_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSATooling/DeviceToolCodeParser.h"
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/ToolCodeGen/CodeDiscoveryPass.h"
#include "luthier/ToolCodeGen/EntryPoint.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/InitialExecutionPointAnalysis.h"
#include "luthier/ToolCodeGen/InstructionTracesAnalysis.h"
#include "luthier/ToolCodeGen/InstrumentationPMDriver.h"
#include "luthier/ToolCodeGen/InstrumentationPass.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/Metadata.h"
#include "luthier/ToolCodeGen/MetadataParserAnalysis.h"
#include "luthier/ToolCodeGen/NewPMAsmPrinter.h"
#include "luthier/ToolCodeGen/PrePostAmbleEmitter.h"
#include "luthier/ToolCodeGen/RegValueMapAnalysis.h"
#include "luthier/ToolCodeGen/TraceCallGraph.h"
#include "luthier/ToolCodeGen/TraceIRTranslatorAnalysis.h"
#include "luthier/ToolCodeGen/TraceInstrMappingAnalysis.h"
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <memory>
#include <string>

namespace luthier {

/// \brief CRTP trait that runs Luthier's per-dispatch instrumentation pipeline.
///
/// \tparam Derived the concrete tool (an \c HSATool subclass). It must provide
/// \c buildTargetMachineForKD, \c getEmbeddedModule,
/// \c getIntrinsicProcessorRegistry, and be an \c InstrumentationPass for the
/// payload-injection adapter cast to succeed — all of which \c HSATool already
/// supplies.
/// \tparam TargetUnitT the instrumentation target unit (matches \c HSATool's).
template <typename Derived, typename TargetUnitT = llvm::MachineFunction>
class InstrumentationPipelineTrait {
  Derived &derived() { return static_cast<Derived &>(*this); }

  /// Thin module-pass adapter that forwards into the tool's
  /// \c InstrumentationPass<Derived, Module, TargetUnitT>::run() so the driver
  /// can hook the tool's payload-injection logic into its IR pipeline without
  /// copying the singleton tool object.
  struct InjectPayloadsAdapter
      : public llvm::PassInfoMixin<InjectPayloadsAdapter> {
    Derived *T;
    explicit InjectPayloadsAdapter(Derived *T) : T(T) {}
    llvm::PreservedAnalyses run(llvm::Module &IModule,
                                llvm::ModuleAnalysisManager &IMAM) {
      using Base =
          luthier::InstrumentationPass<Derived, llvm::Module, TargetUnitT>;
      return static_cast<Base *>(T)->run(IModule, IMAM);
    }
  };

public:
  /// Register the common set of instrumentation analyses on \p MAM / \p MFAM
  /// for the kernel described by \p KD. \p MMI and \p MDParser must outlive the
  /// pass run that consumes them. After the common analyses are registered, the
  /// tool's optional \c registerInstrumentationAnalyses(MAM, MFAM) hook (if
  /// present) is invoked so a tool can add its own.
  void registerInstrumentationAnalyses(
      const llvm::amdhsa::kernel_descriptor_t &KD, llvm::TargetMachine &TM,
      llvm::MachineModuleInfo &MMI, amdgpu::hsamd::MetadataParser &MDParser,
      llvm::ModuleAnalysisManager &MAM, llvm::FunctionAnalysisManager &FAM,
      llvm::MachineFunctionAnalysisManager &MFAM) {
    Derived &D = derived();
    (void)TM;

    MAM.registerPass([&] { return llvm::MachineModuleAnalysis(MMI); });
    MFAM.registerPass([] { return luthier::InstructionTracesAnalysis(); });
    MFAM.registerPass([] { return luthier::TraceInstrMappingAnalysis(); });
    MFAM.registerPass([] { return luthier::TraceIRTranslatorAnalysis(); });
    FAM.registerPass([] { return luthier::RegValueMapAnalysis(); });
    MAM.registerPass([&] {
      return luthier::InitialEntryPointAnalysis(
          [&](llvm::Module &, llvm::ModuleAnalysisManager &) {
            return luthier::EntryPoint(KD);
          });
    });
    MAM.registerPass([&] {
      return luthier::InitialExecutionPointAnalysis(
          [&](llvm::Module &, llvm::ModuleAnalysisManager &)
              -> const llvm::amdhsa::kernel_descriptor_t & { return KD; });
    });
    MAM.registerPass([&] {
      return luthier::MemoryAllocationAnalysis(
          std::make_unique<luthier::HsaMemoryAllocationAccessor>(
              static_cast<const LoadedCodeObjectCache &>(D),
              D.getCoreApiTableSnapshot(), D.getAmdExtTableSnapshot(),
              D.getLoaderTableSnapshot().getTable()));
    });
    MAM.registerPass([&] { return luthier::MetadataParserAnalysis(MDParser); });
    MAM.registerPass([] { return luthier::TraceCallGraphAnalysis(); });
    MAM.registerPass(
        [] { return luthier::FunctionPreambleDescriptorAnalysis(); });
    MAM.registerPass([] { return luthier::IPPredCFGAnalysis(); });

    if constexpr (requires(Derived &Tool) {
                    Tool.registerInstrumentationAnalyses(MAM, MFAM);
                  })
      D.registerInstrumentationAnalyses(MAM, MFAM);
  }

  /// Run \c CodeDiscoveryPass -> \c InstrumentationPMDriver ->
  /// \c NewPMAsmPrinter for the kernel referenced by \p KD and return the
  /// resulting relocatable object-file bytes.
  ///
  /// \p DriverOpts / \p DiscoveryOpts are the tool's per-pipeline options
  /// (a tool typically threads these from its own \c Options member).
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  runInstrumentationPipelineForDispatch(
      const llvm::amdhsa::kernel_descriptor_t &KD,
      const InstrumentationPMDriverOptions &DriverOpts,
      const CodeDiscoveryPassOptions &DiscoveryOpts) {
    Derived &D = derived();

    if constexpr (requires(Derived &Tool) {
                    Tool.onBeginDispatchInstrumentation();
                  })
      D.onBeginDispatchInstrumentation();

    std::unique_ptr<llvm::TargetMachine> TM;
    LUTHIER_RETURN_ON_ERROR(D.buildTargetMachineForKD(&KD).moveInto(TM));

    llvm::LLVMContext Ctx;
    auto M = std::make_unique<llvm::Module>("luthier.target", Ctx);
    M->setTargetTriple(TM->getTargetTriple());
    M->setDataLayout(TM->createDataLayout());

    llvm::MachineModuleInfo MMI(TM.get());

    // Declaration order matters: C++ destroys these in reverse, and MAM caches
    // a `FunctionAnalysisManagerModuleProxy::Result` whose destructor calls
    // `FAM->clear()`. So MAM must die before FAM. Keep MAM declared LAST among
    // the managers (same convention LLVM uses in `PassBuilder` examples).
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::MachineFunctionAnalysisManager MFAM;
    llvm::ModuleAnalysisManager MAM;

    // PIC + SI must outlive MPM.run(). StandardInstrumentations reads
    // --print-after-all / --print-before-all / --print-changed / -time-passes
    // and registers the corresponding PassInstrumentationCallbacks.
    llvm::PassInstrumentationCallbacks PIC;
    llvm::StandardInstrumentations SI(Ctx, /*DebugLogging=*/false);

    llvm::PassBuilder PB(TM.get(), llvm::PipelineTuningOptions(), std::nullopt,
                         &PIC);
    PB.registerModuleAnalyses(MAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerMachineFunctionAnalyses(MFAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);
    SI.registerCallbacks(PIC, &MAM);

    luthier::amdgpu::hsamd::MetadataParser MetadataParserInstance;
    registerInstrumentationAnalyses(KD, *TM, MMI, MetadataParserInstance, MAM,
                                    FAM, MFAM);

    llvm::SmallVector<char, 0> ObjBuf;
    llvm::raw_svector_ostream ObjOS(ObjBuf);

    llvm::Triple ToolTriple = TM->getTargetTriple();
    std::string ToolCPU(TM->getTargetCPU());
    llvm::SubtargetFeatures ToolFeatures(TM->getTargetFeatureString());

    llvm::ModulePassManager MPM;
    MPM.addPass(luthier::CodeDiscoveryPass(DiscoveryOpts));
    MPM.addPass(luthier::InstrumentationPMDriver(
        DriverOpts, D.getIntrinsicProcessorRegistry(), /*Plugins=*/{},
        // IModule creator: a tool may override via createInstrumentationModule;
        // the default materializes the tool's embedded device-side bitcode so
        // the hooks live in the IModule.
        [&D, ToolTriple, ToolCPU, ToolFeatures](
            llvm::LLVMContext &IModCtx) -> std::unique_ptr<llvm::Module> {
          if constexpr (requires(Derived &Tool) {
                          Tool.createInstrumentationModule(IModCtx);
                        }) {
            return D.createInstrumentationModule(IModCtx);
          } else {
            auto ModOrErr =
                D.parseModule(ToolTriple, ToolCPU, ToolFeatures, IModCtx);
            if (!ModOrErr)
              llvm::report_fatal_error(ModOrErr.takeError());
            return std::move(*ModOrErr);
          }
        },
        // PassBuilder augmentation (optional).
        [&D](llvm::PassBuilder &IPB) {
          if constexpr (requires(Derived &Tool) {
                          Tool.augmentInstrumentationPassBuilder(IPB);
                        })
            D.augmentInstrumentationPassBuilder(IPB);
        },
        // PreIROptimization: always insert the payload-injection adapter (its
        // payload functions reference the hooks, preventing GlobalDCE from
        // stripping them), then run the tool's optional extras.
        [&D](llvm::ModulePassManager &IMPM) {
          IMPM.addPass(InjectPayloadsAdapter(&D));
          if constexpr (requires(Derived &Tool) {
                          Tool.preIROptimizationPasses(IMPM);
                        })
            D.preIROptimizationPasses(IMPM);
        },
        // PreLuthierIRIntrinsicLowering (optional).
        [&D](llvm::ModulePassManager &IMPM) {
          if constexpr (requires(Derived &Tool) {
                          Tool.preLuthierIRIntrinsicLoweringPasses(IMPM);
                        })
            D.preLuthierIRIntrinsicLoweringPasses(IMPM);
        },
        // PostLuthierIRIntrinsicLowering (optional).
        [&D](llvm::ModulePassManager &IMPM) {
          if constexpr (requires(Derived &Tool) {
                          Tool.postLuthierIRIntrinsicLoweringPasses(IMPM);
                        })
            D.postLuthierIRIntrinsicLoweringPasses(IMPM);
        },
        // AugmentTargetPassConfig (optional).
        [&D](llvm::PassRegistry &PR, llvm::TargetPassConfig &TPC,
             llvm::TargetMachine &ITM) {
          if constexpr (requires(Derived &Tool) {
                          Tool.augmentTargetPassConfig(PR, TPC, ITM);
                        })
            D.augmentTargetPassConfig(PR, TPC, ITM);
        }));
    MPM.addPass(
        luthier::NewPMAsmPrinter(llvm::CodeGenFileType::ObjectFile, ObjOS));

    MPM.run(*M, MAM);

    return std::make_unique<llvm::SmallVectorMemoryBuffer>(
        std::move(ObjBuf), "luthier.instrumented",
        /*RequiresNullTerminator=*/false);
  }
};

} // namespace luthier

#endif // LUTHIER_TOOLING_INSTRUMENTATION_PIPELINE_TRAIT_H
