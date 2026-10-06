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

  /// Host-side resolver for the AMD hostcall FUNCTION_CALL service the
  /// PatchPCUsagesPass runtime resolver falls back to on a wave-side map
  /// miss. Drives Luthier's instrumentation pipeline synchronously against
  /// the callee whose runtime address is \p In[0], loads the resulting
  /// object, publishes the mapping into the on-device resolver table (so
  /// subsequent waves hit the fast path), and hands the instrumented
  /// entry-point address back to the wave in \p Out[0].
  ///
  /// In[] layout (from patchRegMultiFallback):
  ///   In[0] = current callee-reg value (original device-function address)
  ///   In[1] = dispatch packet address
  ///   In[2] = &EntryPointToTraceFunctionAddrMap on the device
  ///   In[3] = &EntryPointToTraceFunctionAddrMapSize on the device
  ///   In[4] = &EntryPointToTraceFunctionAddrMapMaxSize on the device —
  ///           unused (only read for logging).
  ///
  /// The newly-instrumented code object's \c
  /// initEntryPointToTraceFunctionAddrMap ctor appends its own seed entry
  /// to the runtime table (grow-if-needed) at load time — the callback
  /// itself never writes the map. It only reads the (possibly-grown) map
  /// back after the load and returns the \c FnHandleAddr the ctor
  /// installed for \c CalleeAddr in \p Out[0] .
  static void patchPCUsagesHostCallback(std::uint64_t Out[2],
                                        const std::uint64_t In[7]) {
    Out[0] = 0;
    Out[1] = 0;

    const uint64_t CalleeAddr = In[0];
    const uint64_t DispatchPacketAddr = In[1];
    const uint64_t MapPtrLocation = In[2];
    const uint64_t MapSizeLocation = In[3];

    if (DispatchPacketAddr == 0 || CalleeAddr == 0) {
      luthier::errs() << "[InstrCountTool] patchPCUsagesHostCallback: null "
                         "dispatch packet or callee address; giving up\n";
      return;
    }

    const auto *DispatchPacket =
        reinterpret_cast<const hsa_kernel_dispatch_packet_t *>(
            DispatchPacketAddr);
    // What the wave is actually running: by the time this callback fires,
    // overrideWithInstrumented has already rewritten kernel_object, so the
    // packet names the *instrumented* KD. Normalized to the application's KD
    // below — see getOriginalKernelDescriptor.
    const auto *DispatchedKD =
        reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
            DispatchPacket->kernel_object);

    Singleton<Derived>::withInstance(
        [&](Derived &T) {
          const auto Core = T.getCoreApiTableSnapshot().getTable();

          // The enclosing kernel for everything below has to be the *original*
          // application KD, not the instrumented one the packet points at. The
          // launcher's caches are keyed by the original KD, and
          // CodeDiscoveryPass takes the initial execution point from it — a
          // device function lifted against the instrumented KD would inherit
          // the instrumented kernel's register usage as the application's
          // launch budget (luthier-app-num-vgpr / -sgpr) instead of what the
          // wave really launched with.
          const llvm::amdhsa::kernel_descriptor_t *EnclosingKD =
              T.getOriginalKernelDescriptor(DispatchedKD);
          if (EnclosingKD == nullptr) {
            luthier::errs()
                << "[InstrCountTool] patchPCUsagesHostCallback: dispatched "
                   "kernel_object 0x"
                << llvm::utohexstr(reinterpret_cast<uint64_t>(DispatchedKD))
                << " maps to no cached original kernel descriptor; giving up\n";
            return;
          }

          struct EntryPointTraceFunctionEntry {
            uint64_t TraceAddr;
            uint64_t FnHandleAddr;
          };

          auto DevRead = [&](uint64_t DevAddr, void *HostBuf, size_t Bytes) {
            return Core.template callFunction<hsa_memory_copy>(
                HostBuf, reinterpret_cast<void *>(DevAddr), Bytes);
          };

          // Walk the on-device resolver table for \c TraceAddr == \c CalleeAddr
          // and return the matching \c FnHandleAddr . The runtime table is the
          // single source of truth for the caller/callee mapping — no
          // host-side loader cache to consult. Returns \c true iff we found
          // the entry and wrote its handle into \c Out[0] .
          auto TryResolveFromDevice = [&]() -> bool {
            uint64_t MapPtr = 0;
            uint32_t Size = 0;
            if (DevRead(MapPtrLocation, &MapPtr, sizeof(MapPtr)) !=
                    HSA_STATUS_SUCCESS ||
                DevRead(MapSizeLocation, &Size, sizeof(Size)) !=
                    HSA_STATUS_SUCCESS) {
              luthier::errs() << "[InstrCountTool] callback: failed to read "
                                 "on-device map metadata\n";
              return false;
            }
            const uint64_t EntryStride = sizeof(EntryPointTraceFunctionEntry);
            EntryPointTraceFunctionEntry Entry{};
            for (uint32_t I = 0; I < Size; ++I) {
              if (DevRead(MapPtr + I * EntryStride, &Entry, EntryStride) !=
                  HSA_STATUS_SUCCESS) {
                luthier::errs() << "[InstrCountTool] callback: failed to read "
                                   "on-device map entry "
                                << I << "\n";
                return false;
              }
              if (Entry.TraceAddr == CalleeAddr) {
                Out[0] = Entry.FnHandleAddr;
                return true;
              }
            }
            return false;
          };

          // Fast path: another wave brought this callee up under the same lock
          // hold, so its (CalleeAddr → FnHandleAddr) entry is already in the
          // table. This can happen when concurrent misses converge — the
          // device-side loop already runs before the host call, but that
          // walked the pre-append table; re-checking here after re-acquiring
          // the (still-held) lock catches the race.
          if (TryResolveFromDevice()) {
            luthier::errs() << "[InstrCountTool] callback: table hit for 0x"
                            << llvm::utohexstr(CalleeAddr) << " -> 0x"
                            << llvm::utohexstr(Out[0]) << "\n";
            return;
          }

          // Run the instrumentation pipeline over the device function starting
          // at CalleeAddr. Same TargetMachine / segment context as the
          // dispatch's enclosing kernel.
          std::unique_ptr<llvm::MemoryBuffer> ObjBuf;
          if (auto E = T.runInstrumentationPipelineForDeviceFunction(
                            *EnclosingKD, CalleeAddr)
                           .moveInto(ObjBuf)) {
            luthier::errs()
                << "[InstrCountTool] callback: pipeline failed for 0x"
                << llvm::utohexstr(CalleeAddr) << ": "
                << llvm::toString(std::move(E)) << "\n";
            return;
          }

          // Dump the pipeline output for offline inspection.
          {
            static std::atomic<unsigned> Seq{0};
            unsigned Idx = Seq.fetch_add(1);
            std::string Path =
                (llvm::Twine("/tmp/luthier-devfn-") + llvm::Twine(Idx) + ".o")
                    .str();
            std::error_code EC;
            llvm::raw_fd_ostream OS(Path, EC);
            if (!EC) {
              OS.write(ObjBuf->getBufferStart(), ObjBuf->getBufferSize());
              luthier::errs()
                  << "[InstrCountTool] callback: dumped devfn "
                  << ObjBuf->getBufferSize() << " bytes to " << Path << "\n";
            }
          }

          // Load the object linked against the enclosing kernel's
          // already-loaded Luthier globals. The load fires the new code
          // object's
          // \c initEntryPointToTraceFunctionAddrMap ctor, which appends its
          // (CalleeAddr → \c FnHandleAddr ) seed entry to the existing runtime
          // table under the caller's already-held map lock — that publish is
          // the entire point of loading here; the loader itself no longer
          // hands back an entry-point address.
          if (auto E = T.loadInstrumentedDeviceFunction(std::move(ObjBuf),
                                                        EnclosingKD, 0)) {
            luthier::errs() << "[InstrCountTool] callback: load failed for 0x"
                            << llvm::utohexstr(CalleeAddr) << ": "
                            << llvm::toString(std::move(E)) << "\n";
            return;
          }

          // The device table now carries the ctor-installed mapping. Read it
          // back (the ctor may have reallocated the base pointer during a
          // grow step, so \c MapPtr we sampled before the load — had we —
          // could have been stale). Do the lookup fresh.
          if (!TryResolveFromDevice()) {
            luthier::errs()
                << "[InstrCountTool] callback: no map entry for 0x"
                << llvm::utohexstr(CalleeAddr)
                << " after load; the new code object's ctor did not seed one\n";
            return;
          }
          luthier::errs() << "[InstrCountTool] callback: mapped 0x"
                          << llvm::utohexstr(CalleeAddr) << " -> 0x"
                          << llvm::utohexstr(Out[0]) << "\n";
        });
  }

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
      llvm::OptimizationLevel Level = llvm::OptimizationLevel::O3) {
    return runInstrumentationPipelineImpl(KD, luthier::EntryPoint(KD), Level);
  }

  /// Assemble and run the instrumentation pipeline for a device function
  /// reached at runtime by \p EnclosingKD 's dispatch, seeded to start
  /// discovery at \p DevFuncAddr rather than a kernel entry.
  ///
  /// Used from \c PatchPCUsagesPass 's host callback when a wave hits an
  /// indirect callee whose runtime address is not yet in the on-device
  /// resolver table — the callback synchronously lifts + instruments the
  /// callee, loads the resulting code object, and publishes the mapping.
  ///
  /// The execution point stays as \p EnclosingKD because segment allocation
  /// lookups and target-machine features are all keyed off the kernel this
  /// device function is being called from; the code-discovery seed however
  /// is the raw device address.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  runInstrumentationPipelineForDeviceFunction(
      const llvm::amdhsa::kernel_descriptor_t &EnclosingKD,
      uint64_t DevFuncAddr,
      llvm::OptimizationLevel Level = llvm::OptimizationLevel::O3) {
    return runInstrumentationPipelineImpl(
        EnclosingKD, luthier::EntryPoint(DevFuncAddr), Level);
  }

private:
  /// Shared body for the kernel and device-function pipeline entries. \p
  /// TargetMachineKD is the KD whose subtarget features and agent context
  /// the pipeline lowers against (always the kernel being dispatched, even
  /// for a mid-dispatch device-function bring-up). \p InitialEP is the
  /// entry point \c CodeDiscoveryPass will walk from — a kernel descriptor
  /// for a kernel launch, a raw device address for an indirect callee.
  /// \p Level is the IR optimization level.
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  runInstrumentationPipelineImpl(
      const llvm::amdhsa::kernel_descriptor_t &TargetMachineKD,
      const luthier::EntryPoint &InitialEP, llvm::OptimizationLevel Level) {
    Derived &D = derived();

    std::unique_ptr<llvm::TargetMachine> TM;
    LUTHIER_RETURN_ON_ERROR(
        D.buildTargetMachineForKD(&TargetMachineKD).moveInto(TM));

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
    // descriptor comes from. Entry point can be a kernel or a raw device
    // address (device-function bring-up path); execution point is always
    // the enclosing kernel, since segment/agent context is that kernel's.
    luthier::setInitialEntryPoint(*TargetM, InitialEP);
    luthier::setInitialExecutionPoint(*TargetM, TargetMachineKD);

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
        patchPCUsagesHostCallback, Level, llvm::CodeGenFileType::ObjectFile,
        CGPBO, &ObjOS, &PIC));

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
