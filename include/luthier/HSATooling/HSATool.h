//===-- HSATool.h - Luthier HSA Tool Trait ----------------------*- C++ -*-===//
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
/// \file HSATool.h
/// CRTP base class for static Luthier HSA tools. Composes the per-tool traits
/// and exposes frequently used methods in tools written in HIP.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_HSA_TOOL_H
#define LUTHIER_TOOLING_HSA_TOOL_H

#include "luthier/Common/Singleton.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/HSA/ISA.h"
#include "luthier/HSATooling/DeviceToolCodeFatBinaryLoader.h"
#include "luthier/HSATooling/InstrumentationPipelineTrait.h"
#include "luthier/HSATooling/InstrumentedKernelLoaderAndLauncher.h"
#include "luthier/HSATooling/LLVMUserTrait.h"
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/HSATooling/PacketMonitorTrait.h"
#include "luthier/PassPlugin/LuthierPassPlugin.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/ToolCodeGen/InjectedPayloadCreationPass.h"
#include "luthier/ToolCodeGen/InstrumentationPMDriver.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorRegistry.h"
#include <hsa/hsa_ext_amd.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/AMDGPUTargetParser.h>

namespace luthier {

/// Per-tool trait that owns the \c IntrinsicProcessorRegistry the tool's
/// instrumentation pipeline consults during IR/MIR intrinsic lowering. The
/// registry default-constructs (its ctor auto-registers the built-in Luthier
/// intrinsics from \c IntrinsicRegistry.def), so the trait needs no
/// constructor arguments and never fails.
template <typename Derived> class IntrinsicProcessorRegistryTraitBase {
  IntrinsicProcessorRegistry Registry;

public:
  IntrinsicProcessorRegistry &getIntrinsicProcessorRegistry() {
    return Registry;
  }

  const IntrinsicProcessorRegistry &getIntrinsicProcessorRegistry() const {
    return Registry;
  }
};

/// \brief CRTP base for static HSA tools. Inherits the HIP fat-binary
/// registration slots and per-agent HSA executable state from
/// \c DeviceToolCodeFatBinaryLoader, and the per-process singleton identity
/// from \c Singleton<Derived>; composes the per-subsystem traits.
///
/// \c Singleton<Derived> is listed first so its subobject is constructed before
/// any trait and destroyed after all of them. Teardown safety itself does not
/// rely on this ordering: it comes from
/// \c Singleton<Derived>::destroyInstance(), which unpublishes the tool and
/// then waits for every in-flight \c withInstance() call to finish before
/// any/// destructor runs. A trait's HSA API-table interceptor therefore never
/// observes a half-destroyed tool.
///
/// \par Construction/teardown (see \c Singleton)
/// Because the trait constructors install HSA API-table interceptors that may
/// fire on runtime threads, an \c HSATool must be constructed and destroyed via
/// \c createInstance and \c destroyInstance from inside \c rocprofiler's
/// configure callback.
///
/// Installed HSA API-table wrappers are NOT uninstalled at tool teardown;
/// uninstalling a wrapper the runtime may still call would cause a race
/// condition. Instead, every trait wrapper does its tool-specific work inside
/// \c Singleton<Derived>::withInstance(), which keeps the tool alive via a
/// reference count for the duration of the call. It becomes a forwarding
/// function once the tool has been destroyed.
///
/// \warning Inside a \c withInstance() callback, call HSA only through the
/// captured \e snapshot tables (the underlying, pre-interception function
/// pointers held by each trait, e.g. \c CoreApiSnapshot / \c AmdExtSnapshot),
/// \b never through the live (wrapped) API table. Call each snapshot's
/// \c forceTriggerApiTableCallback method to force initialize the snapshot
/// tables before using them if needed.
template <typename Derived, typename TargetUnitT = llvm::MachineFunction>
class HSATool : public Singleton<Derived>,
                public LLVMUserTrait<Derived>,
                public LoadedCodeObjectCacheTrait<Derived>,
                public DeviceToolCodeFatBinaryLoader<Derived>,
                public InstrumentedKernelLoaderAndLauncherTrait<Derived>,
                public InjectedPayloadCreationPass<Derived, TargetUnitT>,
                public IntrinsicProcessorRegistryTraitBase<Derived>,
                public InstrumentationPipelineTrait<Derived, TargetUnitT>,
                public PacketMonitorTrait<Derived> {
private:
  /// Sentinel \c __managed__ variable. Its only purpose is to give HIP-Clang's
  /// host emission *something* to register so that every Luthier tool TU emits
  /// \c __hipRegisterFatBinary + \c __hip_fatbin_wrapper + the bundle bytes
  /// (via \c -fcuda-include-gpubinary) — even tools whose own code is purely
  /// passthrough
  /// \c inline lets the in-class initializer also serve as the definition;
  /// \c used keeps the host TU from optimizing it away
  static inline __attribute__((managed, used)) char LuthierMarker = 0;

public:
  HSATool(typename Singleton<Derived>::CreationKey,
          const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
          const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
          const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
              &VenLoader,
          llvm::Error &Err)
      : Singleton<Derived>(), LLVMUserTrait<Derived>(),
        LoadedCodeObjectCacheTrait<Derived>(CoreApi, VenLoader, Err),
        DeviceToolCodeFatBinaryLoader<Derived>(CoreApi, AmdExt, VenLoader, Err),
        InstrumentedKernelLoaderAndLauncherTrait<Derived>(
            CoreApi, AmdExt, VenLoader,
            static_cast<DeviceToolCodeLoader &>(*this), Err),
        PacketMonitorTrait<Derived>(CoreApi, AmdExt, VenLoader, Err) {
    /// Force instantiation of LuthierMarker so HIP-Clang's host emission
    /// sees the managed variable even in tools that have no managed/device
    /// statics of their own
    (void)&LuthierMarker;
  }

  /// Build the instrumentation pass pipeline driver for a target application
  /// module. The returned \c InstrumentationPMDriver is a
  /// \c PassInfoMixin-shaped pass to be added to a target module pass manager
  /// by the caller (\c MPM.addPass(tool.buildPipeline(opts, plugins))). The
  /// driver internally materializes an IModule, runs the IR-level
  /// instrumentation stages, lowers Luthier intrinsics, and drives the
  /// per-IModule MIR codegen pipeline.
  InstrumentationPMDriver
  buildPipeline(const InstrumentationPMDriverOptions &Opts,
                llvm::ArrayRef<PassPlugin> Plugins = {}) {
    return InstrumentationPMDriver(Opts, this->getIntrinsicProcessorRegistry(),
                                   Plugins);
  }

  /// Convenience: build the pipeline and immediately run it against \p
  /// TargetAppM. Equivalent to constructing a one-pass MPM that contains
  /// \c buildPipeline() and invoking it. Returned \c PreservedAnalyses
  /// reflects what the driver preserved on the target module's analyses.
  llvm::PreservedAnalyses
  runInstrumentationPipeline(llvm::Module &TargetAppM,
                             llvm::ModuleAnalysisManager &TargetMAM,
                             const InstrumentationPMDriverOptions &Opts,
                             llvm::ArrayRef<PassPlugin> Plugins = {}) {
    return buildPipeline(Opts, Plugins).run(TargetAppM, TargetMAM);
  }

  /// Build a fully-configured \c TargetMachine for the agent that owns the
  /// kernel referenced by \p KD. Resolves the owning agent via
  /// \c hsa_amd_pointer_info, then queries that agent's first supported ISA
  /// for the LLVM triple / CPU / subtarget-feature string. The agent ISA
  /// query alone yields only the architectural feature set (xnack, sramecc,
  /// ...); the per-kernel wavefront size and CU/WGP execution mode are encoded
  /// in the kernel descriptor, not the ISA, so they are folded into the
  /// feature string here. The lifted MIR depends on the subtarget reflecting
  /// both — EXEC-mask predication width follows the wavefront size, and the
  /// re-lowered KD's \c WGP_MODE / \c TG_SPLIT bits are derived from the
  /// \c cumode feature (see \c CodeDiscoveryPass). The returned machine is what
  /// the instrumentation codegen pipeline lowers against.
  llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
  buildTargetMachineForKD(const llvm::amdhsa::kernel_descriptor_t *KD) {
    const auto AmdExt = DeviceToolCodeLoader::AmdExtSnapshot.getTable();
    const auto Core = DeviceToolCodeLoader::CoreApiSnapshot.getTable();

    hsa_amd_pointer_info_t PointerInfo{};
    PointerInfo.size = sizeof(hsa_amd_pointer_info_t);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        AmdExt.template callFunction<hsa_amd_pointer_info>(
            const_cast<void *>(reinterpret_cast<const void *>(KD)),
            &PointerInfo, nullptr, nullptr, nullptr),
        "Failed to query HSA pointer info for kernel descriptor."));
    hsa_agent_t Agent = PointerInfo.agentOwner;

    llvm::SmallVector<hsa_isa_t, 1> Isas;
    LUTHIER_RETURN_ON_ERROR(
        luthier::hsa::agentGetSupportedISAs(Core, Agent, Isas));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        !Isas.empty(), "Agent reports no supported ISAs."));
    hsa_isa_t Isa = Isas.front();

    auto TripleOrErr = luthier::hsa::isaGetTargetTriple(Core, Isa);
    LUTHIER_RETURN_ON_ERROR(TripleOrErr.takeError());
    auto CPUOrErr = luthier::hsa::isaGetGPUName(Core, Isa);
    LUTHIER_RETURN_ON_ERROR(CPUOrErr.takeError());
    auto FeaturesOrErr = luthier::hsa::isaGetSubTargetFeatures(Core, Isa);
    LUTHIER_RETURN_ON_ERROR(FeaturesOrErr.takeError());

    // Fold the per-kernel wavefront size and CU/WGP execution mode out of the
    // kernel descriptor into the subtarget feature string. Both features only
    // exist on gfx10+; pre-gfx10 hardware is always wave64 and CU mode and has
    // no \c wavefrontsize32 / \c cumode subtarget features to set.
    if (llvm::AMDGPU::getIsaVersion(*CPUOrErr).Major >= 10) {
      const bool IsWave32 = AMDHSA_BITS_GET(
          KD->kernel_code_properties,
          llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
      FeaturesOrErr->AddFeature(IsWave32 ? "wavefrontsize32"
                                         : "wavefrontsize64");
      // WGP_MODE set => the kernel runs in WGP mode (cumode disabled); clear
      // => CU mode (cumode enabled).
      const bool IsWGPMode =
          AMDHSA_BITS_GET(KD->compute_pgm_rsrc1,
                          llvm::amdhsa::COMPUTE_PGM_RSRC1_GFX10_PLUS_WGP_MODE);
      FeaturesOrErr->AddFeature("cumode", /*Enable=*/!IsWGPMode);
    }

    std::string ErrMsg;
    const llvm::Target *TheTarget =
        llvm::TargetRegistry::lookupTarget(*TripleOrErr, ErrMsg);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        TheTarget != nullptr,
        llvm::formatv("TargetRegistry::lookupTarget failed for triple {0}: {1}",
                      TripleOrErr->str(), ErrMsg)));

    llvm::TargetOptions TMOpts;
    std::unique_ptr<llvm::TargetMachine> TM(TheTarget->createTargetMachine(
        *TripleOrErr, *CPUOrErr, FeaturesOrErr->getString(), TMOpts,
        /*RM=*/std::nullopt));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        TM != nullptr, "createTargetMachine returned nullptr."));
    TM->setOptLevel(llvm::CodeGenOptLevel::Default);
    return TM;
  }

  /// Resolve a payload function's host shadow handle to the corresponding
  /// \c llvm::Function inside the given instrumentation module.
  ///
  /// \c lookupNameByHandle returns the device-side mangled name as
  /// recorded by \c LoadHIPFATBinaryInfoPass. For \c __global__ kernels
  /// (HIP \c __hipRegisterFunction path) the recorded name is the
  /// kernel's natural Itanium mangling; for tagged \c __device__
  /// functions (CXX-plugin export-handle path) the IR pass already
  /// demangles the synthesized host sibling and stores the original
  /// device function's Itanium-mangled name. In both cases a single
  /// \c Module::getFunction lookup against the IModule resolves the
  /// payload.
  ///
  /// The handle is taken as a typed pointer so callers can pass
  /// \c &MyTool::myHook directly; \c lookupNameByHandle does the cast to the
  /// opaque key internally.
  template <typename T>
  llvm::Expected<llvm::Function *>
  resolvePayloadHandle(T *HostHandle, llvm::Module &InstrumentationModule) {
    auto &Self = static_cast<Derived &>(*this);
    auto NameOrErr = Self.lookupNameByHandle(HostHandle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());

    if (llvm::Function *F = InstrumentationModule.getFunction(*NameOrErr))
      return F;

    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Payload function '{0}' not present in the instrumentation module.",
        *NameOrErr));
  }

  /// Lift the inherited \c createInjectedPayload overloads from
  /// \c InjectedPayloadCreationPass into the public surface so tool code
  /// can call them directly from \c onPackets / \c runInstrumentationPass
  /// without going through the protected base.
  using InjectedPayloadCreationPass<Derived,
                                    TargetUnitT>::createInjectedPayload;

  /// Convenience overload that takes a HIP host-shadow handle (the
  /// pointer used by HIP to reference a \c __device__ function from the
  /// host side, e.g. \c &MyTool::myHook) instead of a pre-resolved
  /// \c llvm::Function. Resolves the handle via \c resolvePayloadHandle
  /// then forwards to the \c Function&-taking base overload. The handle is
  /// taken as a typed pointer so callers need not cast to \c void*.
  template <typename T>
  llvm::Error
  createInjectedPayload(T *HostHandle, llvm::Module &IModule,
                        const llvm::MachineInstr &TargetMI,
                        llvm::ArrayRef<typename InjectedPayloadCreationPass<
                            Derived, TargetUnitT>::PayloadArg>
                            Args = {}) {
    auto FnOrErr = resolvePayloadHandle(HostHandle, IModule);
    LUTHIER_RETURN_ON_ERROR(FnOrErr.takeError());
    return InjectedPayloadCreationPass<
        Derived, TargetUnitT>::createInjectedPayload(**FnOrErr, TargetMI, Args);
  }
};

} // namespace luthier

#endif // LUTHIER_TOOLING_HSA_TOOL_H
