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
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"
#include "luthier/HSATooling/InstrumentationPipelineTrait.h"
#include "luthier/HSATooling/InstrumentedKernelLoaderAndLauncher.h"
#include "luthier/HSATooling/LLVMUserTrait.h"
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/KFD/KfdAllocationResolver.h"
#include "luthier/HSATooling/PacketMonitorTrait.h"
#include "luthier/PassPlugin/LuthierPassPlugin.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorRegistry.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h"
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

/// \brief CRTP base for static HSA tools. Inherits the HIP fat-binary
/// registration slots and per-agent HSA executable state from
/// \c ToolDeviceCodeOffloadParser, and the per-process singleton identity
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
                public ToolDeviceCodeOffloadParserTrait<Derived>,
                public InstrumentedKernelLoaderAndLauncherTrait<Derived>,
                public IntrinsicProcessorRegistryTraitBase<Derived>,
                public InstrumentationPipelineTrait<Derived, TargetUnitT>,
                public PacketMonitorTrait<Derived> {
public:
  HSATool(typename Singleton<Derived>::CreationKey,
          const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
          const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
          const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
              &VenLoader,
          llvm::Error &Err)
      : Singleton<Derived>(), LLVMUserTrait<Derived>(),
        LoadedCodeObjectCacheTrait<Derived>(CoreApi, VenLoader, Err),
        ToolDeviceCodeOffloadParserTrait<Derived>(Err),
        InstrumentedKernelLoaderAndLauncherTrait<Derived>(CoreApi, AmdExt,
                                                          VenLoader, Err),
        PacketMonitorTrait<Derived>(CoreApi, AmdExt, VenLoader, Err) {}

  /// Build the accessor this tool's instrumentation pipeline should use.
  ///
  /// HSA first, then whatever the driver-level resolver knows. The resolver is
  /// constructed unconditionally and reports for itself whether it has records
  /// to serve, so an application that never touched KFD directly is unaffected
  /// by its presence — the accessor only consults it for addresses HSA does not
  /// manage.
  ///
  /// The four HSA references come off this tool and exist nowhere else, which is
  /// why this is a member here rather than something the pipeline trait builds.
  ///
  /// Snapshots are passed, not the tables they wrap: reading a snapshot whose
  /// registration callback never fired is a fatal error by design, and the
  /// accessor has to be able to test for that before reading.
  std::unique_ptr<MemoryAllocationAccessor> createMemoryAllocationAccessor() {
    auto &D = static_cast<Derived &>(*this);
    return std::make_unique<HsaMemoryAllocationAccessor>(
        static_cast<const LoadedCodeObjectCache &>(D),
        D.getCoreApiTableSnapshot(), D.getAmdExtTableSnapshot(),
        D.getLoaderTableSnapshot(),
        std::make_unique<KfdAllocationResolver>());
  }

  /// \note There is no longer a single "pipeline driver" pass to hand a target
  /// module pass manager. The instrumentation pipeline now runs over a
  /// \c Prototype (both modules at once) and is assembled by
  /// \c InstrumentationPassBuilder::buildInstrumentationPipeline; a tool reaches
  /// it through \c runInstrumentationPipelineForDispatch on
  /// \c InstrumentationPipelineTrait.

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
    const auto AmdExt = InstrumentedKernelLoaderAndLauncher::AmdExt.getTable();
    const auto Core = InstrumentedKernelLoaderAndLauncher::CoreApi.getTable();

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
      // TODO: add missing tgsplit feature
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

  /// Bring the launcher's name-based device-global lookup into scope alongside
  /// the host-handle convenience overload below.
  using InstrumentedKernelLoaderAndLauncher::lookupGlobalVariable;

  /// Resolve a device-global host shadow handle (e.g. \c &MyTool::MyDeviceVar)
  /// to its \c hsa_executable_symbol_t inside the instrumented executable
  /// cached under <tt>(KD, Preset)</tt>. Converts the handle to its device
  /// symbol name via \c lookupHandleName, then forwards to the launcher. The
  /// handle is taken as a typed pointer so callers can pass \c &MyTool::Var
  /// directly.
  template <typename T>
  llvm::Expected<hsa_executable_symbol_t>
  lookupGlobalVariable(T *Handle, const llvm::amdhsa::kernel_descriptor_t *KD,
                       uint64_t Preset = 0) {
    auto NameOrErr = this->lookupHandleName(Handle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    return InstrumentedKernelLoaderAndLauncher::lookupGlobalVariable(
        *NameOrErr, KD, Preset);
  }
};

} // namespace luthier

#endif // LUTHIER_TOOLING_HSA_TOOL_H
