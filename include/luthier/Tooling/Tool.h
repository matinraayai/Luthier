//===-- Tool.h - Luthier Tool Trait -----------------------------*- C++ -*-===//
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
/// \file Tool.h
/// CRTP base class for Luthier tools.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_TOOL_H
#define LUTHIER_TOOLING_TOOL_H

#include "luthier/Common/Singleton.h"
#include "luthier/Tooling/InstrumentationPipelineTrait.h"
#include "luthier/Tooling/InstrumentedKernelLoaderAndLauncher.h"
#include "luthier/Tooling/LLVMUserTrait.h"
#include "luthier/Tooling/LoadedCodeObjectCache.h"
#include "luthier/Tooling/PacketMonitorTrait.h"
#include "luthier/PassPlugin/LuthierPassPlugin.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorRegistry.h"
#include "luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h"
#include <hsa/hsa_ext_amd.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

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
/// any destructor runs. A trait's HSA API-table interceptor therefore never
/// observes a half-destroyed tool.
///
/// \par Construction/teardown (see \c Singleton)
/// Because the trait constructors install HSA API-table interceptors that may
/// fire on runtime threads, an \c Tool must be constructed and destroyed via
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
class Tool : public Singleton<Derived>,
             public LLVMUserTrait<Derived>,
             public LoadedCodeObjectCacheTrait<Derived>,
             public ToolDeviceCodeOffloadParserTrait<Derived>,
             public InstrumentedKernelLoaderAndLauncherTrait<Derived>,
             public InstrumentationPipelineTrait<Derived, TargetUnitT>,
             public PacketMonitorTrait<Derived> {
public:
  Tool(typename Singleton<Derived>::CreationKey,
       const DynamicLibrary &RocprofilerLib,
       const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
       const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
       const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
           &VenLoader,
       llvm::Error &Err)
      : Singleton<Derived>(), LLVMUserTrait<Derived>(),
        LoadedCodeObjectCacheTrait<Derived>(RocprofilerLib, CoreApi, VenLoader,
                                            Err),
        ToolDeviceCodeOffloadParserTrait<Derived>(Err),
        InstrumentedKernelLoaderAndLauncherTrait<Derived>(
            RocprofilerLib, CoreApi, AmdExt, VenLoader, Err),
        InstrumentationPipelineTrait<Derived, TargetUnitT>(CoreApi, AmdExt,
                                                           VenLoader),
        PacketMonitorTrait<Derived>(RocprofilerLib, CoreApi, AmdExt, VenLoader,
                                    Err) {}

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
  ///
  /// \param LaneMode how many lanes the payload body runs on. Defaults to
  /// \c PayloadLaneMode::AllActiveLanes , which leaves the application's
  /// \c EXEC mask untouched. \c PayloadLaneMode::SingleLane runs the body
  /// with <tt>EXEC = 1</tt>, spilling the app's \c EXEC into the SVA's
  /// exec-mask spill lanes on entry and restoring it on exit, and switches
  /// the preserve pass to whole-wave save/restore of every vector register
  /// the payload clobbers.
  template <typename T>
  llvm::Error createInjectedPayload(
      T *HostHandle, Prototype &P, PrototypeAnalysisManager &PAM,
      llvm::MachineInstr &TargetMI, llvm::ArrayRef<PayloadArg> Args = {},
      PayloadLaneMode LaneMode = PayloadLaneMode::AllActiveLanes) {
    auto FnOrErr = resolvePayloadHandle(HostHandle, P.getInstrumentationModule());
    LUTHIER_RETURN_ON_ERROR(FnOrErr.takeError());
    llvm::FunctionAnalysisManager &IFAM =
        PAM.getResult<IModuleFunctionAnalysisManagerPrototypeProxy>(P)
            .getManager();
    return P.createInjectedPayload(**FnOrErr, TargetMI, IFAM, Args, LaneMode)
        .takeError();
  }

  /// Lambda-taking companion of the above: resolves \p HostHandle to a
  /// \c llvm::Function, hands both the resolved hook and an \c IRBuilder
  /// pointing into the freshly-created payload body to \p Build. Use this
  /// when the payload needs arguments that must be materialized inside the
  /// payload's own function (e.g. \c luthier::readReg intrinsic calls),
  /// which cannot be prepared before the payload's BB exists.
  ///
  /// \param LaneMode how many lanes the payload body runs on; see the
  /// \p Args -taking overload above. \c PayloadLaneMode::SingleLane pins the
  /// body to one lane by way of \c ExecuteSingleLaneAttribute — \c EXEC is
  /// spilled to the SVA's exec-mask spill lanes, forced to 1 for the body,
  /// and restored afterwards, and clobbered vector registers are preserved
  /// whole-wave instead of only across the originally active lanes.
  template <typename T>
  llvm::Error createInjectedPayload(
      T *HostHandle, Prototype &P, PrototypeAnalysisManager &PAM,
      llvm::MachineInstr &TargetMI,
      llvm::function_ref<llvm::Error(llvm::Function &, llvm::IRBuilderBase &)>
          Build,
      PayloadLaneMode LaneMode = PayloadLaneMode::AllActiveLanes) {
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
                 },
                 LaneMode)
        .takeError();
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

#endif // LUTHIER_TOOLING_TOOL_H
