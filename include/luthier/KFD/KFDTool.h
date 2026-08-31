//===-- KFDTool.h -----------------------------------------------*- C++ -*-===//
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
/// CRTP base for a Luthier tool attached to an application that drives the KFD
/// driver directly, with no GPU runtime above it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_KFD_TOOL_H
#define LUTHIER_KFD_KFD_TOOL_H
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/hsa.h"

#include <hsa/hsa.h>
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"
#include "luthier/HSATooling/InstrumentationPipelineTrait.h"
#include "luthier/HSATooling/InstrumentedKernelLoaderAndLauncher.h"
#include "luthier/HSATooling/LLVMUserTrait.h"
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/KFD/FdSharing.h"
#include "luthier/KFD/KfdAgent.h"
#include "luthier/KFD/KfdAllocationResolver.h"
#include "luthier/KFD/KfdPacketMonitorTrait.h"
#include "luthier/KFD/KfdTargetMachine.h"
#include "luthier/KFD/QueueWrapper.h"
#include "luthier/ToolCodeGen/IntrinsicProcessorRegistry.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"

#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>

#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <mutex>

namespace luthier {

/// \brief CRTP base for a static Luthier tool below the runtime.
///
/// \par What it is, stated as a difference from \c HSATool
/// The same instrumentation pipeline, reached through three different doors.
/// \c HSATool composes seven traits; this composes five, and the two it drops
/// are exactly the two that need a runtime:
///
/// | \c HSATool trait | here |
/// | --- | --- |
/// | \c PacketMonitorTrait | replaced by \c KfdPacketMonitorTrait, which reads the ring buffer the preloaded wrapper substituted |
/// | \c ToolDeviceCodeOffloadParserTrait | \b opt-in, see below |
/// | everything else | the same |
///
/// That table used to be longer. The loaded-code-object cache and the
/// instrumented-kernel loader were both dropped on the belief that HSA could not
/// exist in such a process, and they are back because it can -- see below. So the
/// only real difference between a tool attached to an HSA application and one
/// attached to an application that drives the driver itself is \e where
/// \e dispatches \e come \e from.
///
/// \par Why the offload parser is opt-in here and not in \c HSATool
/// It carries the tool's own device code, so a tool that injects payloads needs
/// it and should inherit it alongside this class:
/// \code
///   class MyTool : public luthier::KFDTool<MyTool>,
///                  public luthier::ToolDeviceCodeOffloadParserTrait<MyTool> {
/// \endcode
/// A tool that only \e reads has no device code of its own, and inheriting the
/// trait anyway is not free: its static fields are defined by
/// \c LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES, whose
/// \c __attribute__((managed)) requires the translation unit to be compiled as
/// HIP. Composing it unconditionally would therefore force every analysis-only
/// tool to be a HIP translation unit on account of a base class it never uses.
/// Analysis-only tools are the ones this class can fully serve today, so making
/// them the awkward case would be backwards.
///
/// \par How HSA ends up available, which it is not by default
/// Three per-process driver resources collide when a second party initializes in
/// an application that already claimed them, and all three had to be resolved
/// before \c hsa_init could succeed here. Each needed a different mechanism,
/// which is worth knowing before assuming a fourth would yield to the same trick:
///
/// \li the \b DRM \b address \b space -- one VM per GPU per process. Resolved by
///     handing HSA the descriptor \c ACQUIRE_VM already bound, since the kernel
///     refuses a second \e space rather than a second \e call
///     (\c FdSharing.h);
/// \li \b runtime \b enable -- refused with \c EEXIST once queues exist. Absorbed
///     in the wrapper, because \c EEXIST means the runtime is enabled and the
///     caller is merely late;
/// \li the \b event \b page -- per-process, and hsakmt allocates its own before
///     the ioctl and then indexes into it, so the page cannot be shared the way
///     the descriptor was. Resolved by keeping ROCr away from it entirely with
///     \c HSA_ENABLE_INTERRUPT=0, which makes it use busy-wait signals rather
///     than the KFD events that need the page.
///
/// \c ensureHsaInitialized does all of this, once, on the first dispatch.
///
/// \par Why initialization is late rather than from a constructor
/// The application must claim those resources first. Initializing HSA first does
/// not avoid the collisions, it only moves them onto the application -- measured,
/// the application's \c ACQUIRE_VM then fails with \c EBUSY, and tinygrad does
/// not guard that call. A failure on our side can be reported; a failure on
/// theirs is a crash in someone else's program.
///
/// \par Construction
/// \p Derived must provide
/// <tt>onDispatchPacket(const kfd::QueueInfo &, uint64_t, hsa::AqlPacket &)</tt>.
/// A tool that injects payloads must additionally provide
/// <tt>run(Prototype &, PrototypeAnalysisManager &)</tt> and a source for the
/// instrumentation module -- either \c ToolDeviceCodeOffloadParserTrait, which
/// supplies \c parseModule, or its own \c createInstrumentationModule.
/// A tool that only reads needs neither, because
/// \c runCodeDiscoveryForDispatch never reaches the instrumentation module.
template <typename Derived, typename TargetUnitT = llvm::MachineFunction>
class KFDTool : public Singleton<Derived>,
                public LLVMUserTrait<Derived>,
                public LoadedCodeObjectCacheTrait<Derived>,
                public InstrumentedKernelLoaderAndLauncherTrait<Derived>,
                public IntrinsicProcessorRegistryTraitBase<Derived>,
                public InstrumentationPipelineTrait<Derived, TargetUnitT>,
                public KfdPacketMonitorTrait<Derived> {
public:
  KFDTool(typename Singleton<Derived>::CreationKey,
          const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
          const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
          const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
              &VenLoader,
          llvm::Error &Err)
      : Singleton<Derived>(), LLVMUserTrait<Derived>(),
        LoadedCodeObjectCacheTrait<Derived>(CoreApi, VenLoader, Err),
        InstrumentedKernelLoaderAndLauncherTrait<Derived>(CoreApi, AmdExt,
                                                         VenLoader, Err),
        KfdPacketMonitorTrait<Derived>(Err) {}

  /// \brief Bring HSA up inside this application, once.
  ///
  /// Called on the first dispatch rather than from a constructor: the application
  /// has to claim the driver's per-process resources first, or it is the party
  /// that fails. See the class comment for the three collisions this works
  /// around and why each needs a different mechanism.
  ///
  /// \note Safe to call repeatedly; only the first call does anything.
  llvm::Error ensureHsaInitialized() {
    llvm::Error Err = llvm::Error::success();
    std::call_once(HsaInitOnce, [&] {
      // Everything HSA does from here creates queues and allocations inside an
      // application that did not ask for them. Without this the wrapper would
      // treat the runtime's own queues as the application's and feed our
      // dispatches to our own callback.
      //
      // Process-wide, not the per-thread region: bringing the runtime up creates
      // queues on threads it spawns itself, so a thread-local flag does not
      // cover them. Measured -- with the per-thread region the runtime's queue
      // was wrapped as the process's second queue.
      kfd::ProcessWideToolRegion Region;

      // Redirect HSA's render-node opens onto the descriptor the application
      // already had bound. Enabled now rather than at load time because there is
      // nothing to redirect to until the application has claimed a GPU.
      kfd::enableFdSharing();

      // Keep ROCr off the application's KFD event page. It reads its flags while
      // constructing the runtime, which happens inside hsa_init, so setting this
      // here is in time.
      setenv("HSA_ENABLE_INTERRUPT", "0", /*overwrite=*/1);

      // Called directly rather than through a captured API table, which is the
      // convention everywhere else in a Luthier tool. It cannot be followed
      // here: the tables are captured *by* HSA initializing, so reading the
      // snapshot first is circular -- and reading one that was never captured is
      // a fatal error, so the circularity presents as the tool killing the
      // application from inside a packet callback rather than as a bad result.
      //
      // The convention exists to stop a tool re-entering its own wrappers. That
      // risk does not apply to this one call, because there is nothing wrapped
      // until it returns.
      const hsa_status_t St = ::hsa_init();
      if (St != HSA_STATUS_SUCCESS)
        Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "hsa_init failed with status {0} inside an application that drives "
            "the KFD driver. Run with LUTHIER_VERBOSE=1: the wrapper prints "
            "every failing KFD ioctl with its number and errno, which is what "
            "identifies a per-process driver resource the application already "
            "claimed.",
            static_cast<int>(St)));
    });
    return Err;
  }

  /// \brief The HSA agent for the GPU whose dispatch is being handled.
  ///
  /// The loader needs an agent, and cannot get one from the kernel descriptor:
  /// an application that allocates through the driver leaves
  /// \c hsa_amd_pointer_info reporting the descriptor as owned by nothing. The
  /// queue names the device instead.
  llvm::Expected<hsa_agent_t> agentForCurrentDispatch() {
    const uint32_t GpuId = KfdPacketMonitorTrait<Derived>::getDispatchGpuId();
    if (GpuId == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "No dispatch is in flight on this thread, so the GPU it would run on "
          "is unknown. Outside a packet callback there is nothing that names the "
          "device.");
    return kfd::agentForGpuId(this->getCoreApiTableSnapshot().getTable(), GpuId);
  }

  /// \brief The accessor this tool's pipeline uses.
  ///
  /// Both sources, and both are load-bearing here -- which is the reason this is
  /// not the driver-only accessor it once was. Two different kinds of kernel get
  /// looked up during one run:
  ///
  /// \li the \b application's kernels, in allocations the driver handed out and
  ///     HSA has never heard of. Only the driver-level resolver can describe
  ///     those;
  /// \li our \b own instrumented kernels, which are loaded through HSA once the
  ///     pipeline has produced them. The HSA loader names those exactly, with a
  ///     parsed code object the driver-level resolver could never supply.
  ///
  /// So the accessor asks HSA first and falls through, which is what it was built
  /// to do -- it just happens that in this process the two halves answer for
  /// different halves of the work rather than one being a fallback for the other.
  std::unique_ptr<MemoryAllocationAccessor> createMemoryAllocationAccessor() {
    auto &D = static_cast<Derived &>(*this);
    return std::make_unique<HsaMemoryAllocationAccessor>(
        static_cast<const LoadedCodeObjectCache &>(D),
        this->getCoreApiTableSnapshot(), this->getAmdExtTableSnapshot(),
        this->getLoaderTableSnapshot(),
        std::make_unique<KfdAllocationResolver>());
  }

  /// \brief Build the \c TargetMachine for the kernel described by \p KD.
  ///
  /// The device comes from the queue the packet arrived on, not from \p KD: a
  /// kernel descriptor does not say where it will run, and below HSA there is no
  /// agent owning its allocation to ask. \c KfdPacketMonitorTrait records the
  /// \c gpu_id for the duration of the callback, which is the only window in
  /// which this question has an answer.
  llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
  buildTargetMachineForKD(const llvm::amdhsa::kernel_descriptor_t *KD) {
    // Still from sysfs rather than from the agent, even though an agent is now
    // reachable. Nothing is gained by routing it through HSA, and the sysfs path
    // is checked against HSA's own answer by
    // KfdIsaInfo.AgreesWithWhatHsaReports.
    const uint32_t GpuId = KfdPacketMonitorTrait<Derived>::getDispatchGpuId();
    if (GpuId == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "No dispatch is in flight on this thread, so the GPU a kernel would "
          "run on is unknown. A target machine can only be built while handling "
          "the packet that dispatches the kernel -- unlike the HSA path, where "
          "the kernel descriptor's owning agent names the device at any time.");
    return buildTargetMachineForKfdDispatch(GpuId, *KD);
  }


  /// Bring the launcher's name-based device-global lookup into scope alongside
  /// the host-handle overload below.
  using InstrumentedKernelLoaderAndLauncher::lookupGlobalVariable;

  /// \brief Resolve a device-global host shadow handle (e.g.
  /// \c &MyTool::MyDeviceVar) to its symbol inside the instrumented executable
  /// cached under <tt>(KD, Preset)</tt>.
  ///
  /// The same six lines as \c HSATool's, and duplicated rather than shared
  /// because it needs two things no single trait has: \c lookupHandleName from
  /// the offload parser, which is opt-in here, and the launcher's lookup. Sharing
  /// it would mean a trait that depends on both, which is more coupling than six
  /// lines are worth.
  template <typename T>
  llvm::Expected<hsa_executable_symbol_t>
  lookupGlobalVariable(T *Handle, const llvm::amdhsa::kernel_descriptor_t *KD,
                       uint64_t Preset = 0) {
    // Through Derived, not through this: lookupHandleName comes from the
    // offload parser trait, which is opt-in and therefore a *sibling* base of
    // Derived rather than a base of this class.
    auto NameOrErr = static_cast<Derived *>(this)->lookupHandleName(Handle);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    return InstrumentedKernelLoaderAndLauncher::lookupGlobalVariable(
        *NameOrErr, KD, Preset);
  }

private:
  /// Guards \c ensureHsaInitialized. The first dispatch on any thread brings HSA
  /// up; the rest go straight through.
  std::once_flag HsaInitOnce;
};

} // namespace luthier

#endif // LUTHIER_KFD_KFD_TOOL_H
