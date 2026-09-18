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
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/hsa.h"

#include <hsa/hsa.h>
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"
#include "luthier/HSATooling/InstrumentationPipelineTrait.h"
#include "luthier/HSATooling/InstrumentedKernelLoaderAndLauncher.h"
#include "luthier/HSATooling/LLVMUserTrait.h"
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/HSATooling/PacketMonitorTrait.h"
#include "luthier/KFD/FdSharing.h"
#include "luthier/KFD/KfdAllocationResolver.h"
#include "luthier/KFD/KfdTargetMachine.h"
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
/// | \c PacketMonitorTrait | the same trait, doing more: as well as wrapping \c hsa_queue_create it intercepts the driver boundary and substitutes each queue's ring buffer |
/// | \c ToolDeviceCodeOffloadParserTrait | \b opt-in, see below |
/// | everything else | the same |
///
/// That first row used to name a second trait, \c KfdPacketMonitorTrait. The two
/// merged: a tool wants the same thing from both doors, and keeping two traits in
/// agreement by hand is what the merge removed.
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
/// \li the \b event \b page -- one per process, and only the party that supplies
///     it may do so. Resolved by \e ordering rather than by a flag: HSA is
///     brought up on the application's first \c /dev/kfd open, so hsakmt
///     supplies the page first and every later caller is given a slot in it.
///     \c HSA_ENABLE_INTERRUPT=0 is now only a fallback for an application that
///     already held a descriptor before the tool attached -- see
///     \c PacketMonitor::applicationClaimedEventPage.
///
/// \c PacketMonitorTrait::ensureHsaInitializedInApplication does all of this,
/// once.
///
/// \par Why initialization is late, and why it is not this class's job any more
/// The application must claim those resources first. Initializing HSA first does
/// not avoid the collisions, it only moves them onto the application -- measured,
/// the application's \c ACQUIRE_VM then fails with \c EBUSY, and tinygrad does
/// not guard that call. A failure on our side can be reported; a failure on
/// theirs is a crash in someone else's program.
///
/// It moved into the trait because the trait is what can see the right moment.
/// This class could only react to the first \e dispatch; the trait hooks
/// \c open, so it reacts to the application first touching \c /dev/kfd -- earlier,
/// and still after the application has claimed what it needs. The trait also
/// loads the runtime with \c dlmopen into \c LM_ID_BASE, because a tool is an
/// audit_hook plugin and therefore lives in the auditor's linker namespace; HSA
/// has to come up in the \e application's namespace, where the descriptors it
/// needs actually exist.
///
/// \par Construction
/// \p Derived must provide both of \c PacketMonitorTrait's callbacks, because
/// the trait opens both doors and fires each unconditionally:
/// <tt>onDispatchPacket(const typename PacketMonitorTrait<Derived>::QueueInfo &,
/// uint64_t, hsa::AqlPacket &)</tt> for packets copied out of a substituted
/// ring, and
/// <tt>onPackets(const hsa_queue_t &, uint64_t, llvm::ArrayRef<hsa::AqlPacket>,
/// hsa_amd_queue_intercept_packet_writer)</tt> for batches from an HSA intercept
/// queue -- which this kind of application does produce, once HSA has been
/// brought up inside it to load an instrumented kernel.
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
                public PacketMonitorTrait<Derived> {
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
        PacketMonitorTrait<Derived>(CoreApi, AmdExt, VenLoader, Err),
        AllocTracker(Err) {}

  /// \brief Bring HSA up in the application's linker namespace, once.
  ///
  /// Kept as a name a tool can call, but the work is the trait's: it hooks
  /// \c open, so it already knows when the application first touches the driver,
  /// and it is what holds the tool region that keeps the runtime's own queues out
  /// of our callback. See the class comment for the three collisions this works
  /// around and why each needs a different mechanism.
  ///
  /// \note Safe to call repeatedly; only the first call does anything.
  llvm::Error ensureHsaInitialized() {
    return PacketMonitorTrait<Derived>::ensureHsaInitializedInApplication();
  }

  /// \brief The HSA agent for the GPU that owns \p KD.
  ///
  /// The loader needs an agent, and cannot get one from HSA: an application that
  /// allocates through the driver leaves \c hsa_amd_pointer_info reporting the
  /// descriptor as owned by nothing. The driver does know, though -- the
  /// \c ALLOC_MEMORY_OF_GPU that produced the descriptor's memory named a
  /// \c gpu_id, and the allocation map recorded it -- so the descriptor's own
  /// address is what names the device.
  llvm::Expected<hsa_agent_t>
  agentForKernelDescriptor(const llvm::amdhsa::kernel_descriptor_t *KD) {
    auto GpuIdOrErr = gpuIdOwning(reinterpret_cast<uint64_t>(KD));
    LUTHIER_RETURN_ON_ERROR(GpuIdOrErr.takeError());
    return hsa::agentForGpuId(this->getCoreApiTableSnapshot().getTable(),
                              *GpuIdOrErr);
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
        std::make_unique<KfdAllocationResolver>(&AllocTracker),
        &AllocTracker);
  }

  /// \brief The driver-boundary watcher this tool owns.
  ///
  /// Owned here rather than by the accessor, which is rebuilt once per pipeline
  /// run: the tracker installs a process-wide audit hook in its constructor, so a
  /// per-run instance would register a hook per dispatch.
  [[nodiscard]] const kfd::AllocationTracker &getAllocationTracker() const {
    return AllocTracker;
  }

  /// \brief Build the \c TargetMachine for the kernel described by \p KD.
  ///
  /// The device comes from the queue the packet arrived on, not from \p KD: a
  /// kernel descriptor does not say where it will run, and below HSA there is no
  /// agent owning its allocation to ask. The driver does know: the allocation
  /// \p KD sits in was created by an \c ALLOC_MEMORY_OF_GPU naming a \c gpu_id,
  /// and \c kfd::AllocationTracker recorded it.
  llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
  buildTargetMachineForKD(const llvm::amdhsa::kernel_descriptor_t *KD) {
    // Still from sysfs rather than from the agent, even though an agent is now
    // reachable. Nothing is gained by routing it through HSA, and the sysfs path
    // is checked against HSA's own answer by
    // KfdIsaInfo.AgreesWithWhatHsaReports.
    auto GpuIdOrErr = gpuIdOwning(reinterpret_cast<uint64_t>(KD));
    LUTHIER_RETURN_ON_ERROR(GpuIdOrErr.takeError());
    return buildTargetMachineForKfdDispatch(*GpuIdOrErr, *KD);
  }


  /// \brief The GPU that owns the driver allocation \p Addr falls inside.
  ///
  /// \par Why the allocation and not the queue
  /// The pipeline reaches a tool as <tt>buildTargetMachineForKD(KD)</tt>, and a
  /// kernel descriptor does not say which device it will run on. On the HSA path
  /// that is recovered from the descriptor's owning agent. Below HSA there is no
  /// agent, but there is still an owner: \c ALLOC_MEMORY_OF_GPU carries a
  /// \c gpu_id, and the tracker recorded it when the application made the call.
  ///
  /// This replaced a thread-local holding the queue's \c gpu_id for the duration
  /// of a packet callback. The allocation is the better source on three counts:
  /// it is a property of the memory rather than of which thread is asking, so it
  /// survives the tool handing work to another thread; it answers for any
  /// address, not only while a callback is on the stack; and it cannot report the
  /// wrong device for a kernel allocated on one GPU and dispatched from a queue
  /// on another.
  llvm::Expected<uint32_t> gpuIdOwning(uint64_t Addr) const {
    if (auto A = AllocTracker.findAllocation(Addr))
      return A->GpuId;
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "No KFD allocation tracked by this tool contains address {0:x}, so the "
        "GPU that owns it is unknown. Either the allocation was made before the "
        "tool attached -- the map only records ALLOC_MEMORY_OF_GPU calls it "
        "observed -- or the address is not device memory.",
        Addr));
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

};

} // namespace luthier

#endif // LUTHIER_KFD_KFD_TOOL_H
