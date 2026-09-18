//===-- PacketMonitorTrait.h -------------------------------------*- C++-*-===//
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
/// Header-only CRTP trait that delivers the application's AQL packets to a
/// tool.
/// \par One trait, two doors
/// An application that uses HSA can be intercepted at \c hsa_queue_create; one
/// that drives the KFD driver itself cannot, and has to be intercepted at the
/// driver boundary. These were two traits, and they are one now because a tool
/// wants the same thing from both and the split leaked: \c HSATool composed
/// \c PacketMonitorTrait and \c KFDTool composed \c KfdPacketMonitorTrait, so
/// every question about packet observation had two answers that had to be kept
/// in agreement by hand.
///
/// \li \b HSA \b door -- \c hsa_queue_create is wrapped through the rocprofiler
///     API tables, the queue is converted to an intercept queue, and \p Derived
///     receives whole batches:
///     \code
///       void onPackets(const hsa_queue_t &Q, uint64_t UserPacketIdx,
///                      llvm::ArrayRef<hsa::AqlPacket> Packets,
///                      hsa_amd_queue_intercept_packet_writer Writer);
///     \endcode
/// \li \b KFD \b door -- the queue's ring buffer is substituted at
///     \c AMDKFD_IOC_CREATE_QUEUE and packets are copied across one at a time,
///     so \p Derived receives them singly:
///     \code
///       void onDispatchPacket(const QueueInfo &Q, uint64_t PacketIndex,
///                             hsa::AqlPacket &Packet);
///     \endcode
///
/// Both doors are opened by the one constructor, and \p Derived must implement
/// both callbacks. Which door an application actually delivers packets through
/// is a property of that application, not of how the tool was built -- and an
/// application can use both, since one that drives KFD itself and then has HSA
/// brought up inside it will produce traffic on each. A tool that expects only
/// one door still has to say what it would do at the other, rather than leaving
/// the answer to whether a template member happened to be instantiated.
///
/// \par How the KFD door works
/// Sending a packet to a GPU queue is a plain memory write -- there is no
/// function call or system call to intercept. Creating a queue *is* a system
/// call, so that is the one place we can insert ourselves. At queue creation we
/// swap the ring buffer the GPU reads for one we own; the application keeps
/// writing its own. A polling thread copies each finished packet across, runs
/// \c onDispatchPacket in between, and writes the copy's header last. The GPU
/// ignores a slot whose header reads INVALID, so the callback is guaranteed to
/// run before the GPU can act on the packet.
///
/// \par Two signals, two jobs
/// Confusing these caused every detection bug in the KFD path:
/// \li The application's **write pointer** counts slots it has *claimed*. A
///     packet cannot exist without having been claimed, so this is a safe upper
///     bound on where to look -- and bounding the scan is what keeps the copy
///     loop finite. It is **not** a "finished" count: the application bumps it
///     before writing the packet (measured: caught mid-write in 33 of 6509
///     samples).
/// \li The slot's **header** says whether that slot is finished, because the
///     producer writes the header last. That is the commit test.
///
/// \par The marker is ours to establish
/// The HSA runtime promises its callers that every slot in a new queue starts
/// as INVALID (at \c amd_aql_queue.cpp:122). That is an HSA promise, not a
/// driver or hardware one -- an application that skips HSA gets whatever the
/// allocator left, and zero is a legal packet type. So we write the markers
/// ourselves at queue creation, and put one back after copying each packet.
///
/// \par Scope of the KFD door
/// Only \c COMPUTE_AQL queues are wrapped; PM4 and SDMA queues pass through
/// untouched, as are queues created inside a \c ToolRegion. Several callbacks
/// may be registered and each may edit a packet in place, but none may add or
/// remove packets. ROCr expresses both by handing packets to a writer and
/// simply not calling it; supporting that here would mean emitting a different
/// number of packets than the application submitted, which invalidates the
/// index-sequence check that is currently the strongest correctness signal the
/// test suite has.
///
/// \par Interception is LD_AUDIT, via audit_hook
/// \c ioctl and the \c open family are wrapped with \c
/// audit_hooks::register_wrap from the trait's constructor. The tool \e is the
/// audit_hook plugin: it is loaded into the auditor's linker namespace out of
/// \c AH_PLUGINS, so by the time this trait is constructed the \c audit_hooks
/// API is available and the registration is in time for the application's first
/// call.
///
/// This replaced GOTCHA, for a mechanical reason rather than a stylistic one.
/// GOTCHA rewrites GOT entries in place, which means flipping page protections
/// on live memory; in a process that also hosts rocprofiler-sdk's bundled
/// libgotcha that is two independent rewriters over the same slots, ordered
/// only by whichever constructor ran first. Under LD_AUDIT the dynamic linker
/// keeps ownership of the GOT and hands the auditor the original function
/// pointer at bind time, so there is nothing to race over and nothing to
/// resolve by search:
/// \c RealIoctl below is written by \c ld.so, not by \c dlsym.
///
/// One safety net went away with GOTCHA, deliberately. The old interceptor
/// counted recursion depth and killed the process at 32 with a message, because
/// a preloaded library that \e defines \c ioctl and chains with
/// \c dlsym(RTLD_NEXT) would call GOTCHA's wrapper, which would call it, until
/// the stack was gone. That cannot arise here: \c ld.so binds each caller once,
/// so a wrapper is not reachable from its own next link, and audit_hook's
/// trampoline additionally pauses hooks for the thread while our code runs
/// (\c tls_hooks_paused). The guard was for a mechanism that is no longer in
/// use.
///
/// \par Nothing here is shared between tools
/// Every field is an instance field. The KFD path used to be file-scope state
/// in
/// \c QueueWrapper.cpp -- one queue table, one callback chain, one poller
/// thread, one set of counters per process -- which meant two tools in one
/// process silently shared a tracking table and a 64-queue budget. The only
/// statics left are the ones the mechanism forces, and each is a member of
/// \c PacketMonitorTrait<Derived> rather than of \c PacketMonitor, so even
/// those are per-tool rather than per-process -- which is the other reason they
/// are in the template rather than the shared base:
/// \li the \c Real* function pointers, whose addresses are handed to \c ld.so
/// at
///     registration and must outlive any one instance, since audit_hook has no
///     unregister call;
/// \li \c ToolRegionDepth, which is \c thread_local and therefore cannot be a
///     non-static member.
///
/// \par Two classes, and where each lives
/// \li \c PacketMonitor -- non-templated, and everything that does not depend
///     on \p Derived: the queue table, the poller thread, the callback chain,
///     the ring substitution, the \c ioctl path. Declared here, compiled once
///     in
///     \c PacketMonitorTrait.cpp;
/// \li \c PacketMonitorTrait<Derived> -- inherits it, and adds only what cannot
///     be written without \p Derived: the audit hooks and the \c static
///     function pointers \c ld.so writes, the \c thread_local tool-region
///     depth, the HSA API-table wrapper, and the two calls into the tool
///     itself.
///
/// The cut is the three pure virtuals in \c PacketMonitor's protected section.
/// Keeping the driver half out of the template also keeps
/// \c linux/kfd_ioctl.h out of the public header of every Luthier tool.
///
/// \par Diagnostics
/// Per-packet logging goes through \c LLVM_DEBUG under
/// \c -debug-only=luthier-packet-monitor. Off by default, and worth leaving off
/// on a busy queue: it is one line per packet, and a runaway once produced 18.6
/// million lines and a 2.5 GB log. A per-queue summary is printed at teardown
/// either way.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_PACKET_MONITOR_TRAIT_H
#define LUTHIER_TOOLING_PACKET_MONITOR_TRAIT_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/AqlPacket.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/KFD/FdSharing.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"

#include "luthier/Audit/audit_hook.hpp"

#include <hsa/hsa_api_trace.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

#include <fcntl.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <memory>

namespace luthier {

/// \brief Everything about monitoring packets that does not depend on which
/// tool is doing the monitoring.
///
/// \par Why this is a separate, non-templated class
/// The queue table, the poller thread, the callback chain, the ring
/// substitution and the whole \c ioctl path are the same code whatever
/// \p Derived is, and compiling them into every tool that instantiates
/// \c PacketMonitorTrait bought nothing but build time -- and dragged
/// \c linux/kfd_ioctl.h into the public header of every Luthier tool. They live
/// here and are compiled once, in \c PacketMonitorTrait.cpp.
///
/// What is left in the template is exactly what cannot be: the pieces that
/// reach \p Derived, and the statics the interception mechanism forces
/// (\c PacketMonitorTrait::RealIoctl and friends, whose addresses \c ld.so
/// writes, and the \c thread_local tool-region depth). Those three points of
/// contact are the pure virtuals below.
///
/// \par Lifetime, which the split makes load-bearing
/// The poller thread calls \c deliverDispatchPacket and \c realIoctl, both
/// virtual. So it has to be stopped while \p Derived is still alive, which is
/// why \c shutdown() is called from \c ~PacketMonitorTrait rather than from
/// \c ~PacketMonitor -- by the time a base destructor runs, those overrides are
/// gone and the calls would be pure-virtual.
class PacketMonitor {
public:
  /// \brief Identifies a wrapped queue to a \c PacketCallback.
  struct QueueInfo {
    /// KFD's identifier for the GPU this queue belongs to.
    uint32_t GpuId;
    /// KFD's identifier for the queue. Unique within the process -- the driver
    /// resolves a queue from this alone (see \c kfd_ioctl_destroy_queue_args,
    /// which carries nothing else).
    uint32_t QueueId;
    /// Size of the ring in bytes.
    uint32_t RingByteSize;
    /// Number of 64-byte packet slots in the ring.
    uint32_t SlotCount;
  };

  /// \brief Invoked once for every packet the application commits, before the
  /// GPU can act on it.
  ///
  /// \param Q the queue the packet was submitted to
  /// \param PacketIndex the packet's position in the stream. Counts from zero
  /// and only increases; it is not a slot number
  /// \param Packet **our copy** of the packet. Edits made here are what the GPU
  /// executes; the application's own copy is left alone
  /// \param UserData the pointer supplied when this callback was registered
  ///
  /// \note Runs on the polling thread, not the application's, so it must be
  /// thread-safe with respect to the tool's own state.
  using PacketCallback = void (*)(const QueueInfo &Q, uint64_t PacketIndex,
                                  hsa::AqlPacket &Packet, void *UserData);

  /// \brief Identifies a registered callback, for removing it again.
  using CallbackHandle = int;

  /// Returned by \c addPacketCallback when there is no room left.
  static constexpr CallbackHandle InvalidCallbackHandle = -1;

  /// \brief Most callbacks that can be registered on one monitor at once.
  static constexpr unsigned MaxPacketCallbacks = 8;

  /// \brief One registered callback and its user pointer.
  struct CallbackEntry {
    PacketCallback CB;
    void *UserData;
  };
  virtual ~PacketMonitor();

  /// \brief Mark the whole process as running the tool's own code.
  ///
  /// \c beginToolRegion covers the calling thread, which is enough for a queue
  /// the runtime creates in response to \c hsa_queue_create -- measured, those
  /// appear on the calling thread. It is \b not enough for \c hsa_init:
  /// bringing the runtime up spawns its own threads and creates queues on them,
  /// and measured in a KFD application the runtime's queue was wrapped as the
  /// process's second queue, after which the tool would instrument its own
  /// dispatches.
  ///
  /// Use for that window only. While it is open a queue the \e application
  /// creates is excluded too, which is wrong in general and acceptable for the
  /// length of one runtime initialization.
  void beginProcessWideToolRegion() { ProcessToolRegionDepth.fetch_add(1); }

  /// \brief End the region opened by \c beginProcessWideToolRegion.
  void endProcessWideToolRegion();

  /// \brief Scoped form of \c beginProcessWideToolRegion /
  /// \c endProcessWideToolRegion.
  class ProcessWideToolRegion {
  public:
    explicit ProcessWideToolRegion(PacketMonitor &Monitor) : Monitor(Monitor) {
      Monitor.beginProcessWideToolRegion();
    }
    ~ProcessWideToolRegion() { Monitor.endProcessWideToolRegion(); }
    ProcessWideToolRegion(const ProcessWideToolRegion &) = delete;
    ProcessWideToolRegion &operator=(const ProcessWideToolRegion &) = delete;

  private:
    PacketMonitor &Monitor;
  };

  //===--------------------------------------------------------------------===//
  // Counters
  //===--------------------------------------------------------------------===//

  /// \brief How many AQL queues were left alone because a tool region was open.
  ///
  /// The counterpart to \c wrappedQueueCount, and the reason both exist: a test
  /// that only counts wrapped queues cannot tell "correctly excluded" from
  /// "never created". Cumulative.
  uint64_t excludedQueueCount() const {
    return __atomic_load_n(&ExcludedQueueTotal, __ATOMIC_ACQUIRE);
  }

  /// \brief How many AQL queues this monitor has substituted a ring for, ever.
  ///
  /// Cumulative, not live, so a destroyed queue still counts.
  ///
  /// Exists because whether a queue was wrapped is otherwise only observable
  /// when packets flow through it -- and the case that matters most is one
  /// where they may not. When a tool initialises HSA, the runtime creates AQL
  /// queues on the tool's behalf; we must leave those alone, but they can sit
  /// empty, so no callback would ever fire either way. A count separates
  /// "correctly ignored" from "wrapped, but idle".
  uint64_t wrappedQueueCount() const {
    return __atomic_load_n(&WrappedQueueTotal, __ATOMIC_ACQUIRE);
  }

  /// \brief How many ioctls have passed through this monitor.
  ///
  /// The stronger guarantee: a non-zero count is evidence that traffic was
  /// actually seen, which the fact that registration succeeded does not
  /// establish.
  uint64_t interceptedIoctlCount() const {
    return __atomic_load_n(&InterceptedIoctlTotal, __ATOMIC_RELAXED);
  }

  //===--------------------------------------------------------------------===//
  // The callback chain
  //===--------------------------------------------------------------------===//

  /// \brief Add a callback without disturbing the ones already registered.
  ///
  /// More than one component may want a turn at each packet -- Luthier
  /// alongside a profiler, say -- and the HSA runtime supports exactly that, so
  /// a driver-level replacement that did not would be a step backwards.
  ///
  /// \par Order
  /// **Last registered runs first**, and each callback sees the edits made by
  /// the ones that ran before it. This is ROCr's order, deliberately: it walks
  /// its interceptor list from the end (\c intercept_queue.cpp:375), so the
  /// most recently attached tool sees the packet as the application wrote it,
  /// and the earliest-attached tool sees it last, just before the GPU does.
  /// Choosing the opposite order would make a tool behave differently depending
  /// on which interception layer it was attached to.
  ///
  /// \par What this does not do
  /// A callback here cannot drop a packet or emit extra ones. ROCr expresses
  /// both by having a callback pass packets to a writer, and not calling it
  /// drops them. That is deferred: emitting a different number of packets than
  /// the application submitted invalidates the index-sequence check, which is
  /// currently the strongest correctness signal the test suite has, so it needs
  /// its own verification story first.
  ///
  /// \return a handle, or \c InvalidCallbackHandle if the chain is full
  CallbackHandle addPacketCallback(PacketCallback CB, void *UserData);

  /// \brief Remove a callback added by \c addPacketCallback.
  ///
  /// Leaves the order of the remaining callbacks unchanged.
  void removePacketCallback(CallbackHandle H);

  /// \brief Replace the whole chain with a single callback.
  ///
  /// Passing \c nullptr removes every callback, after which packets are copied
  /// through unchanged.
  void setPacketCallback(PacketCallback CB, void *UserData);

  /// \brief Run a chain over one packet, last registered first.
  ///
  /// A static function over a plain array, rather than something that reads
  /// this object, so the ordering guarantee above can be tested without a GPU,
  /// a queue or a driver -- which is where a claim like "last registered runs
  /// first" belongs, since it is otherwise the kind of thing that is asserted
  /// in a comment and never checked.
  ///
  /// Entries whose \c CB is null are skipped, which is how removal works
  /// without shuffling the others.
  static void runCallbackChain(const CallbackEntry *Entries, unsigned Count,
                               const QueueInfo &Q, uint64_t PacketIndex,
                               hsa::AqlPacket &Packet);

  /// \brief Run every registered callback over one packet, last registered
  /// first.
  ///
  /// This is what the copier calls for each packet. It is public so the
  /// registration bookkeeping -- handle reuse, the published count, what
  /// \c setPacketCallback does to an existing chain -- can be exercised without
  /// a GPU. That bookkeeping is where the mistakes live, and testing only the
  /// walk over a hand-built array would leave all of it uncovered.
  void runRegisteredCallbacks(const QueueInfo &Q, uint64_t PacketIndex,
                              hsa::AqlPacket &Packet);

  //===--------------------------------------------------------------------===//
  // The driver boundary
  //===--------------------------------------------------------------------===//

  /// \brief Handle one intercepted \c ioctl.
  ///
  /// Deals with queue creation and destruction, allocation tracking and the two
  /// per-process resources a late initializer collides with, and forwards
  /// everything else. Public so the logic can be driven from a test without an
  /// audit engine in the process.
  ///
  /// \return whatever the underlying \c ioctl returned
  int handleIoctl(int Fd, unsigned long Request, void *Arg);

  //===--------------------------------------------------------------------===//
  // Bringing HSA up where it has to live
  //===--------------------------------------------------------------------===//

  /// \brief Load and initialize the HSA runtime in the \e application's linker
  /// namespace, once.
  ///
  /// \par Why the namespace is the whole point
  /// The tool is an audit_hook plugin, so it and everything it links -- LLVM
  /// included -- live in the auditor's namespace, isolated from the application
  /// by design. HSA cannot live there. It talks to the driver through
  /// descriptors the application opened and through a DRM address space the
  /// application bound, and a second copy of the runtime in a second namespace
  /// would be a second party contending for per-process resources rather than a
  /// passenger on the application's. So it is loaded with \c dlmopen into \c
  /// LM_ID_BASE: the application's namespace, where it sees the application's
  /// descriptors.
  ///
  /// \par Why this is triggered by the application touching KFD
  /// Called the first time the application opens \c /dev/kfd or a render node,
  /// rather than from a constructor. The ordering is forced: the application
  /// must claim the driver's per-process resources first. Initializing HSA
  /// earlier does not avoid the collisions, it only moves them onto the
  /// application -- measured, the application's \c ACQUIRE_VM then fails with
  /// \c EBUSY, and tinygrad does not guard that call. A failure on our side can
  /// be reported; a failure on theirs is a crash in someone else's program.
  /// Hooking the \e open is what makes "the application has started using the
  /// driver" observable at the earliest point where it is also already true.
  ///
  /// \par The three collisions this works around
  /// Each needed a different mechanism, which is worth knowing before assuming
  /// a fourth would yield to the same trick:
  /// \li the \b DRM \b address \b space -- one VM per GPU per process. Resolved
  /// by
  ///     handing HSA the descriptor \c ACQUIRE_VM already bound, since the
  ///     kernel refuses a second \e space rather than a second \e call (\c
  ///     FdSharing.h);
  /// \li \b runtime \b enable -- refused with \c EEXIST once queues exist.
  ///     Absorbed in \c handleIoctl, because \c EEXIST means the runtime is
  ///     enabled and the caller is merely late;
  /// \li the \b event \b page -- per-process, and hsakmt allocates its own
  /// before
  ///     the ioctl and then indexes into it, so the page cannot be shared the
  ///     way the descriptor was. Resolved by keeping ROCr away from it entirely
  ///     with
  ///     \c HSA_ENABLE_INTERRUPT=0, which makes it use busy-wait signals rather
  ///     than the KFD events that need the page.
  ///
  /// \note Safe to call repeatedly; only the first call does anything.
  llvm::Error ensureHsaInitializedInApplication();

  /// \brief The handle for the HSA runtime living in the application's
  /// namespace, or \c nullptr before \c ensureHsaInitializedInApplication has
  /// run.
  void *getApplicationHsaHandle() const { return AppHsaHandle; }

  /// \brief Whether the application established the KFD event page before we
  /// could bring HSA up.
  ///
  /// \par What the event page is, and the one rule about it
  /// A process has exactly one KFD event page. The GPU writes an event id into
  /// a slot in it and then traps, and the kernel's interrupt handler turns that
  /// into a signal -- so it has to be memory the GPU can write, which is why
  /// hsakmt allocates a GPU-mapped buffer and hands the driver its handle
  /// (\c libhsakmt/src/events.c:102-112). Everyone else passes
  /// \c event_page_offset as 0 and is given a slot in whatever page already
  /// exists.
  ///
  /// The rule, measured on gfx1036 against a live driver:
  /// \li \c CREATE_EVENT with \c event_page_offset \c == \c 0 always succeeds
  ///     and returns the process's page in that field as an output
  ///     (\c 0x8000000000000000, \c KFD_MMAP_TYPE_EVENTS) plus a fresh slot. A
  ///     later joiner can \c mmap that offset on its own \c /dev/kfd descriptor
  ///     and read its slot;
  /// \li \c CREATE_EVENT \e supplying a page when one is already established is
  ///     refused with \c EINVAL.
  ///
  /// So the page is fully shareable, and exactly one party may supply it: the
  /// first one. hsakmt is the only party that supplies one.
  ///
  /// \par Why this is an ordering question and not a capability question
  /// Measured, with the application claiming the page first and interrupts left
  /// on: hsakmt's \c CREATE_EVENT is refused with \c EINVAL, and \c hsa_init
  /// then \b segfaults on one of its own threads -- \c CreateEvent returns
  /// \c nullptr, \c EventPool::alloc has nothing to give
  /// (\c interrupt_signal.cpp:50-58), and \c BindErrorHandlers' guard against
  /// that is an \c assert (\c runtime.cpp:2176), which a release build drops.
  /// That crash is what forcing \c HSA_ENABLE_INTERRUPT=0 was avoiding.
  ///
  /// With HSA brought up first, the same measurement runs clean: hsakmt's page
  /// is accepted, interrupt-backed \c hsa_signal_wait works, and the
  /// application joining afterwards gets its own slot and maps the page
  /// successfully. So the flag is not needed -- the ordering is.
  ///
  /// \par Which we get, and why
  /// The \c open hook brings HSA up when the application first opens
  /// \c /dev/kfd, which is necessarily before it can issue any \c ioctl on that
  /// descriptor. hsakmt therefore supplies the page first as a matter of
  /// construction rather than of luck.
  ///
  /// This flag records the case that ordering cannot cover: an application that
  /// already held a \c /dev/kfd descriptor, and had already created an event,
  /// before the tool attached. Then interrupts really are unavailable to ROCr,
  /// and busy-wait signals are the difference between slow and crashed.
  bool applicationClaimedEventPage() const {
    return ApplicationClaimedEventPage;
  }

protected:
  //===--------------------------------------------------------------------===//
  // The three points of contact with the templated half
  //
  // Each is here because of something that cannot be expressed without
  // \c Derived. Kept as narrow as possible: everything else about packet
  // monitoring is in this class.
  //===--------------------------------------------------------------------===//

  /// \brief Forward to the next \c ioctl in the audit chain.
  ///
  /// The next link is a \c static member of the templated half, because
  /// \c ld.so is handed its address at registration and audit_hook has no
  /// unregister call, so the storage has to outlive any one instance.
  virtual int realIoctl(int Fd, unsigned long Request, void *Arg) = 0;

  /// \brief Whether this thread is inside a \c ToolRegion.
  ///
  /// The depth is \c thread_local and therefore cannot be a non-static member;
  /// being a member of the templated half keeps it one per tool rather than one
  /// per process.
  [[nodiscard]] virtual bool insideThreadToolRegion() const = 0;

  /// \brief Hand one packet to \p Derived's \c onDispatchPacket.
  virtual void deliverDispatchPacket(const QueueInfo &Q, uint64_t PacketIndex,
                                     hsa::AqlPacket &Packet) = 0;

  /// \brief Stop polling, print the per-queue summaries and give every
  /// substituted ring back.
  ///
  /// Must be called from the derived destructor, not this one. See the class
  /// comment.
  void shutdown();

  static constexpr const char *LogPrefix = "[luthier-kfd] ";

  /// Most queues one monitor tracks at a time.
  static constexpr int MaxTrackedQueues = 64;

private:
  /// \brief Lifecycle of one tracking-table entry.
  ///
  /// A plain "active" flag is not enough once entries are reused. The poller
  /// reads entries without the lock, so "nobody will ever touch this again" is
  /// a different statement from "this queue is gone", and reuse depends on the
  /// first.
  enum SlotState : int {
    /// Never handed out, or handed out for a creation that then failed. Either
    /// way the poller has never looked inside it, so it is reusable at once.
    SlotFree = 0,
    /// Claimed by a creation in progress. Not yet safe to poll: the fields are
    /// still being filled in.
    SlotReserved = 1,
    /// A live wrapped queue. The poller works on exactly these.
    SlotLive = 2,
    /// The queue was destroyed. The poller skips it, but a pass that was
    /// already
    /// inside it may still be running, so it cannot be reused immediately.
    SlotDead = 3,
  };

  struct ForwardedQueue {
    /// The application's own ring. It writes here; after substitution nobody
    /// else reads it but us.
    volatile unsigned char *AppRing;
    /// Our ring, registered with the GPU. The GPU reads here.
    volatile unsigned char *ShimRing;
    /// The application's claim counter. Read-only to us.
    volatile uint64_t *AppWritePointer;
    QueueInfo Info;
    /// Next packet of the **application's** stream to copy. Counts packets, not
    /// slots. This is the number a callback is given, so a tool's packet
    /// numbering matches the application's.
    uint64_t Consumed;
    /// Next slot of **our** ring to write. Counts packets, not slots.
    ///
    /// Equal to \c Consumed today, and separated from it deliberately rather
    /// than because they differ. The two mean different things -- one indexes
    /// the application's stream, the other our output -- and they stop being
    /// equal the moment a callback can emit a different number of packets than
    /// it was given. Keeping one counter for both hid that behind an arithmetic
    /// coincidence; the places that would have to change for 1-to-N are now
    /// exactly the places that mention this field.
    ///
    /// They cannot actually diverge yet, and not just because nothing
    /// increments them differently: the GPU tracks its progress against the
    /// application's own write pointer, which we do not intercept. Emitting a
    /// different count means taking that over too.
    uint64_t Produced;
    uint64_t DispatchCount;
    /// Largest observed gap between the application's claim counter and
    /// \c Consumed. A value above \c Info.SlotCount means the application
    /// wrapped the ring while we were behind, so packets were overwritten
    /// before we copied them -- see \c OverrunPackets.
    uint64_t MaxLag;
    /// Packets the application wrote and we never copied, because it lapped us.
    ///
    /// Should always be zero: the application blocks once its ring is full, and
    /// it cannot free a slot without the GPU consuming our copy of it. Kept as
    /// a standing check on that reasoning rather than as a known failure.
    ///
    /// Note this is *not* what the S10b hang was -- \c MaxLag stayed at or
    /// below
    /// \c Info.SlotCount throughout, which is what ruled an overrun out and
    /// pointed at the re-arm ordering in \c forwardOnePacket instead.
    ///
    /// Counted rather than corrected, if it ever fires. Correcting it means
    /// writing something harmless into our ring for every skipped index -- the
    /// GPU reads those slots regardless, since its read pointer follows the
    /// application's write pointer -- and getting that wrong hangs the GPU
    /// instead of losing a count.
    uint64_t OverrunPackets;
    int State;      ///< one of SlotState
    int Summarized; ///< so the teardown summary prints once
    /// Value of \c PollPass when this queue was destroyed, which is what
    /// decides when the entry may be reused. Only ever touched under \c
    /// QueueLock.
    uint64_t DeadAtPass;

    //=== What it takes to give the substitute ring back ====================//
    /// The descriptor CREATE_QUEUE arrived on, needed to undo the registration.
    int Fd;
    /// Driver handle from ALLOC_MEMORY_OF_GPU.
    uint64_t RingHandle;
    /// Start of the mapping, and its **page-rounded** length -- not the ring
    /// size the application asked for, which would under-unmap.
    void *RingPages;
    size_t RingPagesBytes;
  };

  /// Identify /dev/kfd by device number rather than descriptor number, since
  /// the application chooses its own descriptors.
  ///
  /// Assumption A2: this runs an fstat on **every ioctl in the process**, not
  /// just KFD ones, because the hook covers the symbol process-wide. The device
  /// number is cached, so the cost is one fstat per call and no more. An
  /// earlier design filtered on the request number first, which is free; that
  /// was dropped, and reinstating it is fine for the request (a scalar) but
  /// would be a real bug if the filter touched Arg, since nothing has yet
  /// established that the descriptor is ours.
  bool fdIsKfd(int Fd);

  //===--------------------------------------------------------------------===//
  // Our substitute ring
  //===--------------------------------------------------------------------===//

  /// mmap rather than malloc: we need page alignment for the GPU registration
  /// below, and pages that do not share space with the allocator's bookkeeping.
  ///
  /// \param OutSize receives the **rounded** length. Unmapping needs it: the
  /// application's ring size is not page-aligned in general, and munmap with
  /// the unrounded figure leaves part of the mapping behind.
  void *allocRingPages(size_t MinSize, size_t *OutSize);

  /// \param OutHandle receives the driver's handle for the allocation, which is
  /// what releaseRing needs to give it back.
  bool registerRingWithGpu(int Fd, void *Va, size_t Size, uint32_t GpuId,
                           uint64_t *OutHandle);

  /// Undo registerRingWithGpu and hand the pages back.
  ///
  /// Called when a tracking entry is recycled, not when the queue is destroyed.
  /// Deferring it keeps the application's DESTROY_QUEUE free of two extra
  /// driver calls, and bounds what is held at any moment to one dead ring per
  /// entry.
  ///
  /// Safe by then: the queue is gone, so the GPU is not reading the ring, and
  /// the entry's grace period has expired, so neither are we. Errors are
  /// reported but not acted on -- the application may already have closed the
  /// descriptor, and leaking a ring is better than failing a queue creation
  /// over it.
  void releaseRing(int Fd, uint64_t Handle, uint32_t GpuId, void *Pages,
                   size_t PagesBytes);

  /// Hand every ring still held back to the driver. Runs from the destructor,
  /// after the poller has stopped, so nothing can be reading them.
  void releaseAllRings();

  //===--------------------------------------------------------------------===//
  bool insideToolRegion() const;

  //===--------------------------------------------------------------------===//
  // Copying one packet
  //===--------------------------------------------------------------------===//

  void runCallback(ForwardedQueue &Q, uint64_t Index, hsa::AqlPacket &Packet);

  /// Copy application slot -> our slot, header last.
  ///
  /// The callback is deliberately run on a **staged copy** rather than on our
  /// slot directly. Our slot's header is held at INVALID for the whole
  /// operation
  /// -- that is the gate that stops the GPU acting early -- so a callback
  /// inspecting the slot would see every packet as INVALID rather than its real
  /// type. Staging costs one 64-byte copy and lets the callback see a complete,
  /// coherent packet, including the header, which it may also edit.
  void forwardOnePacket(ForwardedQueue &Q, uint64_t Index);

  //===--------------------------------------------------------------------===//
  // The polling thread
  //===--------------------------------------------------------------------===//

  static void *pollerTrampoline(void *Self);

  void pollerMain();

  /// Stop the poller and join it.
  ///
  /// Required now that the queue table is an instance field: the thread reads
  /// this object, so returning from the destructor while it runs would hand it
  /// freed memory. While the table was file-scope the thread could simply
  /// outlive everything and be reaped by process exit.
  void stopPoller();

  /// Wait until no callback that was already running can still be running.
  ///
  /// Nulling a callback slot stops the poller *starting* another call, but says
  /// nothing about one already in flight: \c runRegisteredCallbacks loads the
  /// function pointer and then calls it, so a poller that got past the load is
  /// still inside the tool's code. A tool deregisters from its own destructor,
  /// so without this wait the rest of that object -- vtable included -- is
  /// destroyed underneath a live call. That presented as \c "pure virtual
  /// method called" at exit in \c S10b-four-producers, whose four producer
  /// threads keep the poller busy right up to process exit.
  ///
  /// Same clock and same argument as \c reclaimQueueSlot: \c PollPass counts
  /// completed passes, an in-flight callback belongs to a pass numbered at or
  /// below the one we read, so one further completed pass is sufficient and we
  /// wait for two.
  ///
  /// Bounded, and deliberately not a guarantee. If the poller has stopped, or
  /// if a tool calls this from inside its own callback (which cannot make
  /// progress by definition), we give up after 5 ms rather than hang the
  /// application on its way out.
  void drainPacketCallbacks();

  //===--------------------------------------------------------------------===//
  // Queue bookkeeping
  //===--------------------------------------------------------------------===//

  /// Reuse the entry of a queue that has been destroyed.
  ///
  /// \par Why a grace period rather than a lock
  /// The poller walks entries without taking \c QueueLock -- deliberately,
  /// since it runs continuously and would otherwise contend with every \c ioctl
  /// the application makes. So marking an entry dead does not mean the poller
  /// has stopped reading it: a pass that already tested the state and moved on
  /// to the ring pointers is still in flight. Overwriting the entry underneath
  /// that pass would point it at a freed ring.
  ///
  /// The clock is \c PollPass, the count of completed passes.
  /// \c deactivateQueue stores \c SlotDead first and only then records the pass
  /// number, so any pass that could still be inside the entry is numbered at or
  /// below the recorded one; waiting for one more completed pass is therefore
  /// sufficient. We wait for two, which costs about 40 microseconds and removes
  /// the need to trust that argument in review.
  ///
  /// \return an index whose entry is now \c SlotReserved, or -1.
  int reclaimQueueSlot();

  /// Claim a tracking slot BEFORE the ring is substituted.
  ///
  /// Order matters. Substituting the ring and then finding we cannot track the
  /// queue is the worst outcome available: the GPU would read our ring, nothing
  /// would ever fill it, and the application would wait forever on work that
  /// can never run -- a silent hang with no error anywhere.
  ///
  /// Prefers a fresh entry and falls back to reusing a dead one. Reuse is not
  /// an optimisation: without it a process that creates and destroys queues
  /// loses interception permanently once it has created \c MaxTrackedQueues of
  /// them over its whole lifetime, even with only one alive at a time.
  int reserveQueueSlot();

  /// Give a reserved slot back after a creation that did not happen.
  ///
  /// Without this every failed CREATE_QUEUE would burn an entry permanently:
  /// reserved is neither live nor dead, so nothing would ever reclaim it.
  void releaseQueueSlot(int Idx);

  void commitQueueSlot(int Idx, volatile unsigned char *AppRing,
                       volatile unsigned char *ShimRing,
                       volatile uint64_t *AppWritePointer,
                       const QueueInfo &Info, int Fd, uint64_t RingHandle,
                       size_t RingPagesBytes);

  /// One line per queue at teardown, so the numbers worth checking survive with
  /// per-packet logging off. Counts are final by then: the GPU cannot complete
  /// work we never copied, so an application that waited for its results has
  /// necessarily waited for us.
  void summarizeQueue(ForwardedQueue &Q);

  void summarizeAll();

  /// Stop the poller touching a queue's ring before the application frees it.
  ///
  /// Matching on queue id alone is correct, not a simplification: DESTROY_QUEUE
  /// passes nothing else, and neither do UPDATE_QUEUE, SET_CU_MASK or
  /// GET_QUEUE_WAVE_STATE. The driver resolves a queue from that id alone,
  /// which it could only do if the id is unique within the process.
  ///
  /// Only live entries are considered. Dead ones may carry the same id -- the
  /// driver reuses queue ids once a queue is gone -- and touching them again
  /// would reset a grace period that is already running.
  void deactivateQueue(uint32_t QueueId);

  //===--------------------------------------------------------------------===//

  int handleCreateQueue(int Fd, unsigned long Request, void *Arg);
  //=== KFD door: the queue table ==========================================//
  ForwardedQueue Queues[MaxTrackedQueues]{};
  int QueueCount{0};
  pthread_mutex_t QueueLock = PTHREAD_MUTEX_INITIALIZER;
  pthread_t PollerThread{};
  bool PollerStarted{false};
  bool PollerStopRequested{false};

  /// Completed passes of the polling thread over the whole table.
  ///
  /// This is the grace-period clock. Reusing an entry is safe once a full pass
  /// has finished that began after the queue was marked dead, because any pass
  /// still inside the entry must have started before that.
  uint64_t PollPass{0};

  /// Every queue we have ever substituted a ring for. Never decremented, so
  /// reclaiming an entry does not erase the evidence that the queue was
  /// wrapped.
  uint64_t WrappedQueueTotal{0};

  /// Queues left alone because the thread that created them was running the
  /// tool's own code. Counted rather than merely skipped so a test can tell
  /// "correctly excluded" from "never created".
  uint64_t ExcludedQueueTotal{0};

  /// Every ioctl that reached \c handleIoctl, KFD or not.
  uint64_t InterceptedIoctlTotal{0};

  /// Process-wide tool region, for work a runtime does on threads we never
  /// touch. See \c beginProcessWideToolRegion.
  std::atomic<unsigned> ProcessToolRegionDepth{0};

  //=== KFD door: the callback chain =======================================//
  /// Fixed array rather than a vector: this is read once per packet on the
  /// poller thread, and that thread has no business allocating.
  CallbackEntry Callbacks[MaxPacketCallbacks]{};

  /// How many entries of \c Callbacks are in use. Read without the lock by the
  /// poller; writers serialise on \c QueueLock.
  ///
  /// Only ever grows while callbacks are registered. Removal leaves a hole with
  /// a null \c CB, which the walk skips -- so the surviving callbacks keep both
  /// their relative order and their handles.
  unsigned CallbackCount{0};

  dev_t KfdRdev{0};
  bool KfdRdevCached{false};
  long PageSize{0};

  //=== HSA in the application's namespace =================================//
  void *AppHsaHandle{nullptr};
  bool HsaInitAttempted{false};
  pthread_mutex_t HsaInitLock = PTHREAD_MUTEX_INITIALIZER;

  /// Set when a \c CREATE_EVENT succeeded before we initialized HSA, which
  /// means the application owns the event page and ROCr cannot use interrupts.
  /// See
  /// \c applicationClaimedEventPage().
  bool ApplicationClaimedEventPage{false};
};

template <typename Derived> class PacketMonitorTrait : public PacketMonitor {
public:
  //===--------------------------------------------------------------------===//
  // Construction
  //===--------------------------------------------------------------------===//

  /// \brief Open both doors.
  ///
  /// Installs the \c hsa_queue_create wrapper through the rocprofiler API
  /// tables and registers the driver-boundary hooks with the audit engine.
  /// Neither is conditional on the other: which door a given application
  /// actually delivers packets through is a property of that application, not
  /// of how the tool was constructed, and an application can use both -- one
  /// that drives KFD itself and then has HSA brought up inside it does exactly
  /// that.
  ///
  /// \param Err receives an error when either installation is refused. An audit
  /// registration that fails is a real failure rather than a quiet no-op: a
  /// tool that attaches and then observes nothing looks exactly like an
  /// application that dispatched nothing, which is the failure mode this whole
  /// path is most prone to.
  ///
  /// \note The HSA wrappers are intentionally never uninstalled.
  PacketMonitorTrait(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &Loader,
      llvm::Error &Err)
      : CoreApiSnapshot(CoreApi), AmdExtSnapshot(AmdExt),
        LoaderApiSnapshot(Loader) {
    llvm::ErrorAsOutParameter EAO(Err);

    HsaApiTableInterceptor = std::make_unique<
        rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
        Err, std::make_tuple(&::CoreApiTable::hsa_queue_create_fn,
                             std::ref(UnderlyingHsaQueueCreateFn),
                             hsaQueueCreateWrapper));
    if (Err)
      return;

    Err = installAuditHooks();
  }

  /// \brief Stop polling and hand every substituted ring back.
  ///
  /// The HSA door needs no teardown -- its wrappers are never uninstalled --
  /// but the KFD door owns a thread that reads this object's fields, so it has
  /// to be stopped before they go away. That is new with per-instance state:
  /// while the queue table was file-scope the poller could simply run until the
  /// process exited.
  ~PacketMonitorTrait() { shutdown(); }

  PacketMonitorTrait(const PacketMonitorTrait &) = delete;
  PacketMonitorTrait &operator=(const PacketMonitorTrait &) = delete;

  //===--------------------------------------------------------------------===//
  // Telling our own queues from the application's
  //===--------------------------------------------------------------------===//

  /// \brief Mark the calling thread as running the tool's own code.
  ///
  /// \par The problem this solves
  /// A tool may create AQL queues for its own use, and those reach the driver
  /// through the same \c ioctl this trait interposes. Wrapping them means
  /// instrumenting ourselves: the tool's dispatches would be fed to the tool's
  /// own callback. ROCr guards the equivalent case in its own interception
  /// layer.
  ///
  /// \par When that actually happens, which is narrower than it used to look
  /// An earlier version of this comment said Luthier "links the HSA runtime and
  /// may call \c hsa_init even when the application never touches HSA". That
  /// premise is false for the case it was written about. An application driving
  /// KFD directly holds the DRM virtual address space for its GPUs, the kernel
  /// permits one such VM per GPU per process, and \c hsa_init inside that
  /// process therefore fails -- measured both orders: the application's \c
  /// ACQUIRE_VM then
  /// \c hsa_init gives \c HSA_STATUS_ERROR_OUT_OF_RESOURCES, and the reverse
  /// makes the application's \c ACQUIRE_VM fail with \c EBUSY.
  ///
  /// So the guard is live in two situations, and neither is the one originally
  /// described:
  /// \li this trait attached to an application that \e does use HSA, where the
  ///     tool and the application share the runtime. \c
  ///     kfd-oracle-self-exclusion tests exactly that;
  /// \li a tool that creates queues through the driver itself, without HSA.
  ///
  /// \par Why a thread-local, and what it rests on
  /// We see only an \c ioctl on a descriptor; nothing in that call says who
  /// asked for it. So the tool has to say so. A thread-local flag is enough
  /// because of a measured fact rather than an assumption: every queue the
  /// runtime created in response to \c hsa_queue_create appeared on the
  /// **calling thread** (Phase 0.2 -- one \c hsa_queue_create produced two AQL
  /// queues, and a device copy an SDMA queue, all on that thread).
  ///
  /// That measurement is the load-bearing part. If some future runtime creates
  /// a queue from a background thread on the tool's behalf, the flag will not
  /// cover it, and the oracle harness is what would catch that.
  ///
  /// Nested regions are counted, so a tool region inside another behaves.
  ///
  /// Only queue creation and allocation tracking consult this. Everything else
  /// the tool does through the driver is passed through regardless.
  static void beginToolRegion() { ToolRegionDepth++; }

  /// \brief End the region opened by \c beginToolRegion on this thread.
  static void endToolRegion() {
    if (ToolRegionDepth == 0) {
      // Unbalanced. Saying so is worth a line: silently clamping would hide a
      // missing beginToolRegion, and the failure that causes -- the tool's own
      // queues getting instrumented -- is invisible until something dispatches.
      fprintf(stderr, "%sendToolRegion() with no matching beginToolRegion()\n",
              LogPrefix);
      return;
    }
    ToolRegionDepth--;
  }

  /// \brief Scoped form of \c beginToolRegion / \c endToolRegion.
  ///
  /// Preferred over the bare calls: an early return between them would
  /// otherwise leave the thread permanently marked as the tool, and from then
  /// on none of the application's queues would be instrumented -- a silent,
  /// total loss of interception, which is the worst failure this path has.
  class ToolRegion {
  public:
    ToolRegion() { PacketMonitorTrait::beginToolRegion(); }
    ~ToolRegion() { PacketMonitorTrait::endToolRegion(); }
    ToolRegion(const ToolRegion &) = delete;
    ToolRegion &operator=(const ToolRegion &) = delete;
  };

protected:
  //===--------------------------------------------------------------------===//
  // PacketMonitor's three points of contact
  //===--------------------------------------------------------------------===//

  [[nodiscard]] bool insideThreadToolRegion() const override {
    return ToolRegionDepth != 0;
  }

  void deliverDispatchPacket(const QueueInfo &Q, uint64_t PacketIndex,
                             hsa::AqlPacket &Packet) override {
    (void)Singleton<Derived>::withInstance(
        [&](Derived &Self) { Self.onDispatchPacket(Q, PacketIndex, Packet); });
  }

private:
  //===--------------------------------------------------------------------===//
  // LD_AUDIT registration
  //===--------------------------------------------------------------------===//

  /// The name the audit engine knows us by. One per symbol, because
  /// \c ah_set_target keys on the tool name alone and stops at the first chain
  /// carrying it -- a shared name would make four of the five hooks
  /// unaddressable.
  static constexpr const char *IoctlToolName = "luthier-packet-monitor-ioctl";
  static constexpr const char *OpenToolName = "luthier-packet-monitor-open";
  static constexpr const char *Open64ToolName = "luthier-packet-monitor-open64";
  static constexpr const char *OpenatToolName = "luthier-packet-monitor-openat";
  static constexpr const char *Openat64ToolName =
      "luthier-packet-monitor-openat64";

  /// \brief The real \c ioctl, as the three-argument form the KFD path always
  /// uses.
  ///
  /// \par Assumption A1, and why it holds
  /// The C library's \c ioctl is variadic. Every KFD request passes exactly one
  /// pointer, so declaring the three-argument form is a deliberate
  /// simplification
  /// -- but note where it applies: the hook covers \c ioctl for the \e whole
  /// process, so a genuine two-argument call (a terminal \c ioctl, say) also
  /// arrives here, and its third parameter is whatever happened to be in the
  /// register. It is never dereferenced: \c handleIoctl establishes that the
  /// descriptor is \c /dev/kfd before reading \c Arg at all.
  ///
  /// So this is safe **by the x86-64 SysV calling convention** -- a variadic
  /// callee receives its third argument in \c rdx regardless of how the caller
  /// declared it -- and not by anything the code does. Worth knowing before
  /// porting this anywhere with a different convention, or before adding a
  /// filter that reads \c Arg earlier than the descriptor check.
  using RealIoctlFn = int (*)(int, unsigned long, void *);

  /// \c open and friends are variadic, and the mode argument is only present
  /// when
  /// \c O_CREAT is set. Declaring the three-argument form is safe for the same
  /// reason it is safe for \c ioctl, and the mode is only ever read when the
  /// flags say it is there.
  using RealOpenFn = int (*)(const char *, int, mode_t);
  using RealOpenatFn = int (*)(int, const char *, int, mode_t);

  /// Written by \c ld.so at bind time, not by \c dlsym.
  ///
  /// Static rather than an instance field, and this is the mechanism forcing
  /// it: the address of each is handed to the audit engine at registration, and
  /// audit_hook has no unregister call, so the storage has to outlive any one
  /// instance. Being members of \c PacketMonitorTrait<Derived> they are still
  /// one set per tool rather than one per process.
  static inline RealIoctlFn RealIoctl{};
  static inline RealOpenFn RealOpen{};
  static inline RealOpenFn RealOpen64{};
  static inline RealOpenatFn RealOpenat{};
  static inline RealOpenatFn RealOpenat64{};

  /// \brief Register the five symbols this trait needs with the audit engine.
  ///
  /// \c register_wrap rather than \c register_replace: other tools may be
  /// wrapping the same symbols, and wrapping chains where replacing discards.
  /// The engine writes the next link inward into \c Real*, which is why nothing
  /// here has to search for it.
  llvm::Error installAuditHooks() {
    struct Registration {
      const char *ToolName;
      const char *Symbol;
      int Result;
    };

    const Registration Registrations[] = {
        {IoctlToolName, "ioctl",
         audit_hooks::register_wrap<&PacketMonitorTrait::ioctlHook,
                                    &PacketMonitorTrait::RealIoctl>(
             IoctlToolName, "ioctl")},
        {OpenToolName, "open",
         audit_hooks::register_wrap<&PacketMonitorTrait::openWrapper,
                                    &PacketMonitorTrait::RealOpen>(OpenToolName,
                                                                   "open")},
        {Open64ToolName, "open64",
         audit_hooks::register_wrap<&PacketMonitorTrait::open64Hook,
                                    &PacketMonitorTrait::RealOpen64>(
             Open64ToolName, "open64")},
        {OpenatToolName, "openat",
         audit_hooks::register_wrap<&PacketMonitorTrait::openatHook,
                                    &PacketMonitorTrait::RealOpenat>(
             OpenatToolName, "openat")},
        {Openat64ToolName, "openat64",
         audit_hooks::register_wrap<&PacketMonitorTrait::openat64Hook,
                                    &PacketMonitorTrait::RealOpenat64>(
             Openat64ToolName, "openat64")},
    };

    for (const Registration &R : Registrations) {
      if (R.Result != 0)
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "ah_register_hook(\"{0}\", \"{1}\") returned {2}, so this tool "
            "would attach and then observe nothing. Check that the application "
            "was started under LD_AUDIT=libaudit_core.so with this tool named "
            "in AH_PLUGINS.",
            R.ToolName, R.Symbol, R.Result));
    }
    return llvm::Error::success();
  }

  //===--------------------------------------------------------------------===//
  // The hooks themselves
  //
  // Reached through the singleton rather than through a captured pointer, for
  // the two reasons the HSA path does the same. Registration happens inside
  // this trait's constructor, so a call arriving before the tool finishes
  // constructing would otherwise reach a half-built Derived; and withInstance
  // holds the instance alive for the call, so a concurrent teardown cannot
  // destroy the tool underneath a hook that is already running. When there is
  // no instance the call is forwarded untouched, which is what makes attaching
  // late safe.
  //===--------------------------------------------------------------------===//

  static int ioctlHook(int Fd, unsigned long Request, void *Arg) {
    int Ret = 0;
    const bool Handled = Singleton<Derived>::withInstance([&](Derived &Self) {
      Ret =
          static_cast<PacketMonitorTrait &>(Self).handleIoctl(Fd, Request, Arg);
    });
    if (!Handled)
      return RealIoctl(Fd, Request, Arg);
    return Ret;
  }

  /// Shared by all four open wrappers: hand back the bound descriptor if there
  /// is one, and take the application's first touch of the driver as the cue to
  /// bring HSA up in its namespace.
  ///
  /// All four variants are hooked, because they are not aliases at the symbol
  /// level and missing one is invisible: interposing only \c open and \c openat
  /// missed tinygrad entirely, since CPython's \c os.open resolves to \c
  /// open64, and the symptom was the address-space collision looking exactly as
  /// though none of this were installed.
  ///
  /// \return a descriptor, or -1 to mean "let the real call through", which is
  /// also what makes the first opener the party whose descriptor gets bound.
  static int borrowedOrMinusOne(const char *Path) {
    if (Path == nullptr)
      return -1;

    llvm::StringRef PathStrRef{Path};
    const bool IsKfdNode = isKfdDevicePath(Path);
    const bool IsRenderNode = kfd::isRenderNodePath(Path);
    if (!IsKfdNode && !IsRenderNode)
      return -1;

    (void)Singleton<Derived>::withInstance([&](Derived &Self) {
      auto &Monitor = static_cast<PacketMonitorTrait &>(Self);
      // The application is about to talk to the driver. This is the earliest
      // point at which bringing HSA up in its namespace is both safe and
      // possible; see ensureHsaInitializedInApplication.
      //
      // Reported and then dropped rather than propagated: we are standing in
      // for the application's open(), and there is no way to return a Luthier
      // error through it. Failing the open instead would break an application
      // that never asked for any of this, over a runtime only the tool needs.
      // The message is what makes the subsequent "loader could not find an
      // agent" diagnosable.
      if (llvm::Error Err = Monitor.ensureHsaInitializedInApplication()) {
        fprintf(stderr, "%s%s\n", LogPrefix,
                llvm::toString(std::move(Err)).c_str());
      }
    });

    if (!IsRenderNode || !kfd::isFdSharingEnabled())
      return -1;
    return kfd::borrowBoundRenderNodeFd(Path);
  }

  static int openWrapper(const char *Path, int Flags, mode_t Mode) {
    if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
      return Borrowed;
    return RealOpen(Path, Flags, Mode);
  }

  static int open64Hook(const char *Path, int Flags, mode_t Mode) {
    if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
      return Borrowed;
    return RealOpen64(Path, Flags, Mode);
  }

  static int openatHook(int DirFd, const char *Path, int Flags, mode_t Mode) {
    if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
      return Borrowed;
    return RealOpenat(DirFd, Path, Flags, Mode);
  }

  static int openat64Hook(int DirFd, const char *Path, int Flags, mode_t Mode) {
    if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
      return Borrowed;
    return RealOpenat64(DirFd, Path, Flags, Mode);
  }

  static bool isKfdDevicePath(const char *Path) {
    return Path != nullptr && std::strcmp(Path, "/dev/kfd") == 0;
  }

  //===--------------------------------------------------------------------===//
  // HSA door: queue creation and packet delivery
  //===--------------------------------------------------------------------===//

  inline static decltype(hsa_queue_create) *UnderlyingHsaQueueCreateFn{};

  static hsa_status_t
  hsaQueueCreateWrapper(hsa_agent_t Agent, uint32_t Size,
                        hsa_queue_type32_t Type,
                        void (*Callback)(hsa_status_t, hsa_queue_t *, void *),
                        void *Data, uint32_t PrivateSegmentSize,
                        uint32_t GroupSegmentSize, hsa_queue_t **Queue) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        UnderlyingHsaQueueCreateFn != nullptr,
        "The underlying hsa_queue_create function for "
        "PacketMonitorTrait is nullptr"));
    hsa_status_t Out =
        UnderlyingHsaQueueCreateFn(Agent, Size, Type, Callback, Data,
                                   PrivateSegmentSize, GroupSegmentSize, Queue);
    if (Out != HSA_STATUS_SUCCESS)
      return Out;

    (void)Singleton<Derived>::withInstance([&](Derived &Self) {
      auto &Trait = static_cast<PacketMonitorTrait &>(Self);

      /// Try to install an event handler on the newly-created queue.
      const hsa_status_t EventHandlerStatus =
          Trait.AmdExtSnapshot.getTable()
              .template callFunction<hsa_amd_queue_intercept_register>(
                  *Queue, interceptQueuePacketHandler, *Queue);
      /// If we failed to install an event handler, the queue was a normal
      /// queue; destroy it and recreate an intercept queue in its place.
      if (EventHandlerStatus == HSA_STATUS_ERROR_INVALID_QUEUE) {
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            Trait.CoreApiSnapshot.getTable()
                .template callFunction<hsa_queue_destroy>(*Queue),
            "Failed to destroy the application's queue"));
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            Trait.AmdExtSnapshot.getTable()
                .template callFunction<hsa_amd_queue_intercept_create>(
                    Agent, Size, Type, Callback, Data, PrivateSegmentSize,
                    GroupSegmentSize, Queue),
            "Failed to create an intercept queue"));
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            Trait.AmdExtSnapshot.getTable()
                .template callFunction<hsa_amd_queue_intercept_register>(
                    *Queue, interceptQueuePacketHandler, *Queue),
            "Failed to assign a packet handler to the intercept queue"));
      } else {
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            EventHandlerStatus,
            "Failed to install HSA queue intercept handler"));
      }
    });
    return Out;
  }

  static void
  interceptQueuePacketHandler(const void *Packets, uint64_t PacketCount,
                              uint64_t UserPacketIdx, void *Data,
                              hsa_amd_queue_intercept_packet_writer Writer) {
    bool Handled = Singleton<Derived>::withInstance([&](Derived &Self) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          Data != nullptr,
          "Failed to get the queue used to dispatch packets."));
      auto &Queue = *static_cast<hsa_queue_t *>(Data);

      Self.onPackets(
          Queue, UserPacketIdx,
          llvm::ArrayRef(static_cast<const hsa::AqlPacket *>(Packets),
                         PacketCount),
          Writer);
    });
    if (!Handled)
      Writer(Packets, PacketCount);
  }

  //===--------------------------------------------------------------------===//
  // Reaching the real ioctl
  //===--------------------------------------------------------------------===//

  /// Forward to the next call in the chain.
  ///
  /// Aborting beats limping on when the engine never filled this in. A hook
  /// that intercepts an ioctl but cannot forward it leaves the application
  /// waiting for work the driver never received -- a hang with no error
  /// anywhere, which is the worst outcome available here.
  int realIoctl(int Fd, unsigned long Request, void *Arg) override {
    if (RealIoctl == nullptr) {
      fprintf(
          stderr,
          "%sthe audit engine never bound the real ioctl, so an intercepted "
          "call cannot be forwarded. The tool was constructed without being "
          "loaded as an audit_hook plugin; start the application with "
          "LD_AUDIT=libaudit_core.so and this tool in AH_PLUGINS.\n",
          LogPrefix);
      abort();
    }
    return RealIoctl(Fd, Request, Arg);
  }
  // Tool regions
  //===--------------------------------------------------------------------===//

  /// Depth of nested tool regions on this thread. A depth rather than a flag so
  /// that a tool region opened inside another does not end interception when
  /// the inner one closes.
  ///
  /// \c thread_local, so it cannot be a non-static member; being a member of
  /// \c PacketMonitorTrait<Derived> keeps it one per tool rather than one per
  /// process.
  static inline thread_local unsigned ToolRegionDepth{0};

  //===--------------------------------------------------------------------===//
  // State. Instance fields, every one that can be.
  //===--------------------------------------------------------------------===//

  //=== HSA door ==========================================================//
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApiSnapshot;
  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtSnapshot;
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &LoaderApiSnapshot;
  std::unique_ptr<rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>
      HsaApiTableInterceptor;
};

} // namespace luthier

#endif // LUTHIER_TOOLING_PACKET_MONITOR_TRAIT_H
