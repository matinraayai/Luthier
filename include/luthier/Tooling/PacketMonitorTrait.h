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
/// untouched, as are queues created inside a tool region (see
/// \c PinHSATrait). \p Derived's \c onDispatchPacket may edit a packet in
/// place, but may not add or remove packets. ROCr expresses both by handing packets to a writer and simply not
/// calling it; supporting that here would mean emitting a different number of
/// packets than the application submitted, which invalidates the
/// index-sequence check that is currently the strongest correctness signal the
/// test suite has.
///
/// \par Interception is LD_AUDIT, via audit_hook
/// \c ioctl is wrapped with \c audit_hooks::register_wrap from the trait's
/// constructor. The tool \e is the audit_hook plugin: it is loaded into the
/// auditor's linker namespace out of \c AH_PLUGINS, so by the time this trait
/// is constructed the \c audit_hooks API is available and the registration is
/// in time for the application's first call.
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
/// \c QueueWrapper.cpp -- one queue table, one poller thread, one set of
/// counters per process -- which meant two tools in one process silently
/// shared a tracking table and a 64-queue budget. The only
/// static left is the one the mechanism forces -- \c RealIoctl, whose address
/// is handed to \c ld.so at registration and must therefore outlive any one
/// instance, since audit_hook has no unregister call -- and it is a member of
/// \c PacketMonitorTrait<Derived> rather than of \c PacketMonitor, so even
/// that one is per-tool rather than per-process. That is the other reason it is
/// in the template rather than in the shared base.
///
/// \par Two classes, and where each lives
/// \li \c PacketMonitor -- non-templated, and everything that does not depend
///     on \p Derived: the queue table, the poller thread, the ring
///     substitution, the \c ioctl path. Declared here, compiled once in
///     \c PacketMonitorTrait.cpp;
/// \li \c PacketMonitorTrait<Derived> -- inherits it, and adds only what cannot
///     be written without \p Derived: the \c ioctl audit hook and the
///     \c static function pointer \c ld.so writes, the HSA API-table wrapper,
///     and the calls into the tool itself.
///
/// The cut is the four pure virtuals in \c PacketMonitor's protected section.
/// Keeping the driver half out of the template also keeps
/// \c linux/kfd_ioctl.h out of the public header of every Luthier tool.
///
/// \par What this trait no longer does
/// Pinning the HSA runtime into the application's linker namespace and marking
/// the windows in which the tool -- rather than the application -- is the party
/// talking to the driver both moved to \c PinHSATrait. They arrived here
/// because they share this trait's mechanism, the audit hooks, and not because
/// they are part of observing packets: a tool can want either job without the
/// other. What is left of them here is the two questions this trait has to ask
/// about them at the driver boundary, \c insideToolRegion and
/// \c noteApplicationCreatedEvent, which are pure virtual for the same reason
/// the other two are.
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
#include "luthier/Tooling/PinHSATrait.h"
#include "luthier/Rocprofiler/HsaApiTableSnapshot.h"
#include "luthier/Rocprofiler/HsaApiTableWrapperInstaller.h"

#include "luthier/Audit/audit_hook.hpp"

#include <hsa/hsa_api_trace.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <memory>

namespace luthier {

/// \brief Everything about monitoring packets that does not depend on which
/// tool is doing the monitoring.
///
/// \par Why this is a separate, non-templated class
/// The queue table, the poller thread, the ring substitution and the whole
/// \c ioctl path are the same code whatever \p Derived is, and compiling them
/// into every tool that instantiates
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
  /// \brief Identifies a wrapped queue to \c deliverDispatchPacket.
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

  virtual ~PacketMonitor();

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

protected:
  //===--------------------------------------------------------------------===//
  // The four points of contact with the templated half
  //
  // Each is here because of something that cannot be expressed without
  // \c Derived. Kept as narrow as possible: everything else about packet
  // monitoring is in this class.
  //
  // Two of them -- \c insideToolRegion and \c noteApplicationCreatedEvent --
  // are questions for \c PinHSATrait rather than for the tool. They are
  // virtual for the same reason the other two are: the answers live in
  // \c static members of a class template, which this non-templated half
  // cannot name.
  //===--------------------------------------------------------------------===//

  /// \brief Forward to the next \c ioctl in the audit chain.
  ///
  /// The next link is a \c static member of the templated half, because
  /// \c ld.so is handed its address at registration and audit_hook has no
  /// unregister call, so the storage has to outlive any one instance.
  virtual int realIoctl(int Fd, unsigned long Request, void *Arg) = 0;

  /// \brief Hand one packet to \p Derived's \c onDispatchPacket.
  ///
  /// The one callback the KFD door has. It takes no user pointer and there is
  /// no way to register a second: the implementation is fixed at compile time
  /// by \p Derived, the way the HSA door's \c interceptQueuePacketHandler is.
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
  // Copying one packet
  //===--------------------------------------------------------------------===//

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

  dev_t KfdRdev{0};
  bool KfdRdevCached{false};
  long PageSize{0};
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
  /// \note The KFD door needs the HSA runtime pinned into the application and
  /// needs to know when the tool is the party talking to the driver. Neither is
  /// installed here any more; both are \c PinHSATrait's, which a tool composing
  /// this trait is expected to compose as well.
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

protected:
  //===--------------------------------------------------------------------===//
  // PacketMonitor's four points of contact
  //===--------------------------------------------------------------------===//

  /// The KFD door's one callback, and the counterpart of
  /// \c interceptQueuePacketHandler on the HSA door. Reached through
  /// \c Singleton<Derived> rather than a captured \c this or a user pointer,
  /// for the same two reasons: a packet arriving before the tool has finished
  /// constructing finds no instance and is passed through untouched, and
  /// \c withInstance holds a counted reference for the length of the call, so
  /// a concurrent teardown cannot destroy the tool underneath it.
  ///
  /// That second property is what the callback chain this replaced was
  /// re-implementing with a grace period of its own, less well -- which is why
  /// removing the chain also removed a teardown wait.
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
  /// carrying it -- a shared name would make every hook but the first
  /// unaddressable.
  static constexpr const char *IoctlToolName = "luthier-packet-monitor-ioctl";

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

  /// Written by \c ld.so at bind time, not by \c dlsym.
  ///
  /// Static rather than an instance field, and this is the mechanism forcing
  /// it: its address is handed to the audit engine at registration, and
  /// audit_hook has no unregister call, so the storage has to outlive any one
  /// instance. Being a member of \c PacketMonitorTrait<Derived> it is still one
  /// per tool rather than one per process.
  static inline RealIoctlFn RealIoctl{};

  /// \brief Register the one symbol this trait needs with the audit engine.
  ///
  /// \c register_wrap rather than \c register_replace: other tools may be
  /// wrapping the same symbol -- \c PinHSATrait wraps the \c open family
  /// alongside this -- and wrapping chains where replacing discards. The engine
  /// writes the next link inward into \c RealIoctl, which is why nothing here
  /// has to search for it.
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
