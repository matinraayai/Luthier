//===-- QueueWrapper.h - KFD-level AQL queue interception -------*- C++ -*-===//
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
/// Intercepts AQL packets at the KFD (driver) boundary, below HSA, so packets
/// can be observed and rewritten for applications that never use the HSA
/// runtime. This is the driver-level counterpart to
/// \c luthier::PacketMonitorTrait, which only works for HSA applications.
///
/// \par How it works
/// Sending a packet to a GPU queue is a plain memory write -- there is no
/// function call or system call to intercept. Creating a queue *is* a system
/// call, so that is the one place we can insert ourselves. At queue creation we
/// swap the ring buffer the GPU reads for one we own; the application keeps
/// writing its own. A background thread copies each finished packet across,
/// runs the callback in between, and writes the copy's header last. The GPU
/// ignores a slot whose header reads INVALID, so the callback is guaranteed to
/// run before the GPU can act on the packet.
///
/// \par Scope
/// Only \c COMPUTE_AQL queues are wrapped; PM4 and SDMA queues pass through
/// untouched, as are queues created inside a \c ToolRegion. Several callbacks
/// may be registered and each may edit a packet in place, but none may add or
/// remove packets. ROCr expresses both by handing packets to a writer and simply
/// not calling it; supporting that here would mean emitting a different number of
/// packets than the application submitted, which invalidates the index-sequence
/// check that is currently the strongest correctness signal the test suite has.
///
/// \par Environment variables
/// \li \c LUTHIER_VERBOSE=1 logs every forwarded packet. Off by default: on a
///     busy queue this is one line per packet. A per-queue summary is printed at
///     teardown either way.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_QUEUE_WRAPPER_H
#define LUTHIER_KFD_QUEUE_WRAPPER_H

#include "luthier/HSA/AqlPacket.h"
#include <cstdint>

namespace luthier::kfd {

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

/// \brief Invoked once for every packet the application commits, before the GPU
/// can act on it.
///
/// \param Q the queue the packet was submitted to
/// \param PacketIndex the packet's position in the stream. Counts from zero and
/// only increases; it is not a slot number
/// \param Packet **our copy** of the packet. Edits made here are what the GPU
/// executes; the application's own copy is left alone
/// \param UserData the pointer supplied when this callback was registered
///
/// \note Runs on the wrapper's polling thread, not the application's, so it
/// must be thread-safe with respect to the tool's own state.
using PacketCallback = void (*)(const QueueInfo &Q, uint64_t PacketIndex,
                                hsa::AqlPacket &Packet, void *UserData);

/// \brief Replace the whole chain with a single callback.
///
/// Passing \c nullptr removes every callback, after which packets are copied
/// through unchanged.
///
/// \param CB the callback, or \c nullptr
/// \param UserData opaque pointer handed back to \p CB on each call
void setPacketCallback(PacketCallback CB, void *UserData);

//===----------------------------------------------------------------------===//
// Several tools on one packet
//===----------------------------------------------------------------------===//

/// \brief Identifies a registered callback, for removing it again.
using CallbackHandle = int;

/// Returned by \c addPacketCallback when there is no room left.
static constexpr CallbackHandle InvalidCallbackHandle = -1;

/// \brief Add a callback without disturbing the ones already registered.
///
/// More than one tool may want a turn at each packet -- Luthier alongside a
/// profiler, say -- and the HSA runtime supports exactly that, so a driver-level
/// replacement that did not would be a step backwards.
///
/// \par Order
/// **Last registered runs first**, and each callback sees the edits made by the
/// ones that ran before it. This is ROCr's order, deliberately: it walks its
/// interceptor list from the end (\c intercept_queue.cpp:375), so the most
/// recently attached tool sees the packet as the application wrote it, and the
/// earliest-attached tool sees it last, just before the GPU does. Choosing the
/// opposite order would make a tool behave differently depending on which
/// interception layer it was attached to.
///
/// \par What this does not do
/// A callback here cannot drop a packet or emit extra ones. ROCr expresses both
/// by having a callback pass packets to a writer, and not calling it drops them.
/// That is deferred: emitting a different number of packets than the application
/// submitted invalidates the index-sequence check, which is currently the
/// strongest correctness signal the test suite has, so it needs its own
/// verification story first.
///
/// \return a handle, or \c InvalidCallbackHandle if the chain is full
CallbackHandle addPacketCallback(PacketCallback CB, void *UserData);

/// \brief Remove a callback added by \c addPacketCallback.
///
/// Leaves the order of the remaining callbacks unchanged.
void removePacketCallback(CallbackHandle H);

/// \brief Most callbacks that can be registered at once.
static constexpr unsigned MaxPacketCallbacks = 8;

/// \brief Run every registered callback over one packet, last registered first.
///
/// This is what the copier calls for each packet. It is public so the
/// registration bookkeeping -- handle reuse, the published count, what
/// \c setPacketCallback does to an existing chain -- can be exercised without a
/// GPU. That bookkeeping is where the mistakes live, and testing only the walk
/// over a hand-built array would leave all of it uncovered.
void runRegisteredCallbacks(const QueueInfo &Q, uint64_t PacketIndex,
                            hsa::AqlPacket &Packet);

namespace detail {

/// \brief One registered callback and its user pointer.
struct CallbackEntry {
  PacketCallback CB;
  void *UserData;
};

/// \brief Run a chain over one packet, last registered first.
///
/// A free function over a plain array, rather than a method, so the ordering
/// guarantee above can be tested without a GPU, a queue or a driver -- which is
/// where a claim like "last registered runs first" belongs, since it is
/// otherwise the kind of thing that is asserted in a comment and never checked.
///
/// Entries whose \c CB is null are skipped, which is how removal works without
/// shuffling the others.
void runCallbackChain(const CallbackEntry *Entries, unsigned Count,
                      const QueueInfo &Q, uint64_t PacketIndex,
                      hsa::AqlPacket &Packet);

} // namespace detail

/// \brief Signature of \c luthierKfdSetPacketCallback, for callers that look the
/// symbol up at run time.
using SetPacketCallbackFn = void (*)(PacketCallback, void *);

//===----------------------------------------------------------------------===//
// Telling our own queues from the application's
//===----------------------------------------------------------------------===//

/// \brief Mark the calling thread as running the tool's own code.
///
/// \par The problem this solves
/// A tool may create AQL queues for its own use, and those reach the driver
/// through the same \c ioctl this wrapper interposes. Wrapping them means
/// instrumenting ourselves: the tool's dispatches would be fed to the tool's own
/// callback. ROCr guards the equivalent case in its own interception layer.
///
/// \par When that actually happens, which is narrower than it used to look
/// An earlier version of this comment said Luthier "links the HSA runtime and may
/// call \c hsa_init even when the application never touches HSA". That premise is
/// false for the case it was written about. An application driving KFD directly
/// holds the DRM virtual address space for its GPUs, the kernel permits one such
/// VM per GPU per process, and \c hsa_init inside that process therefore fails --
/// measured both orders: the application's \c ACQUIRE_VM then \c hsa_init gives
/// \c HSA_STATUS_ERROR_OUT_OF_RESOURCES, and the reverse makes the application's
/// \c ACQUIRE_VM fail with \c EBUSY.
///
/// So the guard is live in two situations, and neither is the one originally
/// described:
/// \li this wrapper preloaded into an application that \e does use HSA, where the
///     tool and the application share the runtime. \c kfd-oracle-self-exclusion
///     tests exactly that;
/// \li a tool that creates queues through the driver itself, without HSA.
///
/// It would become live in the original sense only if the wrapper were changed to
/// hand the application HSA's own DRM descriptor, which is one candidate strategy
/// for loading instrumented kernels back. Under that arrangement HSA \e is
/// initialized inside the application, and this guard stops being latent.
///
/// \par Why a thread-local, and what it rests on
/// The wrapper sees only an \c ioctl on a descriptor; nothing in that call says
/// who asked for it. So the tool has to say so. A thread-local flag is enough
/// because of a measured fact rather than an assumption: every queue the runtime
/// created in response to \c hsa_queue_create appeared on the **calling thread**
/// (Phase 0.2 -- one \c hsa_queue_create produced two AQL queues, and a device
/// copy an SDMA queue, all on that thread).
///
/// That measurement is the load-bearing part. If some future runtime creates a
/// queue from a background thread on the tool's behalf, the flag will not cover
/// it, and the oracle harness is what would catch that.
///
/// Nested regions are counted, so a tool region inside another behaves.
///
/// Only queue creation consults this. Everything else the tool does through the
/// driver is passed through regardless, as it already was.
void beginToolRegion();

/// \brief End the region opened by \c beginToolRegion on this thread.
void endToolRegion();

/// \brief Mark the whole process as running the tool's own code.
///
/// \c beginToolRegion covers the calling thread, which is enough for a queue the
/// runtime creates in response to \c hsa_queue_create -- measured, those appear
/// on the calling thread. It is \b not enough for \c hsa_init: bringing the
/// runtime up spawns its own threads and creates queues on them, and measured in
/// a KFD application the runtime's queue was wrapped as the process's second
/// queue, after which the tool would instrument its own dispatches.
///
/// Use for that window only. While it is open a queue the \e application creates
/// is excluded too, which is wrong in general and acceptable for the length of
/// one runtime initialization.
void beginProcessWideToolRegion();

/// \brief End the region opened by \c beginProcessWideToolRegion.
void endProcessWideToolRegion();

/// \brief Scoped form of \c beginProcessWideToolRegion / \c endProcessWideToolRegion.
class ProcessWideToolRegion {
public:
  ProcessWideToolRegion() { beginProcessWideToolRegion(); }
  ~ProcessWideToolRegion() { endProcessWideToolRegion(); }
  ProcessWideToolRegion(const ProcessWideToolRegion &) = delete;
  ProcessWideToolRegion &operator=(const ProcessWideToolRegion &) = delete;
};

/// \brief Scoped form of \c beginToolRegion / \c endToolRegion.
///
/// Preferred over the bare calls: an early return between them would otherwise
/// leave the thread permanently marked as the tool, and from then on none of the
/// application's queues would be instrumented -- a silent, total loss of
/// interception, which is the worst failure this module has.
class ToolRegion {
public:
  ToolRegion() { beginToolRegion(); }
  ~ToolRegion() { endToolRegion(); }
  ToolRegion(const ToolRegion &) = delete;
  ToolRegion &operator=(const ToolRegion &) = delete;
};

/// \brief How many AQL queues were left alone because a tool region was open.
///
/// The counterpart to \c wrappedQueueCount, and the reason both exist: a test
/// that only counts wrapped queues cannot tell "correctly excluded" from "never
/// created". Cumulative.
uint64_t excludedQueueCount();

/// \brief How many AQL queues the wrapper has substituted a ring for, ever.
///
/// Cumulative, not live, so a destroyed queue still counts.
///
/// Exists because whether a queue was wrapped is otherwise only observable when
/// packets flow through it -- and the case that matters most is one where they
/// may not. When a tool initialises HSA, the runtime creates AQL queues on the
/// tool's behalf; the wrapper must leave those alone, but they can sit empty, so
/// no callback would ever fire either way. A count separates "correctly ignored"
/// from "wrapped, but idle".
uint64_t wrappedQueueCount();

/// \brief Entry point for the \c ioctl interposer.
///
/// Handles queue creation and destruction and forwards everything else to the
/// real \c ioctl. Separated from the interposer itself so the logic can be
/// linked into tests without also interposing on the process.
///
/// \return whatever the underlying \c ioctl returned
int handleIoctl(int Fd, unsigned long Request, void *Arg);

} // namespace luthier::kfd

extern "C" {

/// \brief C-linkage form of \c luthier::kfd::setPacketCallback.
///
/// Exists so a program can discover the wrapper at run time with \c dlsym rather
/// than linking against it. That matters for the test suite: the same binary has
/// to run both with the wrapper preloaded and without it, and a link-time
/// dependency would force the wrapper to load either way -- which would make the
/// unwrapped baseline impossible to measure.
///
/// Absent from the process when the wrapper is not loaded, which is exactly how
/// a caller detects that it is running unwrapped.
void luthierKfdSetPacketCallback(luthier::kfd::PacketCallback CB,
                                 void *UserData);

/// \brief C-linkage form of \c luthier::kfd::wrappedQueueCount.
unsigned long long luthierKfdWrappedQueueCount();

/// \brief C-linkage form of \c luthier::kfd::excludedQueueCount.
unsigned long long luthierKfdExcludedQueueCount();

/// \brief C-linkage form of \c luthier::kfd::addPacketCallback.
int luthierKfdAddPacketCallback(luthier::kfd::PacketCallback CB,
                                void *UserData);

/// \brief C-linkage form of \c luthier::kfd::removePacketCallback.
void luthierKfdRemovePacketCallback(int Handle);

/// \brief C-linkage form of \c luthier::kfd::beginToolRegion.
void luthierKfdBeginToolRegion();

/// \brief C-linkage form of \c luthier::kfd::beginProcessWideToolRegion.
void luthierKfdBeginProcessWideToolRegion();

/// \brief C-linkage form of \c luthier::kfd::endProcessWideToolRegion.
void luthierKfdEndProcessWideToolRegion();

/// \brief C-linkage form of \c luthier::kfd::endToolRegion.
void luthierKfdEndToolRegion();
}

#endif // LUTHIER_KFD_QUEUE_WRAPPER_H
