//===-- InstrumentedKernelLoaderAndLauncher.h -------------------*- C++ -*-===//
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
/// Defines the \c InstrumentedKernelLoaderAndLauncher, in charge of loading
/// and unloading instrumented copies of kernels, and the
/// \c InstrumentedKernelLoaderAndLauncherTrait, which provides its
/// \c Derived with the loader functionality, as well as HSA callbacks to
/// invalidate the instrumented kernel cache.
///
/// \anchor extended_kernarg_abi
/// # Extended kernarg buffer ABI
///
/// The instrumented kernel emitted by \c TargetModulePatcherPass reads its
/// implicit args out of an "extended kernarg buffer" the launcher stands up
/// in front of every instrumented dispatch. This is the launcher-side
/// statement of that ABI; the emitter side lives in
/// \c TargetModulePatcherPass::emitKernargBufferExpansion.
///
/// Two shapes, decided per kernel by
/// \c LoadedKernelInfo::HasAppKernargPrefix, which the launcher reads off
/// the code object's \c amdhsa.kernels metadata at load time:
///
/// \li \c HasAppKernargPrefix == \c true — the instrumented kernel expects the
///     original app kernarg address at bytes \c [0,8) and its COV5 hidden-arg
///     block right after it: <tt>[ app_kernarg_ptr : 64 ][ hidden args ... ]</tt>.
///     The app's own explicit args are still read through
///     \c app_kernarg_ptr, so no bytes of them appear in the extended buffer
///     itself.
///
/// \li \c HasAppKernargPrefix == \c false — the instrumented kernel takes no
///     explicit kernarg. The extended buffer is just the COV5 hidden block
///     starting at offset 0: <tt>[ hidden args ... ]</tt>.
///
/// The total buffer size is the instrumented kernel's
/// \c kernel_descriptor_t::kernarg_size. Each hidden slot is placed at the
/// \c .offset and \c .size the code object's \c amdhsa.kernels metadata
/// declares for it, using the values \c writeHiddenKernelArguments derives
/// from the dispatch packet + queue.
///
/// \c overrideWithInstrumented builds the buffer, points the dispatch
/// packet's \c kernarg_address at it, and hands back an
/// \c ExtendedKernargBuffer handle. The caller MUST keep that handle alive
/// until the dispatch completes; releasing early frees memory the device may
/// still be reading. The typical pattern is to move the handle into the
/// completion callback keyed on \c hsa_kernel_dispatch_packet_t::completion_signal
/// and drop it there.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_INSTRUMENTED_KERNEL_LOADER_AND_LAUNCHER_H
#define LUTHIER_HSA_TOOLING_INSTRUMENTED_KERNEL_LOADER_AND_LAUNCHER_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSATooling/DevicePrintf.h"
#include "luthier/HSATooling/HiddenArgBuffers.h"
#include "luthier/HSATooling/HostcallHandler.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"
#include "luthier/ToolCodeGen/Metadata.h"
#include <cstdint>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/RWMutex.h>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

namespace llvm::object {
class ObjectFile;
} // namespace llvm::object

namespace luthier {

namespace object {
class AMDGCNObjectFile;
} // namespace object

class ExtendedKernargBuffer;

/// \brief Loader for loading and caching instrumented copies of kernels
class InstrumentedKernelLoaderAndLauncher {
public:
  InstrumentedKernelLoaderAndLauncher(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &Loader);

  ~InstrumentedKernelLoaderAndLauncher();

  InstrumentedKernelLoaderAndLauncher(
      const InstrumentedKernelLoaderAndLauncher &) = delete;
  InstrumentedKernelLoaderAndLauncher &
  operator=(const InstrumentedKernelLoaderAndLauncher &) = delete;

  /// Parse \p Relocatable as an AMDGCN ELF, require it contain exactly
  /// one kernel function excluding global constructor/destructor kernels
  /// (any name — assumed to be the instrumented kernel), create + load +
  /// freeze a fresh HSA executable, harvest its device-global variable
  /// symbols, allocate + publish the managed variables it carries (each
  /// owned by this record), and cache everything under the key
  /// <tt>(OriginalKD, Preset)</tt>.
  ///
  /// Calling this again for a key that already has code objects loaded adds
  /// another one to that entry rather than failing. The addition is bound
  /// against everything already loaded under the key: every global variable
  /// those code objects define is declared in the new executable as an
  /// external agent global variable pointing at the address it already
  /// occupies (\c hsa_executable_agent_global_variable_define), so the loader
  /// resolves the addition's undefined references to them as it loads it. HSA
  /// cannot load into an executable that is already frozen, so each code
  /// object keeps its own executable and this is what ties them together.
  ///
  /// For a reference to resolve this way, the addition has to reach the
  /// variable through the GOT, which the AMDGPU backend only emits for an
  /// \c extern declaration of *default* visibility. HIP gives \c __device__
  /// globals protected visibility, whose references are PC-relative and bound
  /// at static-link time; ld.lld rejects those outright when they are
  /// undefined, so such an addition fails to link rather than mis-resolving.
  ///
  /// Only the first code object loaded under a key must carry a kernel; later
  /// additions may carry none, in which case a symbol with a zero \c handle is
  /// returned. Whichever kernel the *first* code object carries stays the
  /// instrumented kernel that \c overrideWithInstrumented substitutes into a
  /// dispatch — additions contribute code and globals without changing what
  /// runs.
  ///
  /// If the relocatable also carries an <tt>amdgcn.device.init</tt> global
  /// constructor kernel (emitted by the AMDGPU backend's
  /// <tt>amdgpu-lower-ctor-dtor</tt> pass for dynamically-initialized
  /// <tt>__device__</tt> globals), it is dispatched once, synchronously,
  /// right after the managed variables are published and before this call
  /// returns. If it carries an <tt>amdgcn.device.fini</tt> global
  /// destructor kernel, it is cached on the record and dispatched by
  /// \c unloadInstrumentedIfExists right before the executable is torn
  /// down. See \c launchSingleWorkItemKernelAndWait for how those two are
  /// dispatched.
  ///
  /// Every kernel this finds is described by its kernel descriptor as the
  /// code object defines it, read off the host copy of the ELF; the loader
  /// never re-queries the individual segment sizes back out of HSA.
  ///
  /// Takes ownership of \p Relocatable for the lifetime of the resulting
  /// record — the HSA code-object reader keeps a pointer into it, and so do
  /// the host kernel-descriptor pointers cached on the record.
  ///
  /// \param OriginalKD pointer to the kernel descriptor on the device of
  /// the original (un-instrumented) kernel. The agent that owns the KD's
  /// allocation is queried via \c hsa_amd_pointer_info, which works
  /// regardless of whether the KD was published through the HSA loader or
  /// allocated directly out of an HSA memory pool.
  llvm::Expected<hsa_executable_symbol_t>
  loadInstrumented(std::unique_ptr<llvm::MemoryBuffer> Relocatable,
                   const llvm::amdhsa::kernel_descriptor_t *OriginalKD,
                   uint64_t Preset = 0);

  /// Tear down every HSA executable + reader cached under
  /// <tt>(OriginalKD, Preset)</tt> and remove the entry from the
  /// cache. Code objects are torn down in reverse load order, since a later
  /// one was bound against the globals of the earlier ones and its destructor
  /// may still read them. For each code object that carries an
  /// <tt>amdgcn.device.fini</tt> global destructor kernel (see
  /// \c loadInstrumented), it is dispatched once, synchronously, before that
  /// code object's executable is destroyed. Idempotent: a missing entry is
  /// success. Returns any joined HSA destruction errors.
  llvm::Error unloadInstrumentedIfExists(
      const llvm::amdhsa::kernel_descriptor_t *OriginalKD, uint64_t Preset = 0);

  /// Rewrite \p Packet 's <tt>kernel_object</tt> to the cached instrumented
  /// variant for <tt>(Packet.kernel_object, Preset)</tt>, bump
  /// <tt>private_segment_size</tt> to at least the cached value, allocate +
  /// fill the extended kernarg buffer the instrumented kernel expects (see
  /// \ref extended_kernarg_abi "Extended kernarg buffer ABI"), and point
  /// \c Packet.kernarg_address at it.
  ///
  /// The returned handle owns the extended kernarg buffer. The caller MUST
  /// keep it alive until the dispatch completes; releasing while the device
  /// may still be reading the buffer is undefined behaviour. Typical pattern:
  /// move the handle into the completion callback for
  /// \c Packet.completion_signal.
  ///
  /// \p Queue is the queue the dispatch is being pushed onto; the extended
  /// buffer's \c HiddenPrivateBase and \c HiddenSharedBase slots are read
  /// out of the queue's AMD extension apertures.
  ///
  /// The variant is the kernel of the *first* code object loaded under the
  /// key; code objects added by later \c loadInstrumented calls never change
  /// what a dispatch runs. Errors if no such cached variant exists.
  ///
  /// The extended buffer inherits the record's hostcall and heap buffers
  /// (if either was stood up for a constructor/destructor kernel and the
  /// instrumented kernel also declares them). The record's buffers are sized
  /// for a single wave; a many-wave instrumented dispatch that shares them
  /// with a ctor/dtor kernel would over-subscribe. Hidden slots the
  /// instrumented kernel declares but which the loader has not stood up a
  /// buffer for are left zeroed — writeHiddenKernelArguments documents the
  /// per-kind zero semantics.
  llvm::Expected<ExtendedKernargBuffer>
  overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                           const hsa_queue_t &Queue, uint64_t Preset = 0);

  /// Resolve a device-global variable \p Name to its
  /// \c hsa_executable_symbol_t inside the code objects cached under
  /// <tt>(OriginalKD, Preset)</tt>, searched in load order so the code object
  /// that first defined the variable answers for it. Callers derive the loaded
  /// address / size via \c hsa::executableSymbolGet*. Errors if no such record
  /// or symbol.
  llvm::Expected<hsa_executable_symbol_t>
  lookupGlobalVariable(llvm::StringRef Name,
                       const llvm::amdhsa::kernel_descriptor_t *OriginalKD,
                       uint64_t Preset = 0);

  /// Tear down every cached record. Joins all HSA destruction errors
  /// and returns the joined \c llvm::Error (success only if every
  /// teardown succeeded). Idempotent.
  llvm::Error unloadAll();

  /// Walk \p Exec 's loaded code objects and erase any cache records
  /// whose original KD pointer falls inside one of those loaded ranges.
  /// Called by the trait subclass from inside the
  /// \c hsa_executable_destroy interceptor.
  llvm::Error invalidateOriginalExec(hsa_executable_t Exec);

  /// Accessors for the HSA API-table snapshots. These expose the underlying,
  /// pre-interception function pointers so sibling traits (e.g. the
  /// instrumentation pipeline) can drive HSA from inside a \c withInstance()
  /// callback. (The tool-code loader is HSA-free and no longer holds these;
  /// the launcher is now the canonical owner.)
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &
  getCoreApiTableSnapshot() const {
    return CoreApi;
  }
  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &
  getAmdExtTableSnapshot() const {
    return AmdExt;
  }
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER> &
  getLoaderTableSnapshot() const {
    return Loader;
  }

protected:
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi;
  /// AMD extension table — needed for \c hsa_amd_pointer_info and the
  /// managed-variable allocation paths (memory pools / SVM).
  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt;
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &Loader;

  /// Reader/writer lock: \c lookupGlobalVariable takes the reader lock;
  /// every cache mutation path takes the writer lock. \c overrideWithInstrumented
  /// takes the writer lock because it allocates the extended kernarg buffer
  /// out of an HSA kernarg region and needs a consistent view of the record
  /// while it does so.
  mutable llvm::sys::RWMutex Mutex;

  /// Result of one managed-variable storage allocation. Captures everything
  /// the free path needs so it doesn't have to re-decide the API path.
  struct ManagedAlloc {
    void *Ptr{nullptr};
    /// Bytes actually reserved — page-rounded on the SVM/HMM path, equal to
    /// the requested size on the pool path.
    size_t AllocSize{0};
    /// The managed variable's declared size (from its \c .managed companion
    /// symbol).
    size_t Size{0};
    /// True iff this allocation took the SVM/HMM path.
    bool ViaSvm{false};
  };

  /// One hidden (implicit) kernel argument slot, as declared by the kernel's
  /// entry in the code object's <tt>amdhsa.kernels</tt> metadata. Driving the
  /// kernarg fill off the metadata rather than a hard-coded struct keeps this
  /// correct across code-object versions, which reshuffle the hidden block.
  struct HiddenArgInfo {
    /// The argument's <tt>.value_kind</tt>; always one of the
    /// <tt>Hidden*</tt> kinds.
    amdgpu::hsamd::ValueKind Kind{amdgpu::hsamd::ValueKind::Unknown};
    /// The argument's <tt>.offset</tt> into the kernarg segment.
    uint32_t Offset{0};
    /// The argument's <tt>.size</tt>, in bytes.
    uint32_t Size{0};
  };

  /// Everything needed to dispatch one kernel of a loaded instrumented
  /// executable.
  struct LoadedKernelInfo {
    /// The kernel's symbol inside the instrumented executable.
    hsa_executable_symbol_t Symbol{};
    /// Address of the kernel descriptor on the device. This is what an AQL
    /// dispatch packet's \c kernel_object field takes.
    uint64_t KDDeviceAddress{0};
    /// The host copy of that same kernel descriptor, pointing into the
    /// instrumented code object's ELF bytes. Every segment size the dispatch
    /// needs is read straight from here instead of being re-queried field by
    /// field through \c hsa_executable_symbol_get_info. Kept alive by
    /// \c InstrumentedRecord::RelocatableBuffer.
    const llvm::amdhsa::kernel_descriptor_t *KDHostAddress{nullptr};
    /// The kernel's hidden arguments, in metadata order.
    llvm::SmallVector<HiddenArgInfo, 8> HiddenArgs;
    /// True iff the kernel expects the original app kernarg pointer at the
    /// first 8 bytes of the extended kernarg buffer. Derived from
    /// \c amdhsa.kernels metadata: the patcher restages the app kernel's
    /// first kernarg as an 8-byte address record at offset 0 (see
    /// \c TargetModulePatcherPass::emitKernargBufferExpansion), so the
    /// prefix's presence is exactly whether the metadata declares any
    /// non-hidden argument. See
    /// \ref extended_kernarg_abi "Extended kernarg buffer ABI".
    bool HasAppKernargPrefix{false};
  };

  /// One code object loaded under a <tt>(OriginalKD, Preset)</tt> entry. The
  /// first one loaded carries the instrumented kernel; each later
  /// \c loadInstrumented call for the same key appends another, bound against
  /// the globals of the ones already there.
  struct InstrumentedRecord {
    /// Caller-supplied relocatable bytes. Outlives \c Reader — the HSA
    /// code-object reader holds a non-owning view into this buffer, and
    /// every \c LoadedKernelInfo::KDHostAddress points into it.
    std::unique_ptr<llvm::MemoryBuffer> RelocatableBuffer;
    hsa_code_object_reader_t Reader{};
    hsa_executable_t Exec{};
    /// This code object's kernel, and \c std::nullopt if it carries none —
    /// only the first code object under a key is required to have one.
    /// \c loadInstrumented hands its \c Symbol back to the caller, and for the
    /// first code object of an entry \c overrideWithInstrumented dispatches
    /// off of it.
    std::optional<LoadedKernelInfo> Kernel;
    /// Agent the kernel runs on.
    hsa_agent_t Agent{};
    /// Device-global variable symbols harvested from this instrumented
    /// executable (name → symbol), for host readback via
    /// \c lookupGlobalVariable.
    llvm::StringMap<hsa_executable_symbol_t> NameToVarSymbol;
    /// Managed-variable storage this record owns — one entry per
    /// \c <base>.managed symbol in its relocatable. Freed when the record is
    /// erased.
    llvm::SmallVector<ManagedAlloc, 2> ManagedAllocs;
    /// This record's <tt>amdgcn.device.fini</tt> global-destructor kernel, if
    /// its relocatable carried one; \c std::nullopt otherwise. Dispatched
    /// once by \c eraseRecordLocked right before the executable is destroyed.
    std::optional<LoadedKernelInfo> DtorKernel;
    /// Constant \c printf format strings this code object's metadata carries,
    /// needed to decode what its kernels write into a \c hidden_printf_buffer.
    PrintfFormatStringMap PrintfFormatStrings;
    /// Buffer through which this record's constructor and destructor kernels
    /// reach the host, if either declares a \c hidden_hostcall_buffer
    /// argument; \c nullptr otherwise.
    ///
    /// It has to outlive the constructor dispatch rather than be torn down
    /// with it: a global constructor that allocates device memory through the
    /// hostcall device-memory service expects that memory to still be there
    /// when the matching destructor frees it, and the service tracks those
    /// allocations here.
    std::unique_ptr<HostcallBufferAllocation> HostcallBufferAlloc;
    /// Heap backing device-side \c malloc for this record's constructor and
    /// destructor kernels, if either declares a \c hidden_heap_v1 argument;
    /// \c nullptr otherwise. Record-scoped for the same reason the hostcall
    /// buffer is: what the constructor allocates has to still be there when
    /// the destructor frees it.
    std::unique_ptr<DeviceHeapBuffer> HeapBuffer;
  };

  /// Cache key — original KD pointer + preset.
  struct Key {
    const llvm::amdhsa::kernel_descriptor_t *KD;
    uint64_t Preset;
  };

  struct KeyDenseMapInfo {
    using PtrInfo =
        llvm::DenseMapInfo<const llvm::amdhsa::kernel_descriptor_t *>;
    using U64Info = llvm::DenseMapInfo<uint64_t>;
    static Key getEmptyKey() {
      return Key{PtrInfo::getEmptyKey(), U64Info::getEmptyKey()};
    }
    static unsigned getHashValue(const Key &K) {
      return llvm::detail::combineHashValue(PtrInfo::getHashValue(K.KD),
                                            U64Info::getHashValue(K.Preset));
    }
    static bool isEqual(const Key &L, const Key &R) {
      return L.KD == R.KD && L.Preset == R.Preset;
    }
  };

  /// Every code object loaded under one key, in load order. Never empty: an
  /// entry appears when its first code object loads and is erased whole.
  using CodeObjectList = llvm::SmallVector<InstrumentedRecord, 1>;

  /// Authoritative storage of every cached record.
  llvm::DenseMap<Key, CodeObjectList, KeyDenseMapInfo> ByOriginal;

  /// Cached \c HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED query result.
  std::optional<bool> HmmSupportedCache;

  /// Answers the hostcall requests made by every kernel this loader
  /// dispatches. One listener serves every record, and it is only stood up —
  /// thread and all — once a kernel that can actually make a hostcall is
  /// about to run.
  std::unique_ptr<HostcallListener> Listener;

  /// Destroy every code object of the entry pointed to by \p It — in reverse
  /// load order, since a later one is bound against the globals of the earlier
  /// ones — and erase the entry from \c ByOriginal. Caller must hold the
  /// writer lock.
  llvm::Error eraseRecordLocked(
      llvm::DenseMap<Key, CodeObjectList, KeyDenseMapInfo>::iterator It);

  /// Dispatch \p Rec 's global destructor kernel if it has one, then destroy
  /// its HSA executable + reader and free the managed-variable storage and
  /// record-scoped buffers it owns. Does not touch \c ByOriginal. Caller must
  /// hold the writer lock.
  llvm::Error eraseCodeObjectLocked(InstrumentedRecord &Rec);

  /// Declare every global variable defined by the already-loaded code objects
  /// \p Prior inside the not-yet-loaded executable \p Exec, each pointing at
  /// the device address it already occupies, so that the code object about to
  /// be loaded into \p Exec resolves its undefined references to them.
  ///
  /// A definition injected this way is itself reported by
  /// \c hsa_executable_iterate_agent_symbols, so the same name shows up again
  /// in every later code object's symbol table; the earliest definition of a
  /// name wins and the rest are skipped, because HSA rejects defining one
  /// twice. Caller must hold the writer lock.
  llvm::Error
  defineGlobalsOfPriorCodeObjects(llvm::ArrayRef<InstrumentedRecord> Prior,
                                  hsa_executable_t Exec, hsa_agent_t Agent);

  /// Allocate + publish every managed variable carried by the instrumented
  /// executable \p Rec just loaded: for each \c <base>.managed ELF symbol,
  /// allocate host-coherent storage, copy the init bytes in, and publish the
  /// device base symbol to point at it (via \c hsa_memory_copy). Each
  /// allocation is owned by \p Rec (appended to \c Rec.ManagedAllocs) and freed
  /// when the record is erased; on failure the allocations made so far are
  /// freed before returning. Caller must hold the writer lock.
  llvm::Error loadManagedVarsForRecord(const llvm::object::ObjectFile &Obj,
                                       InstrumentedRecord &Rec);

  //===-------------------------------------------------------------------===//
  // Managed-variable storage allocation (HMM-aware): each instrumented copy
  // owns its managed-var storage.
  //===-------------------------------------------------------------------===//

  /// Pick a host fine-grain memory pool suitable for backing managed
  /// variables (the non-HMM path).
  static llvm::Expected<hsa_amd_memory_pool_t>
  selectManagedVarPool(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                       hsa_agent_t CpuAgent);

  /// HMM-aware managed-storage allocation. On HMM systems reserves a
  /// page-aligned SVM range accessible from \p GpuAgents; otherwise allocates
  /// from \p Pool and grants \p GpuAgents access.
  static llvm::Expected<ManagedAlloc>
  allocateManagedStorage(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                         llvm::ArrayRef<hsa_agent_t> GpuAgents,
                         hsa_amd_memory_pool_t Pool, size_t Size,
                         unsigned Align, bool HmmSupported);

  /// Free a \c ManagedAlloc produced by \c allocateManagedStorage.
  static llvm::Error
  freeManagedStorage(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                     const ManagedAlloc &Alloc);

  /// Lazily probe \c HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED and cache the result.
  /// Caller must hold the writer lock.
  llvm::Expected<bool> getHmmSupported();

  //===-------------------------------------------------------------------===//
  // Global constructor / destructor kernel dispatch
  //===-------------------------------------------------------------------===//

  /// Looks up the kernel named \p KernelName inside the just-loaded
  /// instrumented executable \p Exec and, if present, returns everything
  /// needed to dispatch it: its executable symbol, its kernel-descriptor
  /// address on the device, a pointer to the host copy of that descriptor
  /// inside \p Obj, and its hidden arguments as declared by \p Obj 's
  /// metadata. Expects \c std::nullopt (not an error) if \p Obj declares no
  /// such kernel — used to detect the optional global constructor/destructor
  /// kernels.
  ///
  /// The returned \c LoadedKernelInfo::KDHostAddress points into \p Obj 's
  /// backing buffer, so the caller must keep that buffer alive for as long as
  /// it holds on to the result.
  ///
  /// \param MetadataDoc \p Obj 's already-parsed metadata document, which the
  /// hidden arguments are read out of.
  llvm::Expected<std::optional<LoadedKernelInfo>>
  findKernelIfPresent(const object::AMDGCNObjectFile &Obj,
                      llvm::msgpack::Document &MetadataDoc,
                      hsa_executable_t Exec, hsa_agent_t Agent,
                      llvm::StringRef KernelName);

  /// The largest per-work-item private (scratch) segment size that can be
  /// dispatched on \p Agent, in bytes.
  ///
  /// The hardware encodes a wave's scratch allocation in
  /// \c COMPUTE_TMPRING_SIZE.WAVESIZE, a 15-bit (18-bit on gfx12+) count of
  /// 256-byte granules, which the runtime divides down to a per-work-item
  /// figure; the AMDGPU backend additionally reserves 64 bytes of every
  /// work-item's private segment for its own use. This mirrors ROCclr's
  /// \c GetMaxStackSize, i.e. the ceiling HIP enforces on
  /// <tt>hipLimitStackSize</tt>.
  static llvm::Expected<uint32_t>
  getMaxPrivateSegmentSize(const hsa::ApiTableContainer<::CoreApiTable> &CoreApi,
                           hsa_agent_t Agent);

  /// \returns \c true if \p Kernel declares a hidden argument of kind
  /// \p Kind, i.e. if the dispatch has to supply one.
  static bool declaresHiddenArg(const LoadedKernelInfo &Kernel,
                                amdgpu::hsamd::ValueKind Kind);

  /// Addresses of the buffers a dispatch stood up behind the hidden arguments
  /// that need one. A null member means the dispatch provided nothing for
  /// that argument and its slot is left zeroed.
  struct HiddenArgBufferAddresses {
    /// Buffer the kernel's hostcalls are submitted through.
    void *HostcallBuffer{nullptr};
    /// Buffer a buffered-\c printf kernel writes its records into.
    void *PrintfBuffer{nullptr};
    /// Heap backing device-side \c malloc.
    void *Heap{nullptr};
    /// Cooperative-groups grid barrier state.
    void *GridSyncInfo{nullptr};
    /// Wrapper a device-enqueued child reports completion against.
    void *CompletionAction{nullptr};
  };

  /// Fills in the hidden (implicit) kernel arguments \p HiddenArgs declares
  /// inside the zero-initialized kernarg buffer \p Kernarg, for a dispatch
  /// described by \p Packet on \p Queue, pointing each argument that needs a
  /// buffer at the matching member of \p Buffers. Each argument is written at
  /// the \c .offset / \c .size its metadata declares.
  static llvm::Error writeHiddenKernelArguments(
      llvm::MutableArrayRef<uint8_t> Kernarg,
      llvm::ArrayRef<HiddenArgInfo> HiddenArgs,
      const hsa_kernel_dispatch_packet_t &Packet, const hsa_queue_t &Queue,
      const HiddenArgBufferAddresses &Buffers);

  /// Populate the extended kernarg buffer \p Kernarg per
  /// \ref extended_kernarg_abi "Extended kernarg buffer ABI": prefix
  /// \p AppKernargPtr into bytes \c [0,8) when \p HasAppKernargPrefix is
  /// true, then run \c writeHiddenKernelArguments over the same buffer to
  /// fill the trailing hidden-arg block. \p Kernarg is assumed
  /// zero-initialized; the prefix write skips zero-fill and every hidden
  /// slot is written at its metadata-declared offset.
  ///
  /// Extracted as its own static so tests can exercise the composition
  /// without an HSA runtime. The extra buffer-size checks it does are
  /// redundant against the runtime path (which sizes the buffer from
  /// \c kernarg_size), but they let a caller-side test provide any buffer
  /// it wants and still catch invalid layouts.
  static llvm::Error fillExtendedKernargBuffer(
      llvm::MutableArrayRef<uint8_t> Kernarg, bool HasAppKernargPrefix,
      const void *AppKernargPtr, llvm::ArrayRef<HiddenArgInfo> HiddenArgs,
      const hsa_kernel_dispatch_packet_t &Packet, const hsa_queue_t &Queue,
      const HiddenArgBufferAddresses &Buffers);

  /// Returns the loader's hostcall listener, starting it (and its thread) if
  /// this is the first kernel that needs one. Caller must hold the writer
  /// lock.
  llvm::Expected<HostcallListener *> getOrCreateHostcallListener();

  /// Allocates a hostcall buffer for \p Agent, sized for a single-wave
  /// dispatch, and registers it with the loader's listener so packets pushed
  /// onto it are answered. Caller must hold the writer lock.
  llvm::Expected<std::unique_ptr<HostcallBufferAllocation>>
  createAndRegisterHostcallBuffer(hsa_agent_t Agent);

  /// Deregisters \p Buffer from the loader's listener, so that it can be
  /// freed without the listener still walking it.
  void unregisterHostcallBuffer(HostcallBufferAllocation &Buffer);

  /// Synchronously dispatches \p Kernel over a single work-item on
  /// <tt>Rec.Agent</tt> using a private queue, and blocks until it completes.
  /// Used to invoke the <tt>amdgcn.device.init</tt> /
  /// <tt>amdgcn.device.fini</tt> global constructor/destructor kernels the
  /// AMDGPU backend's <tt>amdgpu-lower-ctor-dtor</tt> pass may emit into an
  /// instrumented relocatable. \p Kernel must be one of \p Rec 's own
  /// kernels, since the record owns the buffers the dispatch hands it.
  ///
  /// Every dispatch parameter is read out of \p Kernel 's host kernel
  /// descriptor. The kernarg segment is backed by a zero-filled buffer of
  /// \c kernarg_size bytes — non-empty even for these argument-less kernels,
  /// because of the COV5 hidden arguments — into which
  /// \c writeHiddenKernelArguments then fills the hidden block. When the
  /// descriptor sets \c USES_DYNAMIC_STACK the dispatch reserves the largest
  /// private segment the agent supports: the constructor/destructor kernels
  /// reach their callees through indirect calls, so the statically-computed
  /// \c private_segment_fixed_size is 0 no matter how much stack those
  /// callees need, and there is no way to recover the real requirement.
  ///
  /// Hidden arguments that need a buffer are supplied only when \p Kernel
  /// declares them. Record-scoped ones (the hostcall buffer and the device
  /// heap) come from \p Rec; the rest — the printf buffer, the grid sync
  /// structure and the completion action — are allocated for this dispatch
  /// and released with it. A declared \c hidden_printf_buffer is drained onto
  /// the host's \c stdout / \c stderr once the dispatch completes.
  llvm::Error
  launchSingleWorkItemKernelAndWait(const InstrumentedRecord &Rec,
                                    const LoadedKernelInfo &Kernel);

  /// Free an extended kernarg buffer previously handed out by
  /// \c overrideWithInstrumented. \p Ptr must have been allocated from an
  /// agent kernarg region via \c hsa_memory_allocate; freeing something
  /// else is undefined behaviour. A null \p Ptr is a no-op. Thread-safe
  /// because \c hsa_memory_free is; called by \c ExtendedKernargBuffer 's
  /// destructor / \c release without any lock on the loader.
  llvm::Error releaseExtendedKernargBuffer(void *Ptr);

  friend class ExtendedKernargBuffer;
};

/// Owns an extended kernarg buffer built by
/// \c InstrumentedKernelLoaderAndLauncher::overrideWithInstrumented (see
/// \ref extended_kernarg_abi "Extended kernarg buffer ABI"). Move-only. On
/// destruction — or on an explicit \c release call — the buffer is returned
/// to HSA. Idempotent: a default-constructed or already-released handle
/// releases nothing.
///
/// The caller MUST keep this handle alive until the dispatch it backs has
/// completed. Releasing while the device is still reading the buffer is
/// undefined behaviour, and one that HSA cannot detect for the loader.
class ExtendedKernargBuffer {
public:
  ExtendedKernargBuffer() = default;
  ExtendedKernargBuffer(const ExtendedKernargBuffer &) = delete;
  ExtendedKernargBuffer &operator=(const ExtendedKernargBuffer &) = delete;

  ExtendedKernargBuffer(ExtendedKernargBuffer &&Other) noexcept
      : Owner(Other.Owner), Ptr(Other.Ptr) {
    Other.Owner = nullptr;
    Other.Ptr = nullptr;
  }
  ExtendedKernargBuffer &operator=(ExtendedKernargBuffer &&Other) noexcept {
    if (this != &Other) {
      llvm::consumeError(release());
      Owner = Other.Owner;
      Ptr = Other.Ptr;
      Other.Owner = nullptr;
      Other.Ptr = nullptr;
    }
    return *this;
  }

  ~ExtendedKernargBuffer() { llvm::consumeError(release()); }

  /// Device-visible pointer that was written to
  /// \c hsa_kernel_dispatch_packet_t::kernarg_address. Null when this
  /// handle is empty (either default-constructed or the instrumented
  /// kernel had a zero-byte kernarg segment).
  void *getKernargAddress() const { return Ptr; }

  /// True iff this handle owns no buffer.
  bool empty() const { return Ptr == nullptr; }

  /// Return the buffer to HSA now. Idempotent; the handle is emptied
  /// whether the underlying free succeeded or not, so a leftover error
  /// from a failed free is surfaced once and only once.
  llvm::Error release() {
    if (Ptr == nullptr)
      return llvm::Error::success();
    void *P = Ptr;
    InstrumentedKernelLoaderAndLauncher *O = Owner;
    Ptr = nullptr;
    Owner = nullptr;
    return O->releaseExtendedKernargBuffer(P);
  }

private:
  friend class InstrumentedKernelLoaderAndLauncher;
  ExtendedKernargBuffer(InstrumentedKernelLoaderAndLauncher *Owner, void *Ptr)
      : Owner(Owner), Ptr(Ptr) {}

  InstrumentedKernelLoaderAndLauncher *Owner{nullptr};
  void *Ptr{nullptr};
};

/// \brief CRTP trait that adds an \c hsa_executable_destroy interceptor
/// on top of \c InstrumentedKernelLoaderAndLauncher.
template <typename Derived>
class InstrumentedKernelLoaderAndLauncherTrait
    : public InstrumentedKernelLoaderAndLauncher {
private:
  inline static decltype(hsa_executable_destroy)
      *UnderlyingHsaExecutableDestroyFn{};

  std::unique_ptr<rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>
      HsaWrapperInstaller;

  static hsa_status_t hsaExecutableDestroyWrapper(hsa_executable_t Exec) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        UnderlyingHsaExecutableDestroyFn != nullptr,
        "The underlying hsa_executable_destroy function for "
        "InstrumentedKernelLoaderAndLauncherTrait is nullptr"));
    (void)Singleton<Derived>::withInstance([&](Derived &Inst) {
      auto &Self = static_cast<InstrumentedKernelLoaderAndLauncher &>(
          static_cast<InstrumentedKernelLoaderAndLauncherTrait<Derived> &>(
              Inst));
      // Swallow the accumulated invalidation error here, as the
      // hsa_executable_destroy ABI cannot surface llvm::Errors.
      llvm::consumeError(Self.invalidateOriginalExec(Exec));
    });
    return UnderlyingHsaExecutableDestroyFn(Exec);
  }

public:
  InstrumentedKernelLoaderAndLauncherTrait(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &Loader,
      llvm::Error &Err)
      : InstrumentedKernelLoaderAndLauncher(CoreApi, AmdExt, Loader) {
    llvm::ErrorAsOutParameter EAO(Err);
    HsaWrapperInstaller = std::make_unique<
        rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
        Err, std::make_tuple(&::CoreApiTable::hsa_executable_destroy_fn,
                             std::ref(UnderlyingHsaExecutableDestroyFn),
                             hsaExecutableDestroyWrapper));
  }

  ~InstrumentedKernelLoaderAndLauncherTrait() = default;

  InstrumentedKernelLoaderAndLauncherTrait(
      const InstrumentedKernelLoaderAndLauncherTrait &) = delete;
  InstrumentedKernelLoaderAndLauncherTrait &
  operator=(const InstrumentedKernelLoaderAndLauncherTrait &) = delete;
};

} // namespace luthier

#endif // LUTHIER_HSA_TOOLING_INSTRUMENTED_KERNEL_LOADER_AND_LAUNCHER_H
