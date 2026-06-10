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
///
/// \file InstrumentedKernelLoaderAndLauncher.h
/// Defines two collaborating classes:
///   - \c InstrumentedKernelLoaderAndLauncher: non-templated base.
///     Owns the per-tool cache of loaded instrumented-kernel HSA
///     executables keyed by the original kernel-descriptor pointer
///     on the device. Each cached executable is self-contained (its
///     device globals / managed variables are baked into its own copy),
///     so the launcher also owns per-record device-global symbol lookup
///     and managed-variable allocation / host-shadow publishing.
///   - \c InstrumentedKernelLoaderAndLauncherTrait<Derived>: header-only
///     CRTP trait that extends the base and installs an
///     \c hsa_executable_destroy interceptor driving
///     \c invalidateOriginalExec.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_INSTRUMENTED_KERNEL_LOADER_AND_LAUNCHER_H
#define LUTHIER_HSA_TOOLING_INSTRUMENTED_KERNEL_LOADER_AND_LAUNCHER_H

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSATooling/HostcallConsumer.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"
#include "luthier/ToolCodeGen/CustomKernargLayout.h"
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

namespace llvm {
namespace object {
class ObjectFile;
} // namespace object
} // namespace llvm

namespace luthier {

/// \brief Per-tool cache of instrumented HSA kernel executables.
///
/// Each cached record packages, for a single
/// <tt>(OriginalKD, Preset)</tt> tuple (where \c OriginalKD is a pointer
/// to the kernel descriptor on the device — the same value that lives in
/// \c hsa_kernel_dispatch_packet_t::kernel_object), the relocatable ELF
/// bytes (owned), the HSA code-object reader created over them, the HSA
/// executable they were loaded into, the resulting instrumented kernel
/// symbol + descriptor address + private segment size, the harvested
/// device-global variable symbols, and any managed-variable allocations
/// made for this instrumented copy.
///
/// \c loadInstrumented is the cold-path entry point; it takes ownership
/// of \p Relocatable so the HSA code-object reader's view into host
/// memory stays valid for the record's lifetime. The relocatable is
/// expected to contain exactly one kernel function and to be
/// self-contained (its device globals are defined in its own copy — the
/// launcher does not resolve UND globals against a global tool image).
/// \c overrideWithInstrumented is the hot path called from a packet
/// interceptor and is reader-locked.
/// TODO: Make passes register "handle records" so that the loader is able
/// to accumulate and publish the handles correctly
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
  /// one kernel function (any name — assumed to be the instrumented
  /// kernel), create + load + freeze a fresh HSA executable, harvest its
  /// device-global variable symbols, allocate + publish any managed
  /// variables it carries, and cache everything under the key
  /// <tt>(OriginalKD, Preset)</tt>.
  ///
  /// Takes ownership of \p Relocatable for the lifetime of the resulting
  /// record — the HSA code-object reader keeps a pointer into it.
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

  /// Tear down the HSA executable + reader cached under
  /// <tt>(OriginalKD, Preset)</tt> and remove the entry from the
  /// cache. Idempotent: a missing entry is success. Returns any
  /// joined HSA destruction errors.
  llvm::Error unloadInstrumentedIfExists(
      const llvm::amdhsa::kernel_descriptor_t *OriginalKD, uint64_t Preset = 0);

  /// Rewrite \p Packet 's <tt>kernel_object</tt> to the cached
  /// instrumented variant for <tt>(Packet.kernel_object, Preset)</tt>,
  /// and bump <tt>private_segment_size</tt> to at least the cached
  /// value. Returns an error if no such cached variant exists.
  llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                                       uint64_t Preset = 0);

  /// Resolve a device-global variable \p Name to its
  /// \c hsa_executable_symbol_t inside the instrumented executable cached
  /// under <tt>(OriginalKD, Preset)</tt>. The symbol lives in that one
  /// instrumented copy; callers derive the loaded address / size via
  /// \c hsa::executableSymbolGet*. Errors if no such record or symbol.
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

  /// Register a tool managed-variable host shadow: the \c void** the HIP
  /// \c __hipRegisterManagedVar emitted, keyed by the managed variable's
  /// device base symbol \p Name. On \c loadInstrumented the launcher writes
  /// the per-instrumented-copy device allocation pointer into the matching
  /// shadow. Called once per managed var by \c HSATool at construction.
  void registerManagedVarHostShadow(llvm::StringRef Name, void **Shadow) {
    llvm::sys::ScopedWriter W(Mutex);
    ManagedVarHostShadows[Name] = Shadow;
  }

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

  /// Reader/writer lock: \c overrideWithInstrumented takes the reader
  /// lock; every cache mutation path takes the writer lock.
  mutable llvm::sys::RWMutex Mutex;

  /// Result of one managed-variable storage allocation. Captures everything
  /// the free path needs so it doesn't have to re-decide the API path.
  struct ManagedAlloc {
    void *Ptr{nullptr};
    /// Bytes actually reserved — page-rounded on the SVM/HMM path, equal to
    /// the requested size on the pool path.
    size_t AllocSize{0};
    /// The managed variable's declared size (from its \c .managed companion
    /// symbol). Used to reject a later instrumented copy that claims a
    /// different size for the same managed variable.
    size_t Size{0};
    /// True iff this allocation took the SVM/HMM path.
    bool ViaSvm{false};
  };

  /// One per <tt>(OriginalKD, Preset)</tt> entry.
  struct InstrumentedRecord {
    /// Caller-supplied relocatable bytes. Outlives \c Reader — the HSA
    /// code-object reader holds a non-owning view into this buffer.
    std::unique_ptr<llvm::MemoryBuffer> RelocatableBuffer;
    hsa_code_object_reader_t Reader{};
    hsa_executable_t Exec{};
    hsa_executable_symbol_t InstrumentedKernelSym{};
    /// Cached kernel-object address; goes into the dispatch packet on
    /// override.
    uint64_t InstrumentedKO{0};
    /// Cached private segment size; \c overrideWithInstrumented bumps
    /// \c Packet.private_segment_size to at least this value.
    uint32_t PrivateSegmentSize{0};
    /// Agent the kernel runs on.
    hsa_agent_t Agent{};
    /// Device-global variable symbols harvested from this instrumented
    /// executable (name → symbol), for host readback via
    /// \c lookupGlobalVariable.
    llvm::StringMap<hsa_executable_symbol_t> NameToVarSymbol;
    /// True when the instrumented object carries a custom kernarg layout.
    bool HasCustomKernarg{false};
    CustomKernargLayout KernargLayout{};
    /// Most-recent custom kernarg buffer allocated by
    /// \c overrideWithInstrumented for this record (kernarg-pool memory).
    void *CustomKernargAlloc{nullptr};
    /// True when the instrumented object carries a \c .luthier.uses_hostcall
    /// marker. Implies \c HasCustomKernarg.
    bool UsesHostcall{false};
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

  /// Authoritative storage of every cached record.
  llvm::DenseMap<Key, InstrumentedRecord, KeyDenseMapInfo> ByOriginal;

  /// Tool managed-variable host shadows: device base symbol name → \c void**
  /// the HIP runtime registered. Populated by
  /// \c registerManagedVarHostShadow; consulted on \c loadInstrumented.
  llvm::StringMap<void **> ManagedVarHostShadows;

  /// One shared allocation per managed variable (keyed by device base symbol
  /// name), allocated the first time any instrumented copy declares it and
  /// reused — re-published into each subsequent copy's base symbol — for the
  /// life of the tool. Every instrumented copy of a given managed variable
  /// therefore sees the SAME storage (and the single host shadow is
  /// unambiguous). Freed in \c unloadAll. A later copy declaring a different
  /// size for the same variable is an error.
  llvm::StringMap<ManagedAlloc> SharedManagedVars;

  /// One hostcall buffer + listener per GPU agent (keyed by \c hsa_agent_t
  /// handle), shared across all instrumented kernels on that agent.
  llvm::DenseMap<uint64_t, std::unique_ptr<HostcallConsumer>>
      HostcallConsumersByAgent;

  /// Cached \c HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED query result.
  std::optional<bool> HmmSupportedCache;

  /// Return the device-visible hostcall buffer pointer for \p Agent, creating
  /// the per-agent consumer on first use. Caller must hold the writer lock.
  llvm::Expected<void *> getOrCreateHostcallBuffer(hsa_agent_t Agent);

  /// Destroy the HSA executable + reader pointed to by \p It, free any managed
  /// allocations and in-flight kernarg buffer, and erase the entry from
  /// \c ByOriginal. Caller must hold the writer lock.
  llvm::Error eraseRecordLocked(
      llvm::DenseMap<Key, InstrumentedRecord, KeyDenseMapInfo>::iterator It);

  /// Allocate, fill, and install a Luthier-managed custom kernarg buffer for a
  /// dispatch of the kernel cached in \p Rec. Caller must hold the writer lock.
  llvm::Error buildCustomKernargBuffer(InstrumentedRecord &Rec,
                                       hsa_kernel_dispatch_packet_t &Packet);

  /// Allocate + publish every managed variable carried by the instrumented
  /// executable \p Rec just loaded: for each \c <base>.managed ELF symbol,
  /// allocate host-coherent storage, copy the init bytes in, publish the
  /// device base symbol to point at it (via \c hsa_memory_copy), and write the
  /// tool's host shadow (if registered). Allocations are recorded in \p Rec.
  /// Caller must hold the writer lock.
  llvm::Error loadManagedVarsForRecord(const llvm::object::ObjectFile &Obj,
                                       InstrumentedRecord &Rec);

  //===-------------------------------------------------------------------===//
  // Managed-variable storage allocation (HMM-aware), moved here from the tool
  // code loader: each instrumented copy owns its managed-var storage.
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

  /// Wrapper intentionally not uninstalled — see class doc.
  ~InstrumentedKernelLoaderAndLauncherTrait() = default;

  InstrumentedKernelLoaderAndLauncherTrait(
      const InstrumentedKernelLoaderAndLauncherTrait &) = delete;
  InstrumentedKernelLoaderAndLauncherTrait &
  operator=(const InstrumentedKernelLoaderAndLauncherTrait &) = delete;
};

} // namespace luthier

#endif // LUTHIER_HSA_TOOLING_INSTRUMENTED_KERNEL_LOADER_AND_LAUNCHER_H
