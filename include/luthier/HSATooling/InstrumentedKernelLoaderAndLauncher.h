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
///     on the device.
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
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"
#include <cstdint>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/RWMutex.h>
#include <memory>
#include <tuple>
#include <utility>

namespace luthier {

class DeviceToolCodeLoader;

/// \brief Per-tool cache of instrumented HSA kernel executables.
///
/// Each cached record packages, for a single
/// <tt>(OriginalKD, Preset)</tt> tuple (where \c OriginalKD is a pointer
/// to the kernel descriptor on the device — the same value that lives in
/// \c hsa_kernel_dispatch_packet_t::kernel_object), the relocatable ELF
/// bytes (owned), the HSA code-object reader created over them, the HSA
/// executable they were loaded into, and the resulting instrumented
/// kernel symbol + descriptor address + private segment size.
///
/// \c loadInstrumented is the cold-path entry point; it takes ownership
/// of \p Relocatable so the HSA code-object reader's view into host
/// memory stays valid for the record's lifetime. The relocatable is
/// expected to contain exactly one kernel function; the launcher does
/// not validate its name against any expected name.
/// \c overrideWithInstrumented is the hot path called from a packet
/// interceptor and is reader-locked.
///
/// \c invalidateOriginalExec is invoked by the
/// \c InstrumentedKernelLoaderAndLauncherTrait subclass from inside its
/// \c hsa_executable_destroy interceptor whenever an application
/// executable is about to be torn down. It walks the executable's
/// loaded code objects and erases any cache records whose original KD
/// pointer falls inside one of those loaded ranges.
class InstrumentedKernelLoaderAndLauncher {
public:
  InstrumentedKernelLoaderAndLauncher(
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
      const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
          &Loader,
      DeviceToolCodeLoader &DeviceCode);

  ~InstrumentedKernelLoaderAndLauncher();

  InstrumentedKernelLoaderAndLauncher(
      const InstrumentedKernelLoaderAndLauncher &) = delete;
  InstrumentedKernelLoaderAndLauncher &
  operator=(const InstrumentedKernelLoaderAndLauncher &) = delete;

  /// Parse \p Relocatable as an AMDGCN ELF, require it contain exactly
  /// one kernel function (any name — assumed to be the instrumented
  /// kernel), resolve the relocatable's UND globals against the
  /// per-agent symbols published by the tool's \c DeviceToolCodeLoader
  /// on the agent that owns \p OriginalKD (resolved via the application
  /// \c LoadedCodeObjectCache), create + load + freeze a fresh HSA
  /// executable, and cache the resulting kernel symbol under the key
  /// <tt>(OriginalKD, Preset)</tt>.
  ///
  /// Takes ownership of \p Relocatable for the lifetime of the
  /// resulting record — the HSA code-object reader keeps a pointer
  /// into it.
  ///
  /// \param OriginalKD pointer to the kernel descriptor on the device
  /// of the original (un-instrumented) kernel. This is the value of the
  /// \c kernel_object field on \c hsa_kernel_dispatch_packet_t cast to
  /// \c const \c llvm::amdhsa::kernel_descriptor_t*. The agent that
  /// owns the KD's allocation is queried via \c hsa_amd_pointer_info,
  /// which works regardless of whether the KD was published through
  /// the HSA loader or allocated directly out of an HSA memory pool.
  ///
  /// Errors:
  ///   - an entry for the same <tt>(OriginalKD, Preset)</tt>
  ///     already exists
  ///   - the ELF contains zero or more than one kernel function
  ///   - \c hsa_amd_pointer_info fails or reports the pointer as
  ///     not owned by an HSA allocation
  ///   - any UND global symbol does not resolve in the
  ///     \c DeviceToolCodeLoader
  ///   - any UND function symbol is present
  ///   - any HSA failure (create / define / load / freeze /
  ///     symbol lookup)
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

  /// Tear down every cached record. Joins all HSA destruction errors
  /// and returns the joined \c llvm::Error (success only if every
  /// teardown succeeded). Idempotent.
  llvm::Error unloadAll();

  /// Walk \p Exec 's loaded code objects and erase any cache records
  /// whose original KD pointer falls inside one of those loaded ranges.
  /// Returns a joined \c llvm::Error of any HSA failures collected
  /// along the way. Called by the trait subclass from inside the
  /// \c hsa_executable_destroy interceptor before chaining to the next
  /// wrapper.
  llvm::Error invalidateOriginalExec(hsa_executable_t Exec);

protected:
  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi;
  /// AMD extension table — needed for \c hsa_amd_pointer_info to resolve
  /// the agent that owns the kernel descriptor allocation. This path
  /// works for both loader-published allocations and direct memory-pool
  /// allocations, so the launcher doesn't have to assume the KD lives
  /// inside a cached loaded code object.
  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt;
  const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      &Loader;
  DeviceToolCodeLoader &DeviceCode;

  /// Reader/writer lock: \c overrideWithInstrumented takes the reader
  /// lock; every cache mutation path (\c loadInstrumented,
  /// \c unloadInstrumentedIfExists, \c unloadAll,
  /// \c invalidateOriginalExec) takes the writer lock.
  mutable llvm::sys::RWMutex Mutex;

  /// One per <tt>(OriginalKD, Preset)</tt> entry. Stores the
  /// instrumented HSA executable and reader the launcher owns, plus
  /// the cached scalar fields needed by the override hot path.
  struct InstrumentedRecord {
    /// Caller-supplied relocatable bytes. Outlives \c Reader — the HSA
    /// code-object reader holds a non-owning view into this buffer.
    std::unique_ptr<llvm::MemoryBuffer> RelocatableBuffer;
    hsa_code_object_reader_t Reader{};
    hsa_executable_t Exec{};
    hsa_executable_symbol_t InstrumentedKernelSym{};
    /// Cached \c hsa_executable_symbol_get_info(KERNEL_OBJECT) of
    /// \c InstrumentedKernelSym. Goes into the dispatch packet on
    /// override.
    uint64_t InstrumentedKO{0};
    /// Cached \c
    /// hsa_executable_symbol_get_info(KERNEL_PRIVATE_SEGMENT_SIZE) of
    /// \c InstrumentedKernelSym. \c overrideWithInstrumented bumps
    /// \c Packet.private_segment_size to at least this value.
    uint32_t PrivateSegmentSize{0};
    /// Agent the kernel runs on. Held for diagnostics / future use.
    hsa_agent_t Agent{};
  };

  /// Cache key — original KD pointer + preset. \c overrideWithInstrumented
  /// builds the key from \c Packet.kernel_object cast to KD*.
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

  /// Destroy the HSA executable + reader pointed to by \p It and erase
  /// the entry from \c ByOriginal. Caller must hold the writer lock.
  /// Returns the joined HSA destruction errors (success if both
  /// teardowns succeeded).
  llvm::Error eraseRecordLocked(
      llvm::DenseMap<Key, InstrumentedRecord, KeyDenseMapInfo>::iterator It);
};

/// \brief CRTP trait that adds an \c hsa_executable_destroy interceptor
/// on top of \c InstrumentedKernelLoaderAndLauncher.
///
/// The wrapper runs the base's \c invalidateOriginalExec on the
/// to-be-destroyed executable before chaining to whatever
/// \c hsa_executable_destroy implementation was previously installed in
/// the HSA table (typically the \c LoadedCodeObjectCacheTrait wrapper,
/// then the real HSA function). This ordering guarantees that no
/// cached instrumented variant references an \c hsa_executable_t whose
/// teardown is already in flight.
///
/// Following the convention used by \c PacketMonitorTrait and
/// \c LoadedCodeObjectCacheTrait, the wrapper is NOT uninstalled at
/// trait teardown — uninstalling while the runtime may still call us
/// would cause a race condition. The wrapper does its tool-specific work inside
/// <tt>Singleton<Derived>::withInstance()</tt>.
template <typename Derived>
class InstrumentedKernelLoaderAndLauncherTrait
    : public InstrumentedKernelLoaderAndLauncher {
private:
  /// Saved pointer to the prior \c hsa_executable_destroy entry. Set by
  /// the wrapper installer; consulted by \c hsaExecutableDestroyWrapper
  /// to chain through.
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
      DeviceToolCodeLoader &DeviceCode, llvm::Error &Err)
      : InstrumentedKernelLoaderAndLauncher(CoreApi, AmdExt, Loader,
                                            DeviceCode) {
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
