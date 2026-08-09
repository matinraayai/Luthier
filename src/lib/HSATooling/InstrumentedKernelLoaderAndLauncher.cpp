//===-- InstrumentedKernelLoaderAndLauncher.cpp ---------------------------===//
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
#include "luthier/HSATooling/InstrumentedKernelLoaderAndLauncher.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/CodeObjectReader.h"
#include "luthier/HSA/Executable.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/HSA/ISA.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/HSA/Memory.h"
#include "luthier/HSA/MemoryPool.h"
#include "luthier/HSA/Queue.h"
#include "luthier/HSA/SVM.h"
#include "luthier/HSA/Signal.h"
#include "luthier/HSA/VMEM.h"
#include "luthier/Linker/Linker.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/Object/ObjectFileUtils.h"

#include <cstring>
#include <hsa/amd_hsa_queue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/TargetParser/AMDGPUTargetParser.h>
#include <vector>

#define DEBUG_TYPE "luthier-instrumented-kernel-loader-and-launcher"

namespace luthier {

namespace {

/// Name of the global constructor kernel emitted by the AMDGPU backend
constexpr llvm::StringLiteral GlobalCtorKernelName = "amdgcn.device.init";
/// Name of the global destructor kernel emitted by the AMDGPU backend
constexpr llvm::StringLiteral GlobalDtorKernelName = "amdgcn.device.fini";

/// Walk the parsed ELF and find the single kernel-function symbol, ignoring
/// the global constructor/destructor kernels (if present) since those are
/// not user-visible instrumented kernels.
///
/// \param Required demand a kernel. The first code object loaded under a key
/// carries the instrumented kernel and so must have exactly one; a code object
/// added to an existing entry may instead carry only device functions and
/// globals for the ones already loaded, and gets \c std::nullopt. More than
/// one kernel is an error either way.
llvm::Expected<std::optional<object::AMDGCNKernelFuncSymbolRef>>
findSingleKernel(const object::AMDGCNObjectFile &Obj, bool Required) {
  llvm::Error IterErr = llvm::Error::success();
  std::optional<object::AMDGCNKernelFuncSymbolRef> Found;
  unsigned KernelCount = 0;
  for (const auto &KSym : Obj.kernel_functions(IterErr)) {
    auto NameOrErr = KSym.getName();
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    if (*NameOrErr == GlobalCtorKernelName ||
        *NameOrErr == GlobalDtorKernelName)
      continue;
    ++KernelCount;
    Found = KSym;
  }
  LUTHIER_RETURN_ON_ERROR(std::move(IterErr));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      KernelCount <= 1 && (KernelCount == 1 || !Required),
      llvm::formatv("Instrumented relocatable must contain {0} kernel "
                    "function (excluding global constructor/destructor "
                    "kernels); found {1}",
                    Required ? "exactly one" : "at most one", KernelCount)));
  return Found;
}

/// \returns \c true if the kernel described by \p KD reaches its callees
/// through calls the compiler could not size a stack for, and so needs the
/// dispatch to reserve a private segment on its behalf.
///
/// The AMDGPU backend emits amdgcn.device.init/fini as a loop of *indirect*
/// calls through .init_array/.fini_array, so it marks them
/// 'uses_dynamic_stack' and leaves their statically-computed
/// private_segment_fixed_size at 0 even though the callees need a stack.
/// Dispatching with that 0 memory-faults as soon as a callee spills.
bool usesDynamicStack(const llvm::amdhsa::kernel_descriptor_t &KD) {
  return AMDHSA_BITS_GET(
             KD.kernel_code_properties,
             llvm::amdhsa::KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK) != 0;
}

/// Writes the low \p Size bytes of \p Value into \p Buf at \p Offset, little
/// endian. Errors out rather than writing out of bounds, and rather than
/// truncating a value that does not fit the slot the metadata declared.
llvm::Error writeKernargAt(llvm::MutableArrayRef<uint8_t> Buf, uint32_t Offset,
                           uint32_t Size, uint64_t Value) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Size <= sizeof(uint64_t) && Offset <= Buf.size() &&
          Size <= Buf.size() - Offset,
      llvm::formatv("Hidden kernel argument at offset {0} of size {1} does not "
                    "fit inside the {2}-byte kernarg segment",
                    Offset, Size, Buf.size())));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Size == sizeof(uint64_t) || (Value >> (Size * 8)) == 0,
      llvm::formatv("Value {0:x} of the hidden kernel argument at offset {1} "
                    "does not fit in its {2}-byte slot",
                    Value, Offset, Size)));
  for (uint32_t I = 0; I < Size; ++I)
    Buf[Offset + I] = static_cast<uint8_t>(Value >> (8 * I));
  return llvm::Error::success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Construction / destruction
//===----------------------------------------------------------------------===//

InstrumentedKernelLoaderAndLauncher::InstrumentedKernelLoaderAndLauncher(
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
    const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
    const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
        &Loader)
    : CoreApi(CoreApi), AmdExt(AmdExt), Loader(Loader) {
  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher] ctor\n");
}

InstrumentedKernelLoaderAndLauncher::~InstrumentedKernelLoaderAndLauncher() {
  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher] dtor\n");
  llvm::consumeError(unloadAll());
}

//===----------------------------------------------------------------------===//
// eraseRecordLocked
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::eraseRecordLocked(
    llvm::DenseMap<Key, CodeObjectList, KeyDenseMapInfo>::iterator It) {
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher] eraseRecordLocked KD="
             << It->first.KD << " preset=" << It->first.Preset << " ("
             << It->second.size() << " code object(s))\n");
  llvm::Error E = llvm::Error::success();
  // Reverse load order: a later code object was bound against the globals of
  // the earlier ones, and its destructor kernel may still read them, so it has
  // to go first.
  CodeObjectList &CodeObjects = It->second;
  for (auto I = CodeObjects.rbegin(), End = CodeObjects.rend(); I != End; ++I)
    E = llvm::joinErrors(std::move(E), eraseCodeObjectLocked(*I));

  ByOriginal.erase(It);
  return E;
}

llvm::Error InstrumentedKernelLoaderAndLauncher::eraseCodeObjectLocked(
    InstrumentedRecord &R) {
  llvm::Error E = llvm::Error::success();
  const auto Core = CoreApi.getTable();
  const auto AmdExtTbl = AmdExt.getTable();

  // Invoke the cached global-destructor kernel ("amdgcn.device.fini"), if
  // this record has one, while the executable is still alive.
  if (R.DtorKernel)
    E = llvm::joinErrors(std::move(E),
                         launchSingleWorkItemKernelAndWait(R, *R.DtorKernel));

  // Now that the destructor has had its chance to release whatever the
  // constructor allocated, stop listening on the hostcall buffer and free it
  // — along with the heap and any device memory the kernels leaked through
  // the device-memory service.
  if (R.HostcallBufferAlloc) {
    unregisterHostcallBuffer(*R.HostcallBufferAlloc);
    R.HostcallBufferAlloc.reset();
  }
  R.HeapBuffer.reset();

  // Executable first (releases its references into the reader's host
  // memory), then reader.
  E = llvm::joinErrors(std::move(E), hsa::executableDestroy(Core, R.Exec));
  E = llvm::joinErrors(std::move(E),
                       hsa::codeObjectReaderDestroy(R.Reader, Core));

  // This record owns its managed-variable storage; free it here.
  for (const ManagedAlloc &Alloc : R.ManagedAllocs)
    E = llvm::joinErrors(std::move(E), freeManagedStorage(AmdExtTbl, Alloc));

  return E;
}

//===----------------------------------------------------------------------===//
// unloadAll / unloadInstrumentedIfExists
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::unloadAll() {
  llvm::sys::ScopedWriter W(Mutex);
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher] unloadAll: "
             << ByOriginal.size() << " record(s)\n");
  llvm::Error E = llvm::Error::success();
  for (auto It = ByOriginal.begin(); It != ByOriginal.end();) {
    auto Curr = It++;
    E = llvm::joinErrors(std::move(E), eraseRecordLocked(Curr));
  }
  return E;
}

llvm::Error InstrumentedKernelLoaderAndLauncher::unloadInstrumentedIfExists(
    const llvm::amdhsa::kernel_descriptor_t *OriginalKD, uint64_t Preset) {
  llvm::sys::ScopedWriter W(Mutex);
  auto It = ByOriginal.find(Key{OriginalKD, Preset});
  if (It == ByOriginal.end())
    return llvm::Error::success();
  return eraseRecordLocked(It);
}

//===----------------------------------------------------------------------===//
// lookupGlobalVariable
//===----------------------------------------------------------------------===//

llvm::Expected<hsa_executable_symbol_t>
InstrumentedKernelLoaderAndLauncher::lookupGlobalVariable(
    llvm::StringRef Name, const llvm::amdhsa::kernel_descriptor_t *OriginalKD,
    uint64_t Preset) {
  llvm::sys::ScopedReader R(Mutex);
  auto It = ByOriginal.find(Key{OriginalKD, Preset});
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      It != ByOriginal.end(),
      llvm::formatv("No instrumented variant cached for kernel_descriptor "
                    "{0:x} preset {1}",
                    reinterpret_cast<uint64_t>(OriginalKD), Preset)));
  // Load order, so the code object that first defined the variable answers for
  // it. Later ones re-export it (they were handed it as an external agent
  // global variable), and every copy names the same device address.
  for (const InstrumentedRecord &Rec : It->second) {
    auto SymIt = Rec.NameToVarSymbol.find(Name);
    if (SymIt != Rec.NameToVarSymbol.end())
      return SymIt->second;
  }
  return LUTHIER_MAKE_GENERIC_ERROR(
      llvm::formatv("Global variable '{0}' not found in any of the {1} code "
                    "object(s) loaded for the requested kernel/preset.",
                    Name, It->second.size()));
}

//===----------------------------------------------------------------------===//
// overrideWithInstrumented + custom kernarg
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::overrideWithInstrumented(
    hsa_kernel_dispatch_packet_t &Packet, uint64_t Preset) {
  llvm::sys::ScopedWriter W(Mutex);
  const auto *KD = reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      Packet.kernel_object);
  auto It = ByOriginal.find(Key{KD, Preset});
  if (It == ByOriginal.end())
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "No instrumented variant cached for kernel_object {0:x} preset {1}",
        Packet.kernel_object, Preset));

  // The first code object loaded under the key carries the instrumented
  // kernel; additions contribute code and globals without changing what a
  // dispatch runs.
  const InstrumentedRecord &Rec = It->second.front();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Rec.Kernel.has_value(),
      llvm::formatv("The instrumented variant cached for kernel_object {0:x} "
                    "preset {1} carries no kernel",
                    Packet.kernel_object, Preset)));
  Packet.kernel_object = Rec.Kernel->KDDeviceAddress;
  Packet.private_segment_size =
      std::max<uint32_t>(Packet.private_segment_size,
                         Rec.Kernel->KDHostAddress->private_segment_fixed_size);

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// invalidateOriginalExec
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::invalidateOriginalExec(
    hsa_executable_t Exec) {
  llvm::sys::ScopedWriter W(Mutex);
  llvm::Error E = llvm::Error::success();

  llvm::SmallVector<hsa_loaded_code_object_t, 2> LCOs;
  if (auto Err =
          hsa::executableGetLoadedCodeObjects(Loader.getTable(), Exec, LCOs))
    return llvm::joinErrors(std::move(E), std::move(Err));

  struct Range {
    uint64_t Start;
    uint64_t End;
  };
  llvm::SmallVector<Range, 2> Ranges;
  Ranges.reserve(LCOs.size());
  for (hsa_loaded_code_object_t LCO : LCOs) {
    auto LoadedMemOrErr =
        hsa::loadedCodeObjectGetLoadedMemory(Loader.getTable(), LCO);
    if (!LoadedMemOrErr) {
      E = llvm::joinErrors(std::move(E), LoadedMemOrErr.takeError());
      continue;
    }
    const auto Start = reinterpret_cast<uint64_t>(LoadedMemOrErr->data());
    Ranges.push_back(Range{Start, Start + LoadedMemOrErr->size()});
  }

  llvm::SmallVector<Key, 4> Victims;
  for (const auto &[K, _] : ByOriginal) {
    const auto Addr = reinterpret_cast<uint64_t>(K.KD);
    for (const Range &Rng : Ranges) {
      if (Addr >= Rng.Start && Addr < Rng.End) {
        Victims.push_back(K);
        break;
      }
    }
  }

  for (const Key &K : Victims) {
    auto It = ByOriginal.find(K);
    if (It != ByOriginal.end())
      E = llvm::joinErrors(std::move(E), eraseRecordLocked(It));
  }
  return E;
}

//===----------------------------------------------------------------------===//
// Managed-variable storage allocation (HMM-aware)
//===----------------------------------------------------------------------===//

llvm::Expected<hsa_amd_memory_pool_t>
InstrumentedKernelLoaderAndLauncher::selectManagedVarPool(
    const hsa::ApiTableContainer<::AmdExtTable> &AmdExt, hsa_agent_t CpuAgent) {
  hsa_amd_memory_pool_t Found{};
  bool DidFind = false;
  LUTHIER_RETURN_ON_ERROR(hsa::agentIterateMemoryPools(
      AmdExt, CpuAgent, [&](hsa_amd_memory_pool_t Pool) -> llvm::Error {
        if (DidFind)
          return llvm::Error::success();
        llvm::Expected<bool> FGOrErr =
            hsa::memoryPoolIsFineGrained(AmdExt, Pool);
        LUTHIER_RETURN_ON_ERROR(FGOrErr.takeError());
        if (!*FGOrErr)
          return llvm::Error::success();
        llvm::Expected<bool> AllocOrErr =
            hsa::memoryPoolGetRuntimeAllocAllowed(AmdExt, Pool);
        LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());
        if (!*AllocOrErr)
          return llvm::Error::success();
        Found = Pool;
        DidFind = true;
        return llvm::Error::success();
      }));
  if (!DidFind)
    return LUTHIER_MAKE_HSA_ERROR(
        "No host fine-grain memory pool available for managed-var allocation.");
  return Found;
}

llvm::Expected<bool> InstrumentedKernelLoaderAndLauncher::getHmmSupported() {
  if (HmmSupportedCache)
    return *HmmSupportedCache;
  auto SupportedOrErr = hsa::systemIsSvmSupported(CoreApi.getTable());
  if (!SupportedOrErr)
    return SupportedOrErr.takeError();
  HmmSupportedCache = *SupportedOrErr;
  return *HmmSupportedCache;
}

llvm::Expected<InstrumentedKernelLoaderAndLauncher::ManagedAlloc>
InstrumentedKernelLoaderAndLauncher::allocateManagedStorage(
    const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
    llvm::ArrayRef<hsa_agent_t> GpuAgents, hsa_amd_memory_pool_t Pool,
    size_t Size, unsigned Align, bool HmmSupported) {
  if (Size == 0)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "allocateManagedStorage: zero-sized request.");

  if (HmmSupported) {
    const size_t PageSize = llvm::sys::Process::getPageSizeEstimate();
    if (Align > PageSize)
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Managed-var alignment ({0}) exceeds system page size ({1}); "
          "over-aligned managed vars are not modelled on the HMM path.",
          Align, PageSize));
    const size_t RoundedSize = (Size + PageSize - 1) & ~(PageSize - 1);

    llvm::Expected<void *> VAOrErr = hsa::vmemAddressReserveAlign(
        AmdExt, RoundedSize, /*Address=*/0, /*Alignment=*/PageSize,
        HSA_AMD_VMEM_ADDRESS_NO_REGISTER);
    if (!VAOrErr)
      return VAOrErr.takeError();

    llvm::SmallVector<hsa_amd_svm_attribute_pair_t, 8> Attrs;
    Attrs.reserve(GpuAgents.size());
    for (hsa_agent_t Agent : GpuAgents)
      Attrs.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, Agent.handle});

    if (!Attrs.empty()) {
      if (auto E = hsa::svmAttributesSet(AmdExt, *VAOrErr, RoundedSize, Attrs))
        return llvm::joinErrors(
            std::move(E), hsa::vmemAddressFree(AmdExt, *VAOrErr, RoundedSize));
    }

    return ManagedAlloc{*VAOrErr, RoundedSize, /*ViaSvm=*/true};
  }

  llvm::Expected<void *> AllocOrErr =
      hsa::memoryPoolAllocate(AmdExt, Pool, Size, /*Flags=*/0);
  if (!AllocOrErr)
    return AllocOrErr.takeError();

  if (!GpuAgents.empty()) {
    if (auto E = hsa::agentsAllowAccess(AmdExt, GpuAgents, *AllocOrErr))
      return llvm::joinErrors(std::move(E),
                              hsa::memoryPoolFree(AmdExt, *AllocOrErr));
  }
  return ManagedAlloc{*AllocOrErr, Size, /*ViaSvm=*/false};
}

llvm::Error InstrumentedKernelLoaderAndLauncher::freeManagedStorage(
    const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
    const ManagedAlloc &Alloc) {
  if (Alloc.Ptr == nullptr)
    return llvm::Error::success();
  if (Alloc.ViaSvm)
    return hsa::vmemAddressFree(AmdExt, Alloc.Ptr, Alloc.AllocSize);
  return hsa::memoryPoolFree(AmdExt, Alloc.Ptr);
}

//===----------------------------------------------------------------------===//
// loadManagedVarsForRecord
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::loadManagedVarsForRecord(
    const llvm::object::ObjectFile &Obj, InstrumentedRecord &Rec) {
  const auto Core = CoreApi.getTable();
  const auto AmdExtTbl = AmdExt.getTable();

  // Walk the instrumented object's symbol table for managed-variable companions
  // (symbols named "<base>.managed"). Each one is allocated host-coherent
  // storage owned by this instrumented copy.
  static constexpr llvm::StringLiteral ManagedSuffix = ".managed";

  llvm::Expected<bool> HmmOrErr = getHmmSupported();
  LUTHIER_RETURN_ON_ERROR(HmmOrErr.takeError());
  const bool HmmSupported = *HmmOrErr;

  // CPU fine-grain pool: resolved lazily on the first non-HMM allocation.
  hsa_amd_memory_pool_t Pool{};
  size_t Granule = 0;
  bool PoolResolved = false;
  auto EnsurePool = [&]() -> llvm::Error {
    if (PoolResolved)
      return llvm::Error::success();
    llvm::SmallVector<hsa_agent_t, 1> CpuAgents;
    LUTHIER_RETURN_ON_ERROR(
        hsa::getAllAgentsWithDeviceType<HSA_DEVICE_TYPE_CPU>(Core, CpuAgents));
    if (CpuAgents.empty())
      return LUTHIER_MAKE_HSA_ERROR(
          "No CPU agent available for managed-var allocation.");
    auto PoolOrErr = selectManagedVarPool(AmdExtTbl, CpuAgents.front());
    LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());
    Pool = *PoolOrErr;
    auto GranuleOrErr = hsa::memoryPoolGetRuntimeAllocGranule(AmdExtTbl, Pool);
    LUTHIER_RETURN_ON_ERROR(GranuleOrErr.takeError());
    Granule = *GranuleOrErr;
    PoolResolved = true;
    return llvm::Error::success();
  };

  const llvm::SmallVector<hsa_agent_t, 1> Agents{Rec.Agent};

  // Allocate + publish each managed variable into this record. Wrapped in a
  // lambda so any early error can free the allocations made so far before
  // returning — the record is not yet cached, so nothing else will reclaim
  // them.
  auto LoadManagedVars = [&]() -> llvm::Error {
    for (const auto &Sym : Obj.symbols()) {
      auto NameOrErr = Sym.getName();
      if (!NameOrErr) {
        llvm::consumeError(NameOrErr.takeError());
        continue;
      }
      if (!NameOrErr->ends_with(ManagedSuffix))
        continue;
      llvm::StringRef BaseName = NameOrErr->drop_back(ManagedSuffix.size());

      auto SectionOrErr = Sym.getSection();
      if (!SectionOrErr) {
        llvm::consumeError(SectionOrErr.takeError());
        continue;
      }
      if (*SectionOrErr == Obj.section_end())
        continue;
      uint64_t Size = (*SectionOrErr)->getSize();
      auto SymSize = llvm::object::ELFSymbolRef(Sym).getSize();
      if (SymSize != 0)
        Size = SymSize;
      if (Size == 0)
        continue;
      const unsigned Align = (*SectionOrErr)->getAlignment().value();

      // Initial bytes live in the .managed companion's section contents.
      auto ContentsOrErr = (*SectionOrErr)->getContents();
      llvm::StringRef InitBytes;
      if (ContentsOrErr)
        InitBytes = *ContentsOrErr;
      else
        llvm::consumeError(ContentsOrErr.takeError());

      // Allocate storage owned by this record, driven entirely by the
      // relocatable's .managed companion.
      if (!HmmSupported) {
        LUTHIER_RETURN_ON_ERROR(EnsurePool());
        if (Align > Granule)
          return LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
              "Managed variable {0} alignment ({1}) exceeds pool granule "
              "({2}); over-aligned managed vars are not modelled.",
              BaseName, Align, Granule));
      }
      auto AllocOrErr = allocateManagedStorage(AmdExtTbl, Agents, Pool, Size,
                                               Align, HmmSupported);
      LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());
      ManagedAlloc Alloc = *AllocOrErr;
      Alloc.Size = Size;
      if (!InitBytes.empty())
        std::memcpy(Alloc.Ptr, InitBytes.data(),
                    std::min<size_t>(InitBytes.size(), Size));
      const ManagedAlloc &Owned = Rec.ManagedAllocs.emplace_back(Alloc);
      LLVM_DEBUG(luthier::dbgs()
                 << "[InstrumentedKernelLoaderAndLauncher]   managed-var "
                 << BaseName << " allocated at " << Owned.Ptr << "\n");

      // Publish the buffer into THIS executable's loaded base symbol so its
      // device code dereferences the record's own storage.
      auto SymIt = Rec.NameToVarSymbol.find(BaseName);
      if (SymIt != Rec.NameToVarSymbol.end()) {
        auto AddrOrErr = hsa::executableSymbolGetAddress(Core, SymIt->second);
        LUTHIER_RETURN_ON_ERROR(AddrOrErr.takeError());
        LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            Core.callFunction<hsa_memory_copy>(
                reinterpret_cast<void *>(*AddrOrErr), &Owned.Ptr,
                sizeof(void *)),
            llvm::formatv("hsa_memory_copy failed publishing managed-var "
                          "pointer for {0}",
                          BaseName)));
      } else {
        LLVM_DEBUG(luthier::dbgs()
                   << "[InstrumentedKernelLoaderAndLauncher]   base symbol "
                   << BaseName << " not found in instrumented executable\n");
      }
    }
    return llvm::Error::success();
  };

  llvm::Error E = LoadManagedVars();
  if (E) {
    // Roll back the allocations made before the failure; the record won't be
    // cached, so eraseRecordLocked will never see them.
    for (const ManagedAlloc &Alloc : Rec.ManagedAllocs)
      E = llvm::joinErrors(std::move(E), freeManagedStorage(AmdExtTbl, Alloc));
    Rec.ManagedAllocs.clear();
  }
  return E;
}

//===----------------------------------------------------------------------===//
// defineGlobalsOfPriorCodeObjects
//===----------------------------------------------------------------------===//

llvm::Error
InstrumentedKernelLoaderAndLauncher::defineGlobalsOfPriorCodeObjects(
    llvm::ArrayRef<InstrumentedRecord> Prior, hsa_executable_t Exec,
    hsa_agent_t Agent) {
  const auto Core = CoreApi.getTable();
  // Earliest definition of a name wins: a variable handed to one code object
  // this way is re-reported by its executable's symbol iteration, so it also
  // appears in the symbol tables of every code object loaded after it, and HSA
  // refuses to define the same name twice.
  llvm::StringSet<> Defined;
  for (const InstrumentedRecord &Rec : Prior) {
    for (const auto &NameAndSym : Rec.NameToVarSymbol) {
      llvm::StringRef Name = NameAndSym.getKey();
      if (!Defined.insert(Name).second)
        continue;
      auto AddrOrErr =
          hsa::executableSymbolGetAddress(Core, NameAndSym.getValue());
      LUTHIER_RETURN_ON_ERROR(AddrOrErr.takeError());
      LUTHIER_RETURN_ON_ERROR(hsa::executableDefineExternalAgentGlobalVariable(
          Core, Exec, Agent, Name, reinterpret_cast<const void *>(*AddrOrErr)));
      LLVM_DEBUG(luthier::dbgs()
                 << "[InstrumentedKernelLoaderAndLauncher] bound '" << Name
                 << "' at " << llvm::format_hex(*AddrOrErr, 18)
                 << " into executable " << Exec.handle << "\n");
    }
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// loadInstrumented
//===----------------------------------------------------------------------===//

llvm::Expected<hsa_executable_symbol_t>
InstrumentedKernelLoaderAndLauncher::loadInstrumented(
    std::unique_ptr<llvm::MemoryBuffer> Relocatable,
    const llvm::amdhsa::kernel_descriptor_t *OriginalKD, uint64_t Preset) {
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher] loadInstrumented KD="
             << OriginalKD << " preset=" << Preset << "\n");
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Relocatable != nullptr,
      "Null relocatable MemoryBuffer passed to loadInstrumented"));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      OriginalKD != nullptr,
      "Null kernel-descriptor pointer passed to loadInstrumented"));

  const auto Core = CoreApi.getTable();

  // Resolve the agent that owns the kernel-descriptor allocation via
  // hsa_amd_pointer_info (works for loader-published and pool allocations).
  auto KDAddr = reinterpret_cast<uint64_t>(OriginalKD);
  hsa_amd_pointer_info_t PointerInfo{};
  PointerInfo.size = sizeof(hsa_amd_pointer_info_t);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExt.getTable().callFunction<hsa_amd_pointer_info>(
          const_cast<void *>(reinterpret_cast<const void *>(OriginalKD)),
          &PointerInfo, /*alloc=*/nullptr, /*num_agents_accessible=*/nullptr,
          /*accessible=*/nullptr),
      llvm::formatv("Failed to query HSA pointer info for kernel "
                    "descriptor at {0:x}",
                    KDAddr)));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      PointerInfo.type != HSA_EXT_POINTER_TYPE_UNKNOWN,
      llvm::formatv("Kernel descriptor at {0:x} is not owned by any HSA "
                    "allocation",
                    KDAddr)));
  hsa_agent_t Agent = PointerInfo.agentOwner;

  llvm::sys::ScopedWriter W(Mutex);

  // A key that already has code objects is not an error: this one joins them
  // and is bound against what they define. The map is not mutated between here
  // and the append below, so this iterator stays valid throughout.
  const auto EntryIt = ByOriginal.find(Key{OriginalKD, Preset});
  const bool IsAdditional = EntryIt != ByOriginal.end();
  if (IsAdditional) {
    // Every code object under a key is bound to the addresses the others were
    // loaded at, which only means anything on the agent that loaded them.
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        EntryIt->second.front().Agent.handle == Agent.handle,
        llvm::formatv("Kernel descriptor {0:x} preset {1} already has code "
                      "objects loaded on agent {2:x}; cannot add one for "
                      "agent {3:x}",
                      KDAddr, Preset, EntryIt->second.front().Agent.handle,
                      Agent.handle)));
  }

  // The instrumented bytes come out of NewPMAsmPrinter as a REL; link to a
  // shared object so we get a proper .dynsym + PT_DYNAMIC layout.
  llvm::SmallVector<char, 0> LinkedBuf;
  LUTHIER_RETURN_ON_ERROR(linker::linkRelocatableToExecutable(
      llvm::ArrayRef<char>(Relocatable->getBufferStart(),
                           Relocatable->getBufferSize()),
      LinkedBuf));
  auto Linked = std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(LinkedBuf), "luthier.instrumented.linked",
      /*RequiresNullTerminator=*/false);
  Relocatable = std::move(Linked);

  llvm::MemoryBufferRef RelocRef = Relocatable->getMemBufferRef();
  auto ParsedOrErr = object::AMDGCNObjectFile::createAMDGCNObjectFile(RelocRef);
  LUTHIER_RETURN_ON_ERROR(ParsedOrErr.takeError());
  std::unique_ptr<object::AMDGCNObjectFile> Parsed = std::move(*ParsedOrErr);

  // Only the first code object of an entry has to carry the instrumented
  // kernel; an addition may be nothing but device functions and globals.
  auto KernelSymOrErr = findSingleKernel(*Parsed, /*Required=*/!IsAdditional);
  LUTHIER_RETURN_ON_ERROR(KernelSymOrErr.takeError());
  std::string KernelName;
  std::string KDName;
  if (KernelSymOrErr->has_value()) {
    auto KernelNameOrErr = (*KernelSymOrErr)->getName();
    LUTHIER_RETURN_ON_ERROR(KernelNameOrErr.takeError());
    KernelName = std::string(*KernelNameOrErr);
    KDName = KernelName + ".kd";
  }

  // Stand up the HSA executable. A first code object is self-contained (its
  // device globals are defined in its own copy); an addition resolves its
  // undefined globals against the ones already loaded under this key.
  auto ExecOrErr = hsa::executableCreate(Core);
  LUTHIER_RETURN_ON_ERROR(ExecOrErr.takeError());
  hsa_executable_t Exec = *ExecOrErr;

  // Has to happen before the code object is loaded: the loader binds undefined
  // references as it loads, so a definition added afterwards comes too late.
  if (IsAdditional) {
    if (auto Err =
            defineGlobalsOfPriorCodeObjects(EntryIt->second, Exec, Agent))
      return llvm::joinErrors(std::move(Err),
                              hsa::executableDestroy(Core, Exec));
  }

  auto ReaderOrErr =
      hsa::codeObjectReaderCreateFromMemory(Core, RelocRef.getBuffer());
  if (!ReaderOrErr)
    return llvm::joinErrors(ReaderOrErr.takeError(),
                            hsa::executableDestroy(Core, Exec));
  hsa_code_object_reader_t Reader = *ReaderOrErr;

  auto Fail = [&](llvm::Error E) -> llvm::Error {
    return llvm::joinErrors(
        llvm::joinErrors(std::move(E), hsa::executableDestroy(Core, Exec)),
        hsa::codeObjectReaderDestroy(Reader, Core));
  };

  if (auto Err = hsa::executableLoadAgentCodeObject(Core, Exec, Reader, Agent)
                     .takeError())
    return Fail(std::move(Err));

  if (auto Err = hsa::executableFreeze(Core, Exec))
    return Fail(std::move(Err));

  // Parse the code object's metadata once: every kernel's hidden-argument
  // layout and the constant printf format strings all come out of it.
  auto MDDocOrErr = Parsed->getMetadataDocument();
  if (!MDDocOrErr)
    return Fail(MDDocOrErr.takeError());
  llvm::msgpack::Document &MetadataDoc = **MDDocOrErr;

  auto NoteMDOrErr =
      amdgpu::hsamd::MetadataParser().parseNoteMetaData(MetadataDoc);
  if (!NoteMDOrErr)
    return Fail(NoteMDOrErr.takeError());
  PrintfFormatStringMap PrintfFormatStrings;
  if ((*NoteMDOrErr)->Printf) {
    auto FormatsOrErr = parsePrintfFormatStrings(*(*NoteMDOrErr)->Printf);
    if (!FormatsOrErr)
      return Fail(FormatsOrErr.takeError());
    PrintfFormatStrings = std::move(*FormatsOrErr);
  }

  // Everything the dispatch needs about the instrumented kernel comes off its
  // kernel descriptor in the host code object. A code object that carries no
  // kernel hands back a zero-handle symbol.
  std::optional<LoadedKernelInfo> InstrKernel;
  hsa_executable_symbol_t InstrSym{};
  if (!KernelName.empty()) {
    auto InstrKernelOrErr =
        findKernelIfPresent(*Parsed, MetadataDoc, Exec, Agent, KernelName);
    if (!InstrKernelOrErr)
      return Fail(InstrKernelOrErr.takeError());
    if (!InstrKernelOrErr->has_value())
      return Fail(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "The instrumented code object defines kernel function '{0}' but no "
          "matching kernel descriptor '{1}'",
          KernelName, KDName)));
    InstrKernel = std::move(**InstrKernelOrErr);
    InstrSym = InstrKernel->Symbol;
  }

  // Detect a global-destructor kernel ("amdgcn.device.fini"), if the
  // 'amdgpu-lower-ctor-dtor' backend pass emitted one for this instrumented
  // copy's dynamically-initialized __device__ globals. It is cached on the
  // record and dispatched by eraseRecordLocked right before the executable is
  // torn down.
  auto DtorKernelOrErr = findKernelIfPresent(*Parsed, MetadataDoc, Exec, Agent,
                                             GlobalDtorKernelName);
  if (!DtorKernelOrErr)
    return Fail(DtorKernelOrErr.takeError());

  // The global-constructor kernel, dispatched further down once the managed
  // variables it may reference have been published.
  auto CtorKernelOrErr = findKernelIfPresent(*Parsed, MetadataDoc, Exec, Agent,
                                             GlobalCtorKernelName);
  if (!CtorKernelOrErr)
    return Fail(CtorKernelOrErr.takeError());

  InstrumentedRecord Rec;
  Rec.RelocatableBuffer = std::move(Relocatable);
  Rec.Reader = Reader;
  Rec.Exec = Exec;
  Rec.Kernel = std::move(InstrKernel);
  Rec.Agent = Agent;
  Rec.DtorKernel = std::move(*DtorKernelOrErr);
  Rec.PrintfFormatStrings = std::move(PrintfFormatStrings);

  // Stand up the record-scoped buffers if either of the constructor and
  // destructor kernels can reach them. They belong to the record rather than
  // to a single dispatch: memory the constructor allocates has to survive
  // until the destructor releases it, and both the hostcall service and the
  // heap track those allocations.
  auto EitherCtorDtorDeclares =
      [&](amdgpu::hsamd::ValueKind Kind) {
        return (CtorKernelOrErr->has_value() &&
                declaresHiddenArg(**CtorKernelOrErr, Kind)) ||
               (Rec.DtorKernel && declaresHiddenArg(*Rec.DtorKernel, Kind));
      };

  if (EitherCtorDtorDeclares(
          amdgpu::hsamd::ValueKind::HiddenHostcallBuffer)) {
    auto HostcallBufferOrErr = createAndRegisterHostcallBuffer(Agent);
    if (!HostcallBufferOrErr)
      return Fail(HostcallBufferOrErr.takeError());
    Rec.HostcallBufferAlloc = std::move(*HostcallBufferOrErr);
  }

  // From here on the record owns resources that Fail() does not know how to
  // release, so unwind them explicitly before handing the error back.
  auto FailWithRecord = [&](llvm::Error E) -> llvm::Error {
    if (Rec.HostcallBufferAlloc) {
      unregisterHostcallBuffer(*Rec.HostcallBufferAlloc);
      Rec.HostcallBufferAlloc.reset();
    }
    Rec.HeapBuffer.reset();
    return Fail(std::move(E));
  };

  if (EitherCtorDtorDeclares(amdgpu::hsamd::ValueKind::HiddenHeapV1)) {
    auto HeapOrErr = DeviceHeapBuffer::create(AmdExt.getTable(), Agent);
    if (!HeapOrErr)
      return FailWithRecord(HeapOrErr.takeError());
    Rec.HeapBuffer = std::move(*HeapOrErr);
    // The heap sources every slab through the device-memory hostcall, so a
    // kernel handed a heap but no hostcall buffer would spin on its first
    // allocation. The compiler emits both arguments together, so this only
    // trips on a hand-written or rewritten code object.
    if (!Rec.HostcallBufferAlloc)
      return FailWithRecord(LUTHIER_MAKE_GENERIC_ERROR(
          "The instrumented code object declares a hidden_heap_v1 argument "
          "but no hidden_hostcall_buffer; device-side malloc obtains its "
          "memory through the hostcall device-memory service and cannot work "
          "without one"));
  }

  // Harvest this executable's device-global variable symbols for host readback.
  auto HarvestCb = [&](hsa_executable_symbol_t Sym) -> llvm::Error {
    auto KindOrErr = hsa::executableSymbolGetType(Core, Sym);
    LUTHIER_RETURN_ON_ERROR(KindOrErr.takeError());
    if (*KindOrErr != HSA_SYMBOL_KIND_VARIABLE)
      return llvm::Error::success();
    auto NameOrErr = hsa::executableSymbolGetName(Core, Sym);
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    Rec.NameToVarSymbol[*NameOrErr] = Sym;
    return llvm::Error::success();
  };
  if (auto Err =
          hsa::executableIterateAgentSymbols(Core, Exec, Agent, HarvestCb))
    return FailWithRecord(std::move(Err));

  // Allocate + publish the managed variables this instrumented copy carries,
  // driven by its own relocatable. The allocations are owned by Rec and freed
  // when the record is erased; loadManagedVarsForRecord frees them itself if it
  // fails here (Rec is not yet cached).
  if (auto Err = loadManagedVarsForRecord(*Parsed, Rec))
    return FailWithRecord(std::move(Err));

  // Invoke the global-constructor kernel ("amdgcn.device.init"), if the
  // relocatable carried one, now that the managed variables it may
  // reference have been published. Runs once, synchronously.
  if (CtorKernelOrErr->has_value()) {
    if (auto Err =
            launchSingleWorkItemKernelAndWait(Rec, **CtorKernelOrErr))
      return FailWithRecord(std::move(Err));
  }

  if (IsAdditional) {
    EntryIt->second.push_back(std::move(Rec));
  } else {
    CodeObjectList CodeObjects;
    CodeObjects.push_back(std::move(Rec));
    auto [It, Inserted] =
        ByOriginal.try_emplace(Key{OriginalKD, Preset}, std::move(CodeObjects));
    assert(Inserted && "Concurrent insert into ByOriginal under writer lock");
    (void)Inserted;
  }

  return InstrSym;
}

//===----------------------------------------------------------------------===//
// findKernelIfPresent
//===----------------------------------------------------------------------===//

llvm::Expected<
    std::optional<InstrumentedKernelLoaderAndLauncher::LoadedKernelInfo>>
InstrumentedKernelLoaderAndLauncher::findKernelIfPresent(
    const object::AMDGCNObjectFile &Obj, llvm::msgpack::Document &MetadataDoc,
    hsa_executable_t Exec, hsa_agent_t Agent, llvm::StringRef KernelName) {
  const auto Core = CoreApi.getTable();
  const std::string KDName = (KernelName + ".kd").str();

  // The kernel descriptor is a 64-byte object the compiler emitted into the
  // code object itself, so read it straight off the host copy of the ELF
  // instead of re-querying HSA one segment size at a time.
  auto KDSymOrErr = Obj.lookupSymbol(KDName);
  LUTHIER_RETURN_ON_ERROR(KDSymOrErr.takeError());
  if (!KDSymOrErr->has_value())
    return std::nullopt;

  auto KDBytesOrErr = object::getContents(**KDSymOrErr);
  LUTHIER_RETURN_ON_ERROR(KDBytesOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      KDBytesOrErr->size() >= sizeof(llvm::amdhsa::kernel_descriptor_t),
      llvm::formatv("Kernel descriptor '{0}' spans {1} bytes in the "
                    "instrumented code object; expected at least {2}",
                    KDName, KDBytesOrErr->size(),
                    sizeof(llvm::amdhsa::kernel_descriptor_t))));

  LoadedKernelInfo Info;
  Info.KDHostAddress =
      reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
          KDBytesOrErr->data());

  // The loader publishes the same descriptor on the device; that address is
  // what an AQL packet's kernel_object takes.
  auto DeviceSymOrErr =
      hsa::executableGetSymbolByName(Core, Exec, KDName, Agent);
  LUTHIER_RETURN_ON_ERROR(DeviceSymOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      DeviceSymOrErr->has_value(),
      llvm::formatv("The instrumented executable does not expose a kernel "
                    "descriptor named '{0}', even though its code object "
                    "defines one",
                    KDName)));
  Info.Symbol = **DeviceSymOrErr;
  auto KDDeviceAddrOrErr = hsa::executableSymbolGetAddress(Core, Info.Symbol);
  LUTHIER_RETURN_ON_ERROR(KDDeviceAddrOrErr.takeError());
  Info.KDDeviceAddress = *KDDeviceAddrOrErr;

  // Harvest the hidden arguments' offsets and widths from the code object's
  // metadata; they are the only description of the hidden block's layout that
  // is guaranteed to match the compiler that produced this object.
  auto KernelMDOrErr =
      amdgpu::hsamd::MetadataParser().parseKernelMetadata(MetadataDoc, KDName);
  LUTHIER_RETURN_ON_ERROR(KernelMDOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      (*KernelMDOrErr)->Symbol == KDName,
      llvm::formatv("The instrumented code object defines kernel descriptor "
                    "'{0}' but has no amdhsa.kernels metadata entry for it",
                    KDName)));
  if ((*KernelMDOrErr)->Args) {
    for (const amdgpu::hsamd::Kernel::Arg::Metadata &Arg :
         *(*KernelMDOrErr)->Args) {
      if (Arg.ValKind < amdgpu::hsamd::ValueKind::HiddenArgKindBegin ||
          Arg.ValKind > amdgpu::hsamd::ValueKind::HiddenArgKindEnd)
        continue;
      Info.HiddenArgs.push_back(
          HiddenArgInfo{Arg.ValKind, Arg.Offset, Arg.Size});
    }
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher]   found kernel "
             << llvm::formatv("{0}: KD at {1:x} on the device, {2} on the "
                              "host, {3} hidden arg(s)\n",
                              KernelName, Info.KDDeviceAddress,
                              static_cast<const void *>(Info.KDHostAddress),
                              Info.HiddenArgs.size()));
  return Info;
}

//===----------------------------------------------------------------------===//
// getMaxPrivateSegmentSize
//===----------------------------------------------------------------------===//

llvm::Expected<uint32_t>
InstrumentedKernelLoaderAndLauncher::getMaxPrivateSegmentSize(
    const hsa::ApiTableContainer<::CoreApiTable> &CoreApi, hsa_agent_t Agent) {
  /// Bytes per COMPUTE_TMPRING_SIZE.WAVESIZE granule.
  constexpr uint64_t ScratchGranuleBytes = 256;
  /// Width of the COMPUTE_TMPRING_SIZE.WAVESIZE field, before and from gfx12.
  constexpr unsigned WaveSizeFieldBits = 15;
  constexpr unsigned WaveSizeFieldBitsGFX12 = 18;
  /// Bytes at the base of every work-item's private segment that the AMDGPU
  /// backend reserves for itself and that therefore cannot go to the stack.
  constexpr uint64_t CompilerReservedBytes = 64;

  llvm::SmallVector<hsa_isa_t, 1> ISAs;
  LUTHIER_RETURN_ON_ERROR(hsa::agentGetSupportedISAs(CoreApi, Agent, ISAs));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !ISAs.empty(),
      llvm::formatv("Agent {0:x} does not support any ISA, so the maximum "
                    "private segment size it accepts cannot be derived",
                    Agent.handle)));
  llvm::Expected<std::string> GPUNameOrErr =
      hsa::isaGetGPUName(CoreApi, ISAs.front());
  LUTHIER_RETURN_ON_ERROR(GPUNameOrErr.takeError());

  llvm::Expected<uint32_t> WavefrontSizeOrErr =
      hsa::agentGetWavefrontSize(CoreApi, Agent);
  LUTHIER_RETURN_ON_ERROR(WavefrontSizeOrErr.takeError());

  // An unrecognized processor parses as major version 0 and takes the
  // narrower, always-valid field width.
  const unsigned FieldBits = llvm::AMDGPU::getIsaVersion(*GPUNameOrErr).Major >=
                                     12
                                 ? WaveSizeFieldBitsGFX12
                                 : WaveSizeFieldBits;
  const uint64_t MaxPerWave =
      ((uint64_t{1} << FieldBits) - 1) * ScratchGranuleBytes;
  const uint64_t MaxPerWorkItem = MaxPerWave / *WavefrontSizeOrErr;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      MaxPerWorkItem > CompilerReservedBytes,
      llvm::formatv("Agent {0:x} ({1}, wavefront size {2}) cannot fit a "
                    "private segment larger than the {3} bytes the compiler "
                    "reserves",
                    Agent.handle, *GPUNameOrErr, *WavefrontSizeOrErr,
                    CompilerReservedBytes)));
  return static_cast<uint32_t>(MaxPerWorkItem - CompilerReservedBytes);
}

//===----------------------------------------------------------------------===//
// writeHiddenKernelArguments
//===----------------------------------------------------------------------===//

bool InstrumentedKernelLoaderAndLauncher::declaresHiddenArg(
    const LoadedKernelInfo &Kernel, amdgpu::hsamd::ValueKind Kind) {
  return llvm::any_of(Kernel.HiddenArgs, [Kind](const HiddenArgInfo &Arg) {
    return Arg.Kind == Kind;
  });
}

llvm::Error InstrumentedKernelLoaderAndLauncher::writeHiddenKernelArguments(
    llvm::MutableArrayRef<uint8_t> Kernarg,
    llvm::ArrayRef<HiddenArgInfo> HiddenArgs,
    const hsa_kernel_dispatch_packet_t &Packet, const hsa_queue_t &Queue,
    const HiddenArgBufferAddresses &Buffers) {
  using amdgpu::hsamd::ValueKind;

  const uint16_t GroupSize[3]{Packet.workgroup_size_x, Packet.workgroup_size_y,
                              Packet.workgroup_size_z};
  const uint32_t GridSize[3]{Packet.grid_size_x, Packet.grid_size_y,
                             Packet.grid_size_z};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      GroupSize[0] != 0 && GroupSize[1] != 0 && GroupSize[2] != 0,
      "Dispatch packet declares a zero-sized workgroup dimension"));
  const uint16_t GridDims =
      (Packet.setup >> HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) &
      ((1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS) - 1);

  // hidden_private_base and hidden_shared_base are the high halves of the
  // scratch and LDS flat apertures. ROCr publishes them in the AMD extension
  // of the queue struct, whose first member is the hsa_queue_t the core API
  // hands back — the same reinterpretation ROCclr performs to fill these two
  // arguments. Both fields sit at the same offsets in amd_queue_t and
  // amd_queue_v2_t.
  const auto &AmdQueue = reinterpret_cast<const amd_queue_t &>(Queue);

  /// Index of \p Kind within the X/Y/Z triple that starts at \p First.
  auto DimensionOf = [](ValueKind Kind, ValueKind First) -> unsigned {
    return static_cast<unsigned>(Kind) - static_cast<unsigned>(First);
  };

  for (const HiddenArgInfo &Arg : HiddenArgs) {
    uint64_t Value = 0;
    switch (Arg.Kind) {
    // Number of workgroups in each dimension.
    case ValueKind::HiddenBlockCountX:
    case ValueKind::HiddenBlockCountY:
    case ValueKind::HiddenBlockCountZ: {
      const unsigned D = DimensionOf(Arg.Kind, ValueKind::HiddenBlockCountX);
      Value = GridSize[D] / GroupSize[D];
      break;
    }
    case ValueKind::HiddenGroupSizeX:
    case ValueKind::HiddenGroupSizeY:
    case ValueKind::HiddenGroupSizeZ:
      Value = GroupSize[DimensionOf(Arg.Kind, ValueKind::HiddenGroupSizeX)];
      break;
    // Size of the trailing partial workgroup, 0 when the grid divides evenly.
    case ValueKind::HiddenRemainderX:
    case ValueKind::HiddenRemainderY:
    case ValueKind::HiddenRemainderZ: {
      const unsigned D = DimensionOf(Arg.Kind, ValueKind::HiddenRemainderX);
      Value = GridSize[D] % GroupSize[D];
      break;
    }
    case ValueKind::HiddenGridDims:
      Value = GridDims;
      break;
    // These dispatches always start at the grid origin.
    case ValueKind::HiddenGlobalOffsetX:
    case ValueKind::HiddenGlobalOffsetY:
    case ValueKind::HiddenGlobalOffsetZ:
      Value = 0;
      break;
    case ValueKind::HiddenPrivateBase:
      Value = AmdQueue.private_segment_aperture_base_hi;
      break;
    case ValueKind::HiddenSharedBase:
      Value = AmdQueue.group_segment_aperture_base_hi;
      break;
    case ValueKind::HiddenQueuePtr:
      Value = reinterpret_cast<uintptr_t>(&Queue);
      break;
    // The dispatch requests no group segment beyond the kernel's fixed size.
    case ValueKind::HiddenDynamicLDSSize:
      Value = 0;
      break;
    // A kernel that makes a hostcall spins until the host answers it, so
    // handing it a null buffer would hang the dispatch outright. The caller
    // only leaves this null when the kernel declared the argument but no
    // listener could be stood up; the kernel then fails the same way it would
    // under HIP without hostcall support.
    case ValueKind::HiddenHostcallBuffer:
      Value = reinterpret_cast<uintptr_t>(Buffers.HostcallBuffer);
      break;
    // Buffered printf bump-allocates its records out of this buffer and
    // checks it for null first, so a null here degrades to dropped output
    // rather than a fault.
    case ValueKind::HiddenPrintfBuffer:
      Value = reinterpret_cast<uintptr_t>(Buffers.PrintfBuffer);
      break;
    // Device-side malloc keeps its slab bookkeeping here.
    case ValueKind::HiddenHeapV1:
      Value = reinterpret_cast<uintptr_t>(Buffers.Heap);
      break;
    // Cooperative-groups barrier state for this grid.
    case ValueKind::HiddenMultiGridSyncArg:
      Value = reinterpret_cast<uintptr_t>(Buffers.GridSyncInfo);
      break;
    // The already-completed wrapper standing in for the parent of a kernel
    // the host launched directly.
    case ValueKind::HiddenCompletionAction:
      Value = reinterpret_cast<uintptr_t>(Buffers.CompletionAction);
      break;
    // Padding, plus the one argument whose correct value here really is a
    // null pointer. A device-enqueue queue is only usable alongside a host
    // scheduler that drains it and runs the children pushed onto it; Luthier
    // runs none, and handing over a queue nobody services would turn a clean
    // enqueue_kernel failure into child kernels that silently never run.
    case ValueKind::HiddenNone:
    case ValueKind::HiddenDefaultQueue:
      continue;
    default:
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Unhandled hidden kernel argument kind {0} at kernarg offset {1}",
          static_cast<unsigned>(Arg.Kind), Arg.Offset));
    }
    LUTHIER_RETURN_ON_ERROR(
        writeKernargAt(Kernarg, Arg.Offset, Arg.Size, Value));
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Hostcall plumbing
//===----------------------------------------------------------------------===//

llvm::Expected<HostcallListener *>
InstrumentedKernelLoaderAndLauncher::getOrCreateHostcallListener() {
  if (!Listener) {
    auto ListenerOrErr = HostcallListener::create(CoreApi.getTable());
    LUTHIER_RETURN_ON_ERROR(ListenerOrErr.takeError());
    Listener = std::move(*ListenerOrErr);
  }
  return Listener.get();
}

llvm::Expected<std::unique_ptr<HostcallBufferAllocation>>
InstrumentedKernelLoaderAndLauncher::createAndRegisterHostcallBuffer(
    const hsa_agent_t Agent) {
  auto ListenerOrErr = getOrCreateHostcallListener();
  LUTHIER_RETURN_ON_ERROR(ListenerOrErr.takeError());

  // These kernels are dispatched over a single work-item, so one wave can
  // ever have a request outstanding. Sizing the buffer for the agent's peak
  // occupancy the way HIP does would cost tens of megabytes for nothing.
  auto BufferOrErr = HostcallBufferAllocation::create(
      CoreApi.getTable(), AmdExt.getTable(), Agent, /*NumWaves=*/1);
  LUTHIER_RETURN_ON_ERROR(BufferOrErr.takeError());

  (*ListenerOrErr)->addBuffer((*BufferOrErr)->getBuffer());
  return std::move(*BufferOrErr);
}

void InstrumentedKernelLoaderAndLauncher::unregisterHostcallBuffer(
    HostcallBufferAllocation &Buffer) {
  if (Listener)
    Listener->removeBuffer(Buffer.getBuffer());
}

//===----------------------------------------------------------------------===//
// launchSingleWorkItemKernelAndWait
//===----------------------------------------------------------------------===//

llvm::Error
InstrumentedKernelLoaderAndLauncher::launchSingleWorkItemKernelAndWait(
    const InstrumentedRecord &Rec, const LoadedKernelInfo &Kernel) {
  const auto Core = CoreApi.getTable();
  const auto AmdExtTbl = AmdExt.getTable();
  const hsa_agent_t Agent = Rec.Agent;

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Kernel.KDHostAddress != nullptr,
      "Kernel to dispatch carries no host kernel descriptor"));
  const llvm::amdhsa::kernel_descriptor_t &KD = *Kernel.KDHostAddress;

  // A kernel that reaches its callees indirectly reports a
  // private_segment_fixed_size of 0 no matter how much stack those callees
  // need; nothing can recover the real figure, so reserve the most the agent
  // will take. The dispatch covers a single work-item, so this is a one-wave
  // scratch allocation rather than a device-wide one.
  uint32_t PrivateSegmentSize = KD.private_segment_fixed_size;
  if (usesDynamicStack(KD)) {
    llvm::Expected<uint32_t> MaxPrivateSegmentSizeOrErr =
        getMaxPrivateSegmentSize(Core, Agent);
    LUTHIER_RETURN_ON_ERROR(MaxPrivateSegmentSizeOrErr.takeError());
    PrivateSegmentSize =
        std::max(PrivateSegmentSize, *MaxPrivateSegmentSizeOrErr);
  }

  auto QueueSizeOrErr = hsa::agentGetQueueMinSize(Core, Agent);
  LUTHIER_RETURN_ON_ERROR(QueueSizeOrErr.takeError());

  auto QueueOrErr = hsa::queueCreate(Core, Agent, *QueueSizeOrErr);
  LUTHIER_RETURN_ON_ERROR(QueueOrErr.takeError());
  hsa_queue_t *Queue = *QueueOrErr;

  auto SignalOrErr = hsa::signalCreate(Core, 1);
  if (!SignalOrErr)
    return llvm::joinErrors(SignalOrErr.takeError(),
                            hsa::queueDestroy(Core, Queue));
  hsa_signal_t Signal = *SignalOrErr;

  auto Cleanup = [&](llvm::Error E) -> llvm::Error {
    return llvm::joinErrors(
        llvm::joinErrors(std::move(E), hsa::signalDestroy(Core, Signal)),
        hsa::queueDestroy(Core, Queue));
  };

  // Build the packet up front so the hidden arguments can be derived from the
  // very geometry that is about to be dispatched.
  hsa_kernel_dispatch_packet_t Dispatch{};
  Dispatch.setup = 1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Dispatch.workgroup_size_x = 1;
  Dispatch.workgroup_size_y = 1;
  Dispatch.workgroup_size_z = 1;
  Dispatch.grid_size_x = 1;
  Dispatch.grid_size_y = 1;
  Dispatch.grid_size_z = 1;
  Dispatch.kernel_object = Kernel.KDDeviceAddress;
  Dispatch.group_segment_size = KD.group_segment_fixed_size;
  Dispatch.private_segment_size = PrivateSegmentSize;
  Dispatch.completion_signal = Signal;

  // Back the kernarg segment with real memory: even an argument-less ctor/dtor
  // declares the COV5 hidden arguments, so a null kernarg_address faults if
  // the kernel touches any of them.
  void *KernargPtr = nullptr;
  if (KD.kernarg_size > 0) {
    auto KernargRegionOrErr = hsa::agentFindKernargRegion(Core, Agent);
    if (!KernargRegionOrErr)
      return Cleanup(KernargRegionOrErr.takeError());
    if (!KernargRegionOrErr->has_value())
      return Cleanup(LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
          "No kernarg region on agent {0:x} to back the {1}-byte kernarg "
          "segment of the constructor/destructor kernel at {2:x}",
          Agent.handle, KD.kernarg_size, Kernel.KDDeviceAddress)));
    auto KernargOrErr =
        hsa::memoryAllocate(Core, **KernargRegionOrErr, KD.kernarg_size);
    if (!KernargOrErr)
      return Cleanup(KernargOrErr.takeError());
    KernargPtr = *KernargOrErr;
    std::memset(KernargPtr, 0, KD.kernarg_size);
    Dispatch.kernarg_address = KernargPtr;
  }

  // Everything below is host-visible memory the device pokes at while the
  // dispatch runs, so it comes from a host fine-grained pool the agent is
  // granted access to. These live only as long as the dispatch; the buffers
  // whose contents have to outlast it belong to the record instead.
  llvm::SmallVector<void *, 3> DispatchAllocs;
  auto CleanupDispatchBuffers = [&](llvm::Error E) -> llvm::Error {
    for (void *Ptr : DispatchAllocs)
      E = llvm::joinErrors(std::move(E), hsa::memoryPoolFree(AmdExtTbl, Ptr));
    DispatchAllocs.clear();
    return Cleanup(std::move(E));
  };

  hsa_amd_memory_pool_t HostPool{};
  bool HostPoolResolved = false;
  auto AllocateDispatchBuffer = [&](size_t Size) -> llvm::Expected<void *> {
    if (!HostPoolResolved) {
      llvm::SmallVector<hsa_agent_t, 1> CpuAgents;
      LUTHIER_RETURN_ON_ERROR(
          hsa::getAllAgentsWithDeviceType<HSA_DEVICE_TYPE_CPU>(Core, CpuAgents));
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          !CpuAgents.empty(),
          "No CPU agent available to back a dispatch's hidden arguments."));
      auto PoolOrErr = selectManagedVarPool(AmdExtTbl, CpuAgents.front());
      LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());
      HostPool = *PoolOrErr;
      HostPoolResolved = true;
    }
    auto AllocOrErr = hsa::memoryPoolAllocate(AmdExtTbl, HostPool, Size);
    LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());
    DispatchAllocs.push_back(*AllocOrErr);

    const llvm::SmallVector<hsa_agent_t, 1> Agents{Agent};
    LUTHIER_RETURN_ON_ERROR(
        hsa::agentsAllowAccess(AmdExtTbl, Agents, *AllocOrErr));
    return *AllocOrErr;
  };

  HiddenArgBufferAddresses Buffers;
  Buffers.HostcallBuffer =
      Rec.HostcallBufferAlloc ? Rec.HostcallBufferAlloc->getDeviceVisibleAddress()
                              : nullptr;
  Buffers.Heap =
      Rec.HeapBuffer ? Rec.HeapBuffer->getDeviceVisibleAddress() : nullptr;

  // A buffered-printf kernel bump-allocates its records out of a buffer the
  // host reads back once the dispatch is done.
  if (declaresHiddenArg(Kernel, amdgpu::hsamd::ValueKind::HiddenPrintfBuffer)) {
    auto PtrOrErr = AllocateDispatchBuffer(DefaultPrintfBufferSize);
    if (!PtrOrErr)
      return CleanupDispatchBuffers(PtrOrErr.takeError());
    if (auto Err = initializePrintfBuffer(llvm::MutableArrayRef<uint8_t>(
            static_cast<uint8_t *>(*PtrOrErr), DefaultPrintfBufferSize)))
      return CleanupDispatchBuffers(std::move(Err));
    Buffers.PrintfBuffer = *PtrOrErr;
  }

  // Cooperative groups synchronize through this. The dispatch is one
  // workgroup on one device, so a this_grid() barrier over it is trivially
  // satisfiable and this_multi_grid() has nothing to join.
  if (declaresHiddenArg(Kernel,
                        amdgpu::hsamd::ValueKind::HiddenMultiGridSyncArg)) {
    auto PtrOrErr = AllocateDispatchBuffer(sizeof(DeviceGridSyncInfo));
    if (!PtrOrErr)
      return CleanupDispatchBuffers(PtrOrErr.takeError());
    initializeSingleGridSyncInfo(
        *static_cast<DeviceGridSyncInfo *>(*PtrOrErr), /*NumWorkgroups=*/1);
    Buffers.GridSyncInfo = *PtrOrErr;
  }

  // The wrapper a device-enqueued child would report completion against.
  // Nothing enqueued this kernel, so it stands above it already finished.
  if (declaresHiddenArg(Kernel,
                        amdgpu::hsamd::ValueKind::HiddenCompletionAction)) {
    auto PtrOrErr = AllocateDispatchBuffer(sizeof(DeviceAqlWrap));
    if (!PtrOrErr)
      return CleanupDispatchBuffers(PtrOrErr.takeError());
    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            reinterpret_cast<uintptr_t>(*PtrOrErr) % DeviceAqlWrapAlignment == 0,
            llvm::formatv("A device-enqueue completion action must be aligned "
                          "to {0} bytes; the pool returned {1}",
                          DeviceAqlWrapAlignment, *PtrOrErr)))
      return CleanupDispatchBuffers(std::move(Err));
    initializeCompletionAction(*static_cast<DeviceAqlWrap *>(*PtrOrErr));
    Buffers.CompletionAction = *PtrOrErr;
  }

  auto CleanupAll = [&](llvm::Error E) -> llvm::Error {
    if (KernargPtr)
      E = llvm::joinErrors(std::move(E), hsa::memoryFree(Core, KernargPtr));
    return CleanupDispatchBuffers(std::move(E));
  };

  if (auto Err = writeHiddenKernelArguments(
          llvm::MutableArrayRef<uint8_t>(static_cast<uint8_t *>(KernargPtr),
                                         KernargPtr ? KD.kernarg_size : 0),
          Kernel.HiddenArgs, Dispatch, *Queue, Buffers))
    return CleanupAll(std::move(Err));

  // Reserve a slot in the queue's ring buffer and spin until it's free.
  const uint64_t WriteIdx =
      Core.callFunction<hsa_queue_add_write_index_screlease>(Queue, 1);
  while (WriteIdx -
             Core.callFunction<hsa_queue_load_read_index_scacquire>(Queue) >=
         Queue->size) {
  }

  auto *Packet = &reinterpret_cast<hsa_kernel_dispatch_packet_t *>(
      Queue->base_address)[WriteIdx & (Queue->size - 1)];
  // Dispatch's header is still zero here; the real one is stored below with
  // release ordering, which is what hands the packet to the packet processor.
  std::memcpy(Packet, &Dispatch, sizeof(Dispatch));

  const uint16_t Header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  __atomic_store_n(reinterpret_cast<uint16_t *>(Packet), Header,
                   __ATOMIC_RELEASE);

  Core.callFunction<hsa_signal_store_screlease>(
      Queue->doorbell_signal, static_cast<hsa_signal_value_t>(WriteIdx));

  hsa::signalWait(Core, Signal, HSA_SIGNAL_CONDITION_LT, 1);

  // The kernel has finished, so nothing is still bump-allocating out of the
  // printf buffer: render whatever it wrote.
  llvm::Error E = llvm::Error::success();
  if (Buffers.PrintfBuffer)
    E = drainPrintfBuffer(llvm::ArrayRef<uint8_t>(
                              static_cast<const uint8_t *>(Buffers.PrintfBuffer),
                              DefaultPrintfBufferSize),
                          Rec.PrintfFormatStrings);

  return CleanupAll(std::move(E));
}

} // namespace luthier
