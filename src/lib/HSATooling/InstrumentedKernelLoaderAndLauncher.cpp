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
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/HSA/Memory.h"
#include "luthier/HSA/MemoryPool.h"
#include "luthier/HSA/SVM.h"
#include "luthier/HSA/VMEM.h"
#include "luthier/Linker/Linker.h"
#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/Object/ObjectFileUtils.h"

#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <vector>

#define DEBUG_TYPE "luthier-instrumented-kernel-loader-and-launcher"

namespace luthier {

namespace {

/// Walk the parsed ELF and find the single kernel-function symbol.
llvm::Expected<object::AMDGCNKernelFuncSymbolRef>
findSingleKernel(const object::AMDGCNObjectFile &Obj) {
  llvm::Error IterErr = llvm::Error::success();
  std::optional<object::AMDGCNKernelFuncSymbolRef> Found;
  unsigned KernelCount = 0;
  for (const auto &KSym : Obj.kernel_functions(IterErr)) {
    ++KernelCount;
    Found = KSym;
  }
  LUTHIER_RETURN_ON_ERROR(std::move(IterErr));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      KernelCount == 1,
      llvm::formatv("Instrumented relocatable must contain exactly one "
                    "kernel function; found {0}",
                    KernelCount)));
  return *Found;
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
    llvm::DenseMap<Key, InstrumentedRecord, KeyDenseMapInfo>::iterator It) {
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher] eraseRecordLocked KD="
             << It->first.KD << " preset=" << It->first.Preset << "\n");
  llvm::Error E = llvm::Error::success();
  InstrumentedRecord &R = It->second;
  const auto Core = CoreApi.getTable();
  const auto AmdExtTbl = AmdExt.getTable();

  // Executable first (releases its references into the reader's host
  // memory), then reader.
  E = llvm::joinErrors(std::move(E), hsa::executableDestroy(Core, R.Exec));
  E = llvm::joinErrors(std::move(E),
                       hsa::codeObjectReaderDestroy(R.Reader, Core));

  // This record owns its managed-variable storage; free it here.
  for (const ManagedAlloc &Alloc : R.ManagedAllocs)
    E = llvm::joinErrors(std::move(E), freeManagedStorage(AmdExtTbl, Alloc));

  ByOriginal.erase(It);
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
  // eraseRecordLocked calls ByOriginal.erase, which bumps DenseMap's epoch and
  // may rehome other buckets — no surviving iterator can bridge iterations.
  // Drain via begin() each pass; DenseMap::end() is recomputed on each cmp.
  while (!ByOriginal.empty())
    E = llvm::joinErrors(std::move(E),
                         eraseRecordLocked(ByOriginal.begin()));
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
  auto SymIt = It->second.NameToVarSymbol.find(Name);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      SymIt != It->second.NameToVarSymbol.end(),
      llvm::formatv("Global variable '{0}' not found in the instrumented "
                    "executable for the requested kernel/preset.",
                    Name)));
  return SymIt->second;
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

  InstrumentedRecord &Rec = It->second;
  Packet.kernel_object = Rec.InstrumentedKO;
  Packet.private_segment_size =
      std::max<uint32_t>(Packet.private_segment_size, Rec.PrivateSegmentSize);

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
    const uint64_t Start = reinterpret_cast<uint64_t>(LoadedMemOrErr->data());
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
// loadInstrumented
//===----------------------------------------------------------------------===//

llvm::Expected<hsa_executable_symbol_t>
InstrumentedKernelLoaderAndLauncher::loadInstrumented(
    std::unique_ptr<llvm::MemoryBuffer> Relocatable,
    const llvm::amdhsa::kernel_descriptor_t *OriginalKD, uint64_t Preset,
    std::optional<hsa_agent_t> ExplicitAgent) {
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

  // Which device to load onto. A caller that already knows says so, which is
  // the only workable answer for a kernel descriptor HSA did not allocate --
  // pointer info cannot describe one, and asking is not merely unhelpful but
  // reports the descriptor as owned by nothing.
  auto KDAddr = reinterpret_cast<uint64_t>(OriginalKD);
  hsa_agent_t Agent;
  if (ExplicitAgent) {
    Agent = *ExplicitAgent;
    LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                   "[InstrumentedKernelLoaderAndLauncher] using the caller's "
                   "agent {0:x} for KD {1:x}\n",
                   Agent.handle, KDAddr));
  } else {
    // Resolve the agent that owns the kernel-descriptor allocation via
    // hsa_amd_pointer_info (works for loader-published and pool allocations).
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
                      "allocation. If this descriptor belongs to an application "
                      "that allocated it through the driver rather than through "
                      "HSA, pass the agent explicitly -- see "
                      "luthier::kfd::agentForGpuId.",
                      KDAddr)));
    Agent = PointerInfo.agentOwner;
  }

  llvm::sys::ScopedWriter W(Mutex);

  if (ByOriginal.contains(Key{OriginalKD, Preset}))
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "An instrumented variant for kernel_descriptor {0:x} preset {1} "
        "is already loaded",
        KDAddr, Preset));

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

  auto KernelSymOrErr = findSingleKernel(*Parsed);
  LUTHIER_RETURN_ON_ERROR(KernelSymOrErr.takeError());
  auto KernelNameOrErr = KernelSymOrErr->getName();
  LUTHIER_RETURN_ON_ERROR(KernelNameOrErr.takeError());
  std::string KernelName(*KernelNameOrErr);
  std::string KDName = KernelName + ".kd";

  // Stand up the HSA executable. The instrumented relocatable is self-contained
  // (its device globals are defined in its own copy), so there are no UND
  // globals to resolve against a global tool image.
  auto ExecOrErr = hsa::executableCreate(Core);
  LUTHIER_RETURN_ON_ERROR(ExecOrErr.takeError());
  hsa_executable_t Exec = *ExecOrErr;

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

  auto InstrSymOrErr =
      hsa::executableGetSymbolByName(Core, Exec, KDName, Agent);
  if (!InstrSymOrErr)
    return Fail(InstrSymOrErr.takeError());
  if (!InstrSymOrErr->has_value())
    return Fail(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Instrumented executable does not expose a kernel descriptor "
        "named '{0}'",
        KDName)));
  hsa_executable_symbol_t InstrSym = **InstrSymOrErr;

  auto InstrKOOrErr = hsa::executableSymbolGetAddress(Core, InstrSym);
  if (!InstrKOOrErr)
    return Fail(InstrKOOrErr.takeError());

  auto PrivSizeOrErr =
      hsa::executableSymbolGetKernelPrivateSegmentSize(Core, InstrSym);
  if (!PrivSizeOrErr)
    return Fail(PrivSizeOrErr.takeError());

  InstrumentedRecord Rec;
  Rec.RelocatableBuffer = std::move(Relocatable);
  Rec.Reader = Reader;
  Rec.Exec = Exec;
  Rec.InstrumentedKernelSym = InstrSym;
  Rec.InstrumentedKO = *InstrKOOrErr;
  Rec.PrivateSegmentSize = *PrivSizeOrErr;
  Rec.Agent = Agent;

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
    return Fail(std::move(Err));

  // Allocate + publish the managed variables this instrumented copy carries,
  // driven by its own relocatable. The allocations are owned by Rec and freed
  // when the record is erased; loadManagedVarsForRecord frees them itself if it
  // fails here (Rec is not yet cached).
  if (auto Err = loadManagedVarsForRecord(*Parsed, Rec))
    return Fail(std::move(Err));

  auto [It, Inserted] =
      ByOriginal.try_emplace(Key{OriginalKD, Preset}, std::move(Rec));
  assert(Inserted && "Concurrent insert into ByOriginal under writer lock");
  (void)Inserted;

  return InstrSym;
}

} // namespace luthier
