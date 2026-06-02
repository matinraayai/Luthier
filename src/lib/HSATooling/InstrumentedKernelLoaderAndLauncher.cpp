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
#include "luthier/HSA/HsaError.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/HSA/Memory.h"
#include "luthier/HSATooling/DeviceToolCodeLoader.h"
#include "luthier/Linker/Linker.h"
#include "luthier/Object/AMDGCNObjectFile.h"

#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolicFile.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <vector>

#define DEBUG_TYPE "luthier-instrumented-kernel-loader-and-launcher"

namespace luthier {

namespace {

/// Scan \p Obj for the \c __luthier_kernarg_layout symbol emitted by
/// \c TargetModulePatcherPass and deserialize the \c CustomKernargLayout POD it
/// anchors. Mirrors \c DeviceToolCodeLoader's \c readSubtargetMarker symbol
/// scan. Returns \c std::nullopt when the object carries no such record (the
/// kernel does not use a custom kernarg buffer).
std::optional<CustomKernargLayout>
readCustomKernargLayout(const llvm::object::ObjectFile &Obj) {
  for (const auto &Sym : Obj.symbols()) {
    auto NameOrErr = Sym.getName();
    if (!NameOrErr) {
      llvm::consumeError(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != CustomKernargLayoutGlobalName)
      continue;
    auto SecOrErr = Sym.getSection();
    if (!SecOrErr) {
      llvm::consumeError(SecOrErr.takeError());
      return std::nullopt;
    }
    auto SecIt = *SecOrErr;
    if (SecIt == Obj.section_end())
      return std::nullopt;
    auto ContentsOrErr = SecIt->getContents();
    if (!ContentsOrErr) {
      llvm::consumeError(ContentsOrErr.takeError());
      return std::nullopt;
    }
    auto AddrOrErr = Sym.getAddress();
    if (!AddrOrErr) {
      llvm::consumeError(AddrOrErr.takeError());
      return std::nullopt;
    }
    uint64_t Off = *AddrOrErr - SecIt->getAddress();
    if (Off + sizeof(CustomKernargLayout) > ContentsOrErr->size())
      return std::nullopt;
    CustomKernargLayout Layout{};
    std::memcpy(&Layout, ContentsOrErr->data() + Off,
                sizeof(CustomKernargLayout));
    if (Layout.Magic != CustomKernargLayoutMagic ||
        Layout.Version != CustomKernargLayoutVersion)
      return std::nullopt;
    return Layout;
  }
  return std::nullopt;
}

/// Construct the ROCclr-style COV5 hidden/implicit kernel arguments for \p P
/// into the \p ImplicitSize-byte region at \p Impl. Mirrors the host-side
/// hidden-arg construction ROCclr performs when reusing the application's
/// kernarg buffer. Pointer-valued hidden args (hostcall buffer, heap, queues,
/// multigrid sync, completion action) are left zero — they are provisioned by
/// the deferred host-call work, not by this kernel-argument framework.
void fillCov5ImplicitArgs(uint8_t *Impl, size_t ImplicitSize,
                          const hsa_kernel_dispatch_packet_t &P) {
  std::memset(Impl, 0, ImplicitSize);
  auto Wr32 = [&](uint32_t Off, uint32_t V) {
    if (Off + sizeof(uint32_t) <= ImplicitSize)
      std::memcpy(Impl + Off, &V, sizeof(uint32_t));
  };
  auto Wr16 = [&](uint32_t Off, uint16_t V) {
    if (Off + sizeof(uint16_t) <= ImplicitSize)
      std::memcpy(Impl + Off, &V, sizeof(uint16_t));
  };
  const uint16_t WgX = P.workgroup_size_x;
  const uint16_t WgY = P.workgroup_size_y;
  const uint16_t WgZ = P.workgroup_size_z;
  const uint32_t Gx = P.grid_size_x;
  const uint32_t Gy = P.grid_size_y;
  const uint32_t Gz = P.grid_size_z;
  auto Blocks = [](uint32_t G, uint16_t Wg) -> uint32_t {
    return Wg ? (G + Wg - 1u) / Wg : 0u;
  };
  Wr32(cov5::BlockCountX, Blocks(Gx, WgX));
  Wr32(cov5::BlockCountY, Blocks(Gy, WgY));
  Wr32(cov5::BlockCountZ, Blocks(Gz, WgZ));
  Wr16(cov5::GroupSizeX, WgX);
  Wr16(cov5::GroupSizeY, WgY);
  Wr16(cov5::GroupSizeZ, WgZ);
  Wr16(cov5::RemainderX, WgX ? static_cast<uint16_t>(Gx % WgX) : 0);
  Wr16(cov5::RemainderY, WgY ? static_cast<uint16_t>(Gy % WgY) : 0);
  Wr16(cov5::RemainderZ, WgZ ? static_cast<uint16_t>(Gz % WgZ) : 0);
  const uint16_t Dims = static_cast<uint16_t>(
      (P.setup >> HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS) &
      ((1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS) - 1u));
  Wr16(cov5::GridDims, Dims);
}

} // namespace

//===----------------------------------------------------------------------===//
// Construction / destruction
//===----------------------------------------------------------------------===//

InstrumentedKernelLoaderAndLauncher::InstrumentedKernelLoaderAndLauncher(
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreApi,
    const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExt,
    const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
        &Loader,
    DeviceToolCodeLoader &DeviceCode)
    : CoreApi(CoreApi), AmdExt(AmdExt), Loader(Loader), DeviceCode(DeviceCode) {
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
             << It->first.KD << " preset=" << It->first.Preset << " instrKO=0x"
             << llvm::Twine::utohexstr(It->second.InstrumentedKO) << "\n");
  llvm::Error E = llvm::Error::success();
  InstrumentedRecord &R = It->second;
  const auto Core = CoreApi.getTable();

  // Executable first (releases its references into the reader's host
  // memory), then reader. The retained MemoryBuffer drops with the map
  // entry below.
  E = llvm::joinErrors(std::move(E), hsa::executableDestroy(Core, R.Exec));
  E = llvm::joinErrors(std::move(E),
                       hsa::codeObjectReaderDestroy(R.Reader, Core));

  // Reclaim any in-flight custom kernarg buffer for this record.
  if (R.CustomKernargAlloc) {
    E = llvm::joinErrors(std::move(E),
                         hsa::memoryFree(Core, R.CustomKernargAlloc));
    R.CustomKernargAlloc = nullptr;
  }

  ByOriginal.erase(It);
  return E;
}

//===----------------------------------------------------------------------===//
// unloadAll
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

//===----------------------------------------------------------------------===//
// unloadInstrumentedIfExists
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::unloadInstrumentedIfExists(
    const llvm::amdhsa::kernel_descriptor_t *OriginalKD, uint64_t Preset) {
  llvm::sys::ScopedWriter W(Mutex);
  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher] "
                                "unloadInstrumentedIfExists KD="
                             << OriginalKD << " preset=" << Preset << "\n");
  auto It = ByOriginal.find(Key{OriginalKD, Preset});
  if (It == ByOriginal.end()) {
    LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                  "no matching record\n");
    return llvm::Error::success();
  }
  return eraseRecordLocked(It);
}

//===----------------------------------------------------------------------===//
// overrideWithInstrumented
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::overrideWithInstrumented(
    hsa_kernel_dispatch_packet_t &Packet, uint64_t Preset) {
  // Writer lock: when the kernel uses a custom kernarg buffer we allocate and
  // record a per-launch buffer (mutating the record), so the reader lock the
  // pure-swap path used is not sufficient.
  llvm::sys::ScopedWriter W(Mutex);
  const auto *KD = reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      Packet.kernel_object);
  LLVM_DEBUG(
      luthier::dbgs()
      << "[InstrumentedKernelLoaderAndLauncher] overrideWithInstrumented "
         "origKO=0x"
      << llvm::Twine::utohexstr(Packet.kernel_object) << " preset=" << Preset
      << "\n");
  auto It = ByOriginal.find(Key{KD, Preset});
  if (It == ByOriginal.end()) {
    LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                  "no cached instrumented variant\n");
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "No instrumented variant cached for kernel_object {0:x} preset {1}",
        Packet.kernel_object, Preset));
  }

  InstrumentedRecord &Rec = It->second;
  Packet.kernel_object = Rec.InstrumentedKO;
  Packet.private_segment_size =
      std::max<uint32_t>(Packet.private_segment_size, Rec.PrivateSegmentSize);
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher]   "
                "swapped to instrKO=0x"
             << llvm::Twine::utohexstr(Rec.InstrumentedKO)
             << " privSegSize=" << Packet.private_segment_size << "\n");

  if (Rec.HasCustomKernarg) {
    if (auto Err = buildCustomKernargBuffer(Rec, Packet))
      return Err;
  }
  return llvm::Error::success();
}

llvm::Error InstrumentedKernelLoaderAndLauncher::buildCustomKernargBuffer(
    InstrumentedRecord &Rec, hsa_kernel_dispatch_packet_t &Packet) {
  const auto Core = CoreApi.getTable();
  const CustomKernargLayout &L = Rec.KernargLayout;

  // Reclaim the previous launch's buffer. Tools serialize dispatches of a given
  // kernel and wait on its completion signal before issuing the next, so the
  // prior buffer is no longer referenced by the device here.
  if (Rec.CustomKernargAlloc) {
    llvm::consumeError(hsa::memoryFree(Core, Rec.CustomKernargAlloc));
    Rec.CustomKernargAlloc = nullptr;
  }

  auto RegionOrErr = hsa::agentFindKernargRegion(Core, Rec.Agent);
  LUTHIER_RETURN_ON_ERROR(RegionOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      RegionOrErr->has_value(),
      "custom kernarg: kernel's agent exposes no kernarg region"));
  hsa_region_t KernargRegion = **RegionOrErr;

  auto AllocOrErr = hsa::memoryAllocate(Core, KernargRegion, L.TotalSize);
  LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());
  void *CustomBuf = *AllocOrErr;

  // Stage the buffer contents on the host, then copy them into the kernarg
  // allocation in one shot.
  std::vector<uint8_t> Stage(L.TotalSize, 0);
  const uint64_t OrigKernarg =
      reinterpret_cast<uint64_t>(Packet.kernarg_address);
  std::memcpy(Stage.data() + L.OrigKernargPtrOffset, &OrigKernarg,
              sizeof(uint64_t));
  // Tool explicit args occupy [ExplicitOffset, ExplicitOffset + ExplicitSize);
  // zero-filled until the explicit-arg declaration API lands (ExplicitSize=0).
  fillCov5ImplicitArgs(Stage.data() + L.ImplicitOffset,
                       L.TotalSize - L.ImplicitOffset, Packet);

  if (auto Err = hsa::memoryCopy(Core, CustomBuf, Stage.data(), L.TotalSize)) {
    llvm::consumeError(hsa::memoryFree(Core, CustomBuf));
    return Err;
  }

  Rec.CustomKernargAlloc = CustomBuf;
  Packet.kernarg_address = CustomBuf;
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher]   custom kernarg buffer "
                "@"
             << CustomBuf << " size=" << L.TotalSize << " orig kernarg=0x"
             << llvm::Twine::utohexstr(OrigKernarg) << "\n");
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// invalidateOriginalExec
//===----------------------------------------------------------------------===//

llvm::Error InstrumentedKernelLoaderAndLauncher::invalidateOriginalExec(
    hsa_executable_t Exec) {
  llvm::sys::ScopedWriter W(Mutex);
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher] invalidateOriginalExec "
                "exec="
             << Exec.handle << "\n");
  llvm::Error E = llvm::Error::success();

  // Collect the loaded device-memory ranges of every LCO owned by the
  // about-to-be-destroyed executable.
  llvm::SmallVector<hsa_loaded_code_object_t, 2> LCOs;
  if (auto Err =
          hsa::executableGetLoadedCodeObjects(Loader.getTable(), Exec, LCOs))
    return llvm::joinErrors(std::move(E), std::move(Err));
  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                             << LCOs.size() << " LCO(s)\n");

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
    LLVM_DEBUG(luthier::dbgs()
               << "[InstrumentedKernelLoaderAndLauncher]   LCO range [0x"
               << llvm::Twine::utohexstr(Start) << ", 0x"
               << llvm::Twine::utohexstr(Start + LoadedMemOrErr->size())
               << ")\n");
  }

  // Snapshot the victim keys first so we don't invalidate map iterators
  // while erasing.
  llvm::SmallVector<Key, 4> Victims;
  for (const auto &[K, _] : ByOriginal) {
    const auto Addr = reinterpret_cast<uint64_t>(K.KD);
    for (const Range &R : Ranges) {
      if (Addr >= R.Start && Addr < R.End) {
        Victims.push_back(K);
        break;
      }
    }
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher]   " << Victims.size()
             << " victim record(s) of " << ByOriginal.size() << " total\n");

  for (const Key &K : Victims) {
    auto It = ByOriginal.find(K);
    if (It != ByOriginal.end())
      E = llvm::joinErrors(std::move(E), eraseRecordLocked(It));
  }
  return E;
}

//===----------------------------------------------------------------------===//
// loadInstrumented
//===----------------------------------------------------------------------===//

namespace {

/// Walk the parsed ELF and find the single kernel-function symbol.
/// Errors unless exactly one kernel function exists.
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
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher]   findSingleKernel: "
             << KernelCount << " kernel function(s)\n");
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      KernelCount == 1,
      llvm::formatv("Instrumented relocatable must contain exactly one "
                    "kernel function; found {0}",
                    KernelCount)));
  return *Found;
}

/// Pair of (UND global variable name, its resolved device address) the
/// caller will feed into \c hsa_executable_agent_global_variable_define.
struct ExternRef {
  std::string Name;
  void *Address;
};

/// Walk \p Obj 's symbol table. For every undefined global/weak symbol
/// of object type, resolve it via \p DeviceCode on \p Agent and queue
/// it. Errors if any undefined symbol is a function (the codegen
/// pipeline must not emit inter-module function references).
llvm::Expected<llvm::SmallVector<ExternRef, 4>>
collectExternals(const hsa::ApiTableContainer<::CoreApiTable> &Core,
                 const object::AMDGCNObjectFile &Obj,
                 DeviceToolCodeLoader &DeviceCode, hsa_agent_t Agent) {
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher]   collectExternals "
                "for agent "
             << Agent.handle << "\n");
  llvm::SmallVector<ExternRef, 4> Out;
  for (const auto &Sym : Obj.symbols()) {
    auto FlagsOrErr = Sym.getFlags();
    LUTHIER_RETURN_ON_ERROR(FlagsOrErr.takeError());
    if (!(*FlagsOrErr & llvm::object::SymbolRef::SF_Undefined))
      continue;

    auto NameOrErr = Sym.getName();
    LUTHIER_RETURN_ON_ERROR(NameOrErr.takeError());
    if (NameOrErr->empty())
      continue;

    const uint8_t EType = Sym.getELFType();
    if (EType == llvm::ELF::STT_FUNC) {
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Instrumented relocatable references undefined function "
          "symbol '{0}'; inter-module function references are not "
          "supported by the launcher",
          *NameOrErr));
    }
    if (EType != llvm::ELF::STT_OBJECT && EType != llvm::ELF::STT_NOTYPE)
      continue;

    const uint8_t Binding = Sym.getBinding();
    if (Binding != llvm::ELF::STB_GLOBAL && Binding != llvm::ELF::STB_WEAK)
      continue;

    auto ExtSymOrErr = DeviceCode.lookupGlobalVariable(*NameOrErr, Agent);
    LUTHIER_RETURN_ON_ERROR(ExtSymOrErr.takeError());
    auto AddrOrErr = hsa::executableSymbolGetAddress(Core, *ExtSymOrErr);
    LUTHIER_RETURN_ON_ERROR(AddrOrErr.takeError());
    LLVM_DEBUG(luthier::dbgs()
               << "[InstrumentedKernelLoaderAndLauncher]     resolved extern "
               << *NameOrErr << " -> 0x" << llvm::Twine::utohexstr(*AddrOrErr)
               << "\n");

    Out.push_back(
        ExternRef{NameOrErr->str(), reinterpret_cast<void *>(*AddrOrErr)});
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher]   collectExternals "
                "produced "
             << Out.size() << " entries\n");
  return Out;
}

} // namespace

llvm::Expected<hsa_executable_symbol_t>
InstrumentedKernelLoaderAndLauncher::loadInstrumented(
    std::unique_ptr<llvm::MemoryBuffer> Relocatable,
    const llvm::amdhsa::kernel_descriptor_t *OriginalKD, uint64_t Preset) {
  LLVM_DEBUG(luthier::dbgs()
             << "[InstrumentedKernelLoaderAndLauncher] loadInstrumented KD="
             << OriginalKD << " preset=" << Preset << " relocSize="
             << (Relocatable ? Relocatable->getBufferSize() : 0) << "\n");
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Relocatable != nullptr,
      "Null relocatable MemoryBuffer passed to loadInstrumented"));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      OriginalKD != nullptr,
      "Null kernel-descriptor pointer passed to loadInstrumented"));

  const auto Core = CoreApi.getTable();

  // Resolve the agent that owns the kernel-descriptor allocation via
  // hsa_amd_pointer_info. This works regardless of whether the KD was
  // published through the HSA loader (the loader path) or allocated
  // directly out of an HSA memory pool — the latter case can't be
  // resolved through the LoadedCodeObjectCache.
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
  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                "owning agent "
                             << Agent.handle << "\n");

  llvm::sys::ScopedWriter W(Mutex);

  // Duplicate-key check.
  if (ByOriginal.contains(Key{OriginalKD, Preset})) {
    LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                  "duplicate key, refusing\n");
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "An instrumented variant for kernel_descriptor {0:x} preset {1} "
        "is already loaded",
        KDAddr, Preset));
  }

  // The instrumented bytes come out of `NewPMAsmPrinter` as a REL
  // (relocatable) — no `.dynsym`, so `kernel_functions()` (which iterates
  // dynamic symbols) yields 0, and HSA's executable loader rejects it
  // outright. Link to a shared object via `ld.lld` so we get a proper
  // `.dynsym` + PT_DYNAMIC layout that downstream code expects.
  llvm::SmallVector<char, 0> LinkedBuf;
  LUTHIER_RETURN_ON_ERROR(linker::linkRelocatableToExecutable(
      llvm::ArrayRef<char>(Relocatable->getBufferStart(),
                           Relocatable->getBufferSize()),
      LinkedBuf));
  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                "linked relocatable -> "
                             << LinkedBuf.size() << " bytes\n");
  auto Linked = std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(LinkedBuf), "luthier.instrumented.linked",
      /*RequiresNullTerminator=*/false);
  Relocatable = std::move(Linked);

  // Parse the linked executable as an AMDGCN ELF. The parse is non-owning
  // over the MemoryBuffer's bytes.
  llvm::MemoryBufferRef RelocRef = Relocatable->getMemBufferRef();
  auto ParsedOrErr = object::AMDGCNObjectFile::createAMDGCNObjectFile(RelocRef);
  LUTHIER_RETURN_ON_ERROR(ParsedOrErr.takeError());
  std::unique_ptr<object::AMDGCNObjectFile> Parsed = std::move(*ParsedOrErr);

  // Pick up the custom kernarg buffer layout, if the patcher emitted one.
  std::optional<CustomKernargLayout> KernargLayout =
      readCustomKernargLayout(*Parsed);
  LLVM_DEBUG(
      if (KernargLayout) luthier::dbgs()
      << "[InstrumentedKernelLoaderAndLauncher]   custom kernarg layout: "
         "total="
      << KernargLayout->TotalSize
      << " implicitOff=" << KernargLayout->ImplicitOffset << "\n");

  // The relocatable is expected to contain exactly one kernel function;
  // we make no claim about its name — the codegen pipeline picks it.
  auto KernelSymOrErr = findSingleKernel(*Parsed);
  LUTHIER_RETURN_ON_ERROR(KernelSymOrErr.takeError());
  auto KernelNameOrErr = KernelSymOrErr->getName();
  LUTHIER_RETURN_ON_ERROR(KernelNameOrErr.takeError());
  std::string KernelName(*KernelNameOrErr);
  std::string KDName = KernelName + ".kd";
  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                "kernel='"
                             << KernelName << "', KD='" << KDName << "'\n");

  auto ExternsOrErr = collectExternals(Core, *Parsed, DeviceCode, Agent);
  LUTHIER_RETURN_ON_ERROR(ExternsOrErr.takeError());

  // Stand up the HSA executable. From here on, every failure path must
  // tear down whatever has been built so far.
  auto ExecOrErr = hsa::executableCreate(Core);
  LUTHIER_RETURN_ON_ERROR(ExecOrErr.takeError());
  hsa_executable_t Exec = *ExecOrErr;

  auto FailExec = [&](llvm::Error E) -> llvm::Error {
    return llvm::joinErrors(std::move(E), hsa::executableDestroy(Core, Exec));
  };

  for (const ExternRef &Ext : *ExternsOrErr) {
    if (auto Err = hsa::executableDefineExternalAgentGlobalVariable(
            Core, Exec, Agent, Ext.Name, Ext.Address))
      return FailExec(std::move(Err));
  }

  auto ReaderOrErr =
      hsa::codeObjectReaderCreateFromMemory(Core, RelocRef.getBuffer());
  if (!ReaderOrErr)
    return FailExec(ReaderOrErr.takeError());
  hsa_code_object_reader_t Reader = *ReaderOrErr;

  auto FailExecAndReader = [&](llvm::Error E) -> llvm::Error {
    return llvm::joinErrors(
        llvm::joinErrors(std::move(E), hsa::executableDestroy(Core, Exec)),
        hsa::codeObjectReaderDestroy(Reader, Core));
  };

  if (auto Err = hsa::executableLoadAgentCodeObject(Core, Exec, Reader, Agent)
                     .takeError())
    return FailExecAndReader(std::move(Err));

  if (auto Err = hsa::executableFreeze(Core, Exec))
    return FailExecAndReader(std::move(Err));

  auto InstrSymOrErr =
      hsa::executableGetSymbolByName(Core, Exec, KDName, Agent);
  if (!InstrSymOrErr)
    return FailExecAndReader(InstrSymOrErr.takeError());
  if (!InstrSymOrErr->has_value())
    return FailExecAndReader(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Instrumented executable does not expose a kernel descriptor "
        "named '{0}'",
        KDName)));
  hsa_executable_symbol_t InstrSym = **InstrSymOrErr;

  auto InstrKOOrErr = hsa::executableSymbolGetAddress(Core, InstrSym);
  if (!InstrKOOrErr)
    return FailExecAndReader(InstrKOOrErr.takeError());

  auto PrivSizeOrErr =
      hsa::executableSymbolGetKernelPrivateSegmentSize(Core, InstrSym);
  if (!PrivSizeOrErr)
    return FailExecAndReader(PrivSizeOrErr.takeError());

  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                "instrumented KO=0x"
                             << llvm::Twine::utohexstr(*InstrKOOrErr)
                             << " privSegSize=" << *PrivSizeOrErr << "\n");

  InstrumentedRecord Rec;
  Rec.RelocatableBuffer = std::move(Relocatable);
  Rec.Reader = Reader;
  Rec.Exec = Exec;
  Rec.InstrumentedKernelSym = InstrSym;
  Rec.InstrumentedKO = *InstrKOOrErr;
  Rec.PrivateSegmentSize = *PrivSizeOrErr;
  Rec.Agent = Agent;
  if (KernargLayout) {
    Rec.HasCustomKernarg = true;
    Rec.KernargLayout = *KernargLayout;
  }

  auto [It, Inserted] =
      ByOriginal.try_emplace(Key{OriginalKD, Preset}, std::move(Rec));
  assert(Inserted && "Concurrent insert into ByOriginal under writer lock");
  (void)Inserted;

  LLVM_DEBUG(luthier::dbgs() << "[InstrumentedKernelLoaderAndLauncher]   "
                                "record inserted; total now "
                             << ByOriginal.size() << "\n");
  return InstrSym;
}

} // namespace luthier
