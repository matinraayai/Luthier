//===-- DeviceToolCodeParser.cpp ----------------------------------*-C++-*-===//
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
/// Implements Clang-offload-bundle parsing + per-subtarget LLVM-bitcode
/// extraction, with a SPIR-V → AMDGCN JIT fallback for ISAs not shipped as a
/// precompiled bitcode slice in \c DeviceToolCodeParser.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/DeviceToolCodeParser.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include <algorithm>
#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/BinaryFormat/Magic.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/OffloadBundle.h>
#include <llvm/Support/CrashRecoveryContext.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <tuple>

#ifdef LUTHIER_HAS_SPIRV_TRANSLATOR
#include "luthier/ToolIRCompilation/FinalizeIntrinsicsPass.h"
#include "luthier/ToolIRCompilation/MarkAnnotationsPass.h"
#include "luthier/ToolIRCompilation/SubstituteAMDGCNIntrinsicsPass.h"
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>
#include <sstream>
#endif

#define DEBUG_TYPE "luthier-device-tool-code-parser"

namespace luthier {

namespace {

/// A device slice kept from the bundle: a view of its bytes plus its Clang
/// offload-bundle entry ID (the target string the bundle was built with).
struct BundleSlice {
  llvm::MemoryBufferRef Buf;
  std::string ID;
};

/// Derive the LLVM ISA tuple (triple, CPU, features) from a Clang offload
/// bundle entry \p ID
llvm::Expected<std::tuple<llvm::Triple, std::string, llvm::SubtargetFeatures>>
parseSliceISA(llvm::StringRef ID) {
  // <kind>-<triple>[-<target id>[:target features]]
  // <triple> := <arch>-<vendor>-<os>-<env>
  llvm::SmallVector<llvm::StringRef, 6> Components;
  ID.split(Components, '-', /*MaxSplit=*/5);
  if (Components.size() < 5) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Malformed target string {0}", ID));
  }

  llvm::StringRef CpuIdWithFeature =
      Components.size() == 6 ? Components.back() : "";

  auto [CpuID, FeatureString] = CpuIdWithFeature.split(':');

  if (CpuID.empty()) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Empy CPU ID in target ID {0}", ID));
  }

  llvm::ArrayRef TripleSlice{&Components[1], /*length=*/4};

  llvm::Triple TT(llvm::join(TripleSlice, "-"));

  llvm::SmallVector<llvm::StringRef, 6> ParsedFeatures;
  FeatureString.split(ParsedFeatures, ':');

  llvm::SubtargetFeatures FS;

  for (llvm::StringRef Feature : ParsedFeatures) {
    if (Feature.empty())
      continue;
    const char Sign = Feature.back();
    if (Sign != '+' && Sign != '-')
      return LUTHIER_MAKE_GENERIC_ERROR("Offload bundle entry ID '" + ID.str() +
                                        "' feature '" + Feature.str() +
                                        "' is missing a +/- sign.");
    const llvm::StringRef Name = Feature.drop_back();
    const bool Enable = Sign == '+';
    // wave32 might appear as `wavefrontsize64-` in the AMDGPU target ID;
    // convert it to `+wavefrontsize32` to match IR's feature string
    if (Name == "wavefrontsize64")
      FS.AddFeature(Enable ? "wavefrontsize64" : "wavefrontsize32",
                    /*Enable=*/true);
    else
      FS.AddFeature(Name, Enable);
  }

  return std::make_tuple(TT, std::string(CpuID), std::move(FS));
}

/// Parses a Clang offload \p Bundle into a list of kept device slices — raw
/// LLVM bitcode (the Luthier offload-bundle format) or an AMD-flavored SPIR-V
/// slice — each paired with its bundle entry ID. Handles compressed (CCOB) and
/// uncompressed bundles. On a compressed input, \p DecompressedHolder receives
/// the owning buffer of the decompressed payload — caller must retain it.
llvm::Error
parseOffloadBundle(llvm::MemoryBufferRef Bundle,
                   llvm::SmallVectorImpl<BundleSlice> &SliceBufs,
                   std::unique_ptr<llvm::MemoryBuffer> &DecompressedHolder) {
  LLVM_DEBUG(luthier::dbgs() << "[DeviceToolCodeParser] parseOffloadBundle: "
                             << Bundle.getBufferSize() << " bytes\n");
  if (Bundle.getBufferSize() == 0)
    return LUTHIER_MAKE_GENERIC_ERROR("Empty fat-binary bundle.");

  auto Magic = llvm::identify_magic(Bundle.getBuffer());

  llvm::MemoryBufferRef ParseBuf = Bundle;
  bool Decompressed = false;
  if (Magic == llvm::file_magic::offload_bundle_compressed) {
    LLVM_DEBUG(luthier::dbgs() << "[DeviceToolCodeParser] bundle is CCOB; "
                                  "decompressing\n");
    auto Input = llvm::MemoryBuffer::getMemBuffer(
        Bundle, /*RequiresNullTerminator=*/false);
    auto DecompOrErr =
        llvm::object::CompressedOffloadBundle::decompress(*Input, nullptr);
    if (!DecompOrErr)
      return DecompOrErr.takeError();
    DecompressedHolder = std::move(*DecompOrErr);
    ParseBuf = DecompressedHolder->getMemBufferRef();
    Decompressed = true;
  } else if (Magic != llvm::file_magic::offload_bundle) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Bundle does not start with __CLANG_OFFLOAD_BUNDLE__ or CCOB magic.");
  }

  auto BundleOrErr = llvm::object::OffloadBundleFatBin::create(
      ParseBuf, /*SectionOffset=*/0, "fatbin", Decompressed);
  if (!BundleOrErr)
    return BundleOrErr.takeError();

  for (auto &Entry : (*BundleOrErr)->getEntries()) {
    if (Entry.Size == 0)
      continue;
    llvm::StringRef SliceBytes(ParseBuf.getBufferStart() + Entry.Offset,
                               Entry.Size);
    // Keep only raw LLVM bitcode and SPIR-V slices. Skip everything else for
    // now
    const llvm::file_magic SliceMagic = llvm::identify_magic(SliceBytes);
    if (SliceMagic == llvm::file_magic::bitcode ||
        SliceMagic == llvm::file_magic::spirv_object)
      SliceBufs.push_back(
          {llvm::MemoryBufferRef{SliceBytes, "fat-binary slice"}, Entry.ID});
  }
  LLVM_DEBUG(luthier::dbgs() << "[DeviceToolCodeParser] parseOffloadBundle "
                                "produced "
                             << SliceBufs.size() << " slice(s)\n");
  return llvm::Error::success();
}

} // namespace

std::string
DeviceToolCodeParser::canonicalLLVMISAKey(const llvm::Triple &T,
                                          llvm::StringRef CPU,
                                          const llvm::SubtargetFeatures &F) {
  std::vector<std::string> Sorted = F.getFeatures();
  llvm::sort(Sorted);
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  OS << T.str() << "--" << CPU;
  for (llvm::StringRef Feat : Sorted)
    OS << ',' << Feat;
  return OS.str();
}

llvm::Expected<uint64_t>
DeviceToolCodeParser::calculateBundleSize(const void *Bundle) {
  if (Bundle == nullptr)
    return 0;
  auto *P = static_cast<const char *>(Bundle);

  constexpr size_t CompressedBundleMagicSize = llvm::StringRef("CCOB").size();
  constexpr size_t ClangOffloadBundleMagicSize =
      llvm::StringRef("__CLANG_OFFLOAD_BUNDLE__").size();

  constexpr size_t MaxMagicSize =
      std::max(CompressedBundleMagicSize, ClangOffloadBundleMagicSize);

  llvm::StringRef BundleMagicBuffer{P, MaxMagicSize};

  llvm::CrashRecoveryContext CRC;
  llvm::Error Err = llvm::Error::success();
  (void)Err.operator bool();
  size_t BundleSize = 0;

  bool Ok = CRC.RunSafely([&] {
    llvm::file_magic BundleMagic = llvm::identify_magic(BundleMagicBuffer);
    if (BundleMagic == llvm::file_magic::offload_bundle_compressed) {
      auto HdrOrErr = llvm::object::CompressedOffloadBundle::
          CompressedBundleHeader::tryParse(
              llvm::StringRef(BundleMagicBuffer.data(), /*Length=*/64));
      if (!HdrOrErr) {
        Err = std::move(HdrOrErr.takeError());
        return;
      }
      BundleSize = HdrOrErr->FileSize.value_or(0);
    } else if (BundleMagic == llvm::file_magic::offload_bundle) {
      P += ClangOffloadBundleMagicSize;
      auto ReadU64 = [&P]() {
        uint64_t V;
        std::memcpy(&V, P, sizeof(V));
        P += sizeof(V);
        return V;
      };
      const uint64_t NumEntries = ReadU64();
      uint64_t MaxEnd = 0;
      for (uint64_t I = 0; I < NumEntries; ++I) {
        const uint64_t Off = ReadU64();
        const uint64_t Sz = ReadU64();
        const uint64_t IDLen = ReadU64();
        P += IDLen;
        MaxEnd = std::max(MaxEnd, Off + Sz);
      }
      BundleSize = MaxEnd;
    } else {
      Err = LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Invalid file magic : {0}", BundleMagic));
    }
  });
  if (!Ok) {
    return llvm::joinErrors(
        LUTHIER_MAKE_GENERIC_ERROR(
            "Failed to determine the size of the FAT binary."),
        std::move(Err));
  }
  if (Err)
    return Err;

  return BundleSize;
}

llvm::Error DeviceToolCodeParser::addSlice(llvm::MemoryBufferRef Slice,
                                           llvm::StringRef ID) {
  const llvm::file_magic Magic = llvm::identify_magic(Slice.getBuffer());
  if (Magic == llvm::file_magic::bitcode) {
    // Derive the LLVM ISA key from the slice's offload-bundle entry ID
    auto ISAOrErr = parseSliceISA(ID);
    if (!ISAOrErr)
      return ISAOrErr.takeError();
    auto &[TT, CPU, Features] = *ISAOrErr;

    std::string Key = canonicalLLVMISAKey(TT, CPU, Features);
    LLVM_DEBUG(luthier::dbgs() << "[DeviceToolCodeParser] addBitcodeSlice id=["
                               << ID << "] key=[" << Key
                               << "] bcSize=" << Slice.getBufferSize() << "\n");
    if (Slices.contains(Key))
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Duplicate LLVM ISA in bitcode input: " + Key);

    Slices.insert({std::move(Key), Slice});
    return llvm::Error::success();
  }
  /// TODO: Support more than one SPIR-V slice
  if (Magic == llvm::file_magic::spirv_object) {
    if (SpirvSlice)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Bundle carries more than one SPIR-V slice.");
    SpirvSlice = Slice;
    LLVM_DEBUG(luthier::dbgs()
               << "[DeviceToolCodeParser] stashed SPIR-V slice ("
               << Slice.getBufferSize() << " bytes)\n");
    return llvm::Error::success();
  }
  return LUTHIER_MAKE_GENERIC_ERROR(
      "Fat-binary slice is neither LLVM bitcode nor SPIR-V.");
}

DeviceToolCodeParser::DeviceToolCodeParser(const void *Bundle,
                                           llvm::Error &Err) {
  /// Enable crash recovery context for potential segfaults when parsing the
  /// FAT binary
  llvm::CrashRecoveryContext::Enable();
  llvm::ErrorAsOutParameter EAO(&Err);
  if (Err)
    return; // Upstream already recorded a failure; don't overwrite.
  if (!Bundle)
    return; // No bundle = no device-side logic.

  size_t BundleSize = 0;

  Err = calculateBundleSize(Bundle).moveInto(BundleSize);
  if (Err)
    return;

  auto BundleMemBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef{static_cast<const char *>(Bundle), BundleSize},
      /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  llvm::MemoryBufferRef BundleRef = BundleMemBuffer->getMemBufferRef();

  RetainedBuffers.push_back(std::move(BundleMemBuffer));

  llvm::SmallVector<BundleSlice, 4> SliceBufs;
  std::unique_ptr<llvm::MemoryBuffer> DecompressedHolder;
  if (llvm::Error E =
          parseOffloadBundle(BundleRef, SliceBufs, DecompressedHolder)) {
    Err = std::move(E);
    return;
  }
  if (DecompressedHolder)
    RetainedBuffers.push_back(std::move(DecompressedHolder));

  for (const BundleSlice &Slice : SliceBufs) {
    if (auto E = addSlice(Slice.Buf, Slice.ID)) {
      Err = std::move(E);
      return;
    }
  }
  LLVM_DEBUG(luthier::dbgs()
             << "[DeviceToolCodeParser] ctor(bundle): registered "
             << Slices.size() << " slice(s)" << (SpirvSlice ? " + SPIR-V" : "")
             << "\n");
}

llvm::Expected<std::unique_ptr<llvm::Module>>
DeviceToolCodeParser::translateSpirvFallback(
    const llvm::Triple &T, llvm::StringRef CPU,
    const llvm::SubtargetFeatures &Features, llvm::StringRef Key,
    llvm::LLVMContext &Ctx, llvm::OptimizationLevel OptLevel) {
#ifndef LUTHIER_HAS_SPIRV_TRANSLATOR
  (void)T;
  (void)CPU;
  (void)Features;
  (void)Key;
  (void)Ctx;
  return LUTHIER_MAKE_GENERIC_ERROR(
      "No precompiled slice matched the requested ISA and Luthier was built "
      "without the SPIR-V translator, so the SPIR-V JIT fallback is "
      "unavailable.");
#else
  if (!SpirvSlice)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "No precompiled slice matched the requested ISA and the bundle "
        "carries no SPIR-V slice for the JIT fallback.");

  LLVM_DEBUG(luthier::dbgs()
             << "[DeviceToolCodeParser] SPIR-V JIT fallback for [" << Key
             << "]\n");

  // 1) SPIR-V -> LLVM IR into the caller's context.
  std::string SpirvStr(SpirvSlice->getBuffer());
  std::istringstream IS(SpirvStr);
  llvm::Module *RawM = nullptr;
  std::string SpirvErr;
  if (!llvm::readSpirv(Ctx, IS, RawM, SpirvErr))
    return LUTHIER_MAKE_GENERIC_ERROR("SPIR-V -> LLVM IR translation failed: " +
                                      SpirvErr);
  std::unique_ptr<llvm::Module> M(RawM);

  // 2) Retarget the module at the requested ISA: triple + datalayout from a
  // fresh TargetMachine, and the per-function target-cpu/target-features the
  // downstream pipeline and codegen rely on.
  std::string TgtErr;
  const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget(T, TgtErr);
  if (TheTarget == nullptr)
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("TargetRegistry::lookupTarget failed for triple {0}: {1}",
                      T.str(), TgtErr));
  llvm::TargetOptions TMOpts;
  std::unique_ptr<llvm::TargetMachine> TM(TheTarget->createTargetMachine(
      T, CPU, Features.getString(), TMOpts, /*RM=*/std::nullopt));
  if (!TM)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "createTargetMachine returned nullptr for the SPIR-V fallback.");
  M->setTargetTriple(T);
  M->setDataLayout(TM->createDataLayout());
  const std::string FeatStr = Features.getString();
  for (llvm::Function &F : *M) {
    F.addFnAttr("target-cpu", CPU);
    if (!FeatStr.empty())
      F.addFnAttr("target-features", FeatStr);
  }

  // 3) Run the Luthier IR compilation pipeline
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB(TM.get());
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptLevel);
  MPM.addPass(MarkAnnotationsPass());
  MPM.addPass(FinalizeIntrinsicsPass());
  MPM.addPass(SubstituteAMDGCNIntrinsicsPass());
  MPM.run(*M, MAM);

  // 4) Serialize the processed module and cache it under the requested key so
  // later requests for this ISA hit the bitcode path.
  llvm::SmallVector<char, 0> BcBuf;
  {
    llvm::raw_svector_ostream OS(BcBuf);
    llvm::WriteBitcodeToFile(*M, OS);
  }
  auto Owned = std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(BcBuf), "luthier.spirv.jit." + Key.str(),
      /*RequiresNullTerminator=*/false);
  llvm::MemoryBufferRef BcRef = Owned->getMemBufferRef();
  RetainedBuffers.push_back(std::move(Owned));

  Slices.insert({Key.str(), std::move(BcRef)});

  LLVM_DEBUG(luthier::dbgs()
             << "[DeviceToolCodeParser] SPIR-V JIT produced + cached " << Key
             << " (" << BcRef.getBufferSize() << " bytes)\n");
  return M;
#endif
}

llvm::Expected<std::unique_ptr<llvm::Module>>
DeviceToolCodeParser::parseModule(const llvm::Triple &T, llvm::StringRef CPU,
                                  const llvm::SubtargetFeatures &Features,
                                  llvm::LLVMContext &Ctx,
                                  llvm::OptimizationLevel OptLevel) {
  std::lock_guard Lock(Mutex);
  std::string Key = canonicalLLVMISAKey(T, CPU, Features);
  LLVM_DEBUG(luthier::dbgs()
             << "[DeviceToolCodeParser] parseModule key=[" << Key << "]\n");
  auto It = Slices.find(Key);
  if (It != Slices.end()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[DeviceToolCodeParser]   matched slice [" << It->first()
               << "], parsing " << It->second.getBufferSize()
               << " bytes of bitcode\n");
    auto MOrErr = llvm::parseBitcodeFile(It->second, Ctx);
#ifndef NDEBUG
    // Debug-only: validate that the bitcode's own ISA matches the slice ID we
    // keyed it under (the cache key is derived from the bundle entry ID, not
    // the bitcode). Re-derive the canonical key from the parsed module the same
    // way the request key is built and compare; warn on a mismatch.
    if (MOrErr) {
      const llvm::Module &M = **MOrErr;
      llvm::StringRef BCPU, BFeat;
      for (const llvm::Function &F : M)
        if (F.hasFnAttribute("target-cpu")) {
          BCPU = F.getFnAttribute("target-cpu").getValueAsString();
          if (F.hasFnAttribute("target-features"))
            BFeat = F.getFnAttribute("target-features").getValueAsString();
          break;
        }
      if (!BCPU.empty()) {
        llvm::SubtargetFeatures BF(BFeat);
        std::string BKey =
            canonicalLLVMISAKey(llvm::Triple(M.getTargetTriple()), BCPU, BF);
        if (BKey != It->first())
          luthier::errs()
              << "[DeviceToolCodeParser] WARNING: slice keyed by its bundle "
                 "ID as ["
              << It->first() << "] but its bitcode reports ISA [" << BKey
              << "]\n";
      }
    }
#endif
    return MOrErr;
  }

  // No precompiled slice matched. Try the SPIR-V JIT fallback (errors with a
  // helpful message when neither a SPIR-V slice nor the translator is
  // available).
  if (SpirvSlice)
    return translateSpirvFallback(T, CPU, Features, Key, Ctx, OptLevel);

  std::string AvailKeys;
  llvm::raw_string_ostream OS(AvailKeys);
  for (const auto &KV : Slices)
    OS << "  [" << KV.first() << "]\n";
  return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
      "No embedded bitcode cached for the requested LLVM ISA tuple, and no "
      "SPIR-V slice for the JIT fallback. Requested: [{0}]. Available ({1} "
      "slices):\n{2}",
      Key, Slices.size(), AvailKeys));
}

} // namespace luthier
