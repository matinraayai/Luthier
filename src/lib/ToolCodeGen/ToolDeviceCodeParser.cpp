//===-- ToolDeviceCodeParser.cpp ----------------------------------*-C++-*-===//
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
/// precompiled bitcode slice in \c ToolDeviceCodeParser.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/ToolDeviceCodeParser.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include <algorithm>
#include <cstring>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>
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
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <sstream>
#include <string>
#include <vector>
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
  LLVM_DEBUG(luthier::dbgs() << "[ToolDeviceCodeParser] parseOffloadBundle: "
                             << Bundle.getBufferSize() << " bytes\n");
  if (Bundle.getBufferSize() == 0)
    return LUTHIER_MAKE_GENERIC_ERROR("Empty fat-binary bundle.");

  auto Magic = llvm::identify_magic(Bundle.getBuffer());

  llvm::MemoryBufferRef ParseBuf = Bundle;
  bool Decompressed = false;
  if (Magic == llvm::file_magic::offload_bundle_compressed) {
    LLVM_DEBUG(luthier::dbgs() << "[ToolDeviceCodeParser] bundle is CCOB; "
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
  LLVM_DEBUG(luthier::dbgs() << "[ToolDeviceCodeParser] parseOffloadBundle "
                                "produced "
                             << SliceBufs.size() << " slice(s)\n");
  return llvm::Error::success();
}

} // namespace

std::string
ToolDeviceCodeParser::canonicalLLVMISAKey(const llvm::Triple &T,
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

llvm::Error ToolDeviceCodeParser::addSlice(llvm::MemoryBufferRef Slice,
                                           llvm::StringRef ID) {
  const llvm::file_magic Magic = llvm::identify_magic(Slice.getBuffer());
  if (Magic == llvm::file_magic::bitcode) {
    // Derive the LLVM ISA key from the slice's offload-bundle entry ID
    auto ISAOrErr = parseSliceISA(ID);
    if (!ISAOrErr)
      return ISAOrErr.takeError();
    auto &[TT, CPU, Features] = *ISAOrErr;

    std::string Key = canonicalLLVMISAKey(TT, CPU, Features);
    LLVM_DEBUG(luthier::dbgs() << "[ToolDeviceCodeParser] addBitcodeSlice id=["
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
               << "[ToolDeviceCodeParser] stashed SPIR-V slice ("
               << Slice.getBufferSize() << " bytes)\n");
    return llvm::Error::success();
  }
  return LUTHIER_MAKE_GENERIC_ERROR(
      "Fat-binary slice is neither LLVM bitcode nor SPIR-V.");
}

ToolDeviceCodeParser::ToolDeviceCodeParser(llvm::MemoryBufferRef BundleRef,
                                           llvm::Error &Err) {
  llvm::ErrorAsOutParameter EAO(&Err);
  if (Err)
    return;
  if (BundleRef.getBufferSize() == 0)
    return; // No bundle = no device-side logic.

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
             << "[ToolDeviceCodeParser] ctor(bundle): registered "
             << Slices.size() << " slice(s)" << (SpirvSlice ? " + SPIR-V" : "")
             << "\n");
}

ToolDeviceCodeParser::ToolDeviceCodeParser(
    std::unique_ptr<llvm::MemoryBuffer> Bundle, llvm::Error &Err)
    : ToolDeviceCodeParser(
          Bundle ? Bundle->getMemBufferRef() : llvm::MemoryBufferRef(), Err) {
  /// Take ownership of the bundle if it's not nullptr
  if (Bundle)
    RetainedBuffers.push_back(std::move(Bundle));
}

#ifdef LUTHIER_HAS_SPIRV_TRANSLATOR
/// Link the ROCm device-library bitcode needed to resolve the OpenCL builtins
/// (\c popcount, \c atom_add, …) that the reverse SPIR-V->IR translation leaves
/// as undefined external calls. This mirrors what a normal HIP compile does via
/// \c -mlink-builtin-bitcode, but in-process: each \c .bc is parsed into \p M
/// 's context and linked with \c LinkOnlyNeeded so only referenced symbols are
/// pulled in. Must run BEFORE the optimization pipeline so the builtins inline.
static llvm::Error linkDeviceLibs(llvm::Module &M, llvm::StringRef CPU,
                                  const llvm::SubtargetFeatures &Features) {
#ifndef LUTHIER_DEVICE_LIBS_DIR
#error "SPIR-V JIT path needs the ROCm device-libs bitcode."
#else
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::StringRef Dir(LUTHIER_DEVICE_LIBS_DIR);

  // wavefrontsize64 is on unless the target features explicitly disable it.
  bool Wave64 = true;
  for (const std::string &F : Features.getFeatures()) {
    if (F == "+wavefrontsize64")
      Wave64 = true;
    else if (F == "-wavefrontsize64")
      Wave64 = false;
  }

  // The ISA-version control lib is named by the bare processor
  // (gfx942 -> 942, gfx9-4-generic -> 9-4-generic).
  std::string IsaVer =
      CPU.starts_with("gfx") ? CPU.drop_front(3).str() : CPU.str();

  // Code-object / ABI version from the module flag clang stamps in; default to
  // the current COV (6) when absent.
  unsigned Cov = 600;
  if (llvm::Metadata *MD = M.getModuleFlag("amdhsa_code_object_version"))
    if (auto *CI = llvm::mdconst::dyn_extract<llvm::ConstantInt>(MD))
      Cov = CI->getZExtValue();

  const llvm::SmallVector<std::string, 8> Libs = {
      "opencl.bc",
      "ocml.bc",
      "ockl.bc",
      "oclc_isa_version_" + IsaVer + ".bc",
      Wave64 ? "oclc_wavefrontsize64_on.bc" : "oclc_wavefrontsize64_off.bc",
      "oclc_finite_only_off.bc",
      "oclc_unsafe_math_off.bc",
      "oclc_abi_version_" + std::to_string(Cov) + ".bc",
  };

  llvm::Linker L(M);
  for (const std::string &Lib : Libs) {
    llvm::SmallString<256> Path(Dir);
    llvm::sys::path::append(Path, Lib);
    llvm::SMDiagnostic Diag;
    std::unique_ptr<llvm::Module> Sub = llvm::parseIRFile(Path, Diag, Ctx);
    if (!Sub)
      return LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Failed to load device-libs bitcode '{0}': {1}",
                        Path.str().str(), Diag.getMessage().str()));
    // The device libs share our AMDGCN triple/datalayout; align them explicitly
    // to suppress spurious Linker mismatch diagnostics.
    Sub->setTargetTriple(M.getTargetTriple());
    Sub->setDataLayout(M.getDataLayout());
    if (L.linkInModule(std::move(Sub), llvm::Linker::Flags::LinkOnlyNeeded))
      return LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Failed to link device-libs bitcode '{0}'", Lib));
  }

  LLVM_DEBUG(luthier::dbgs()
             << "[ToolDeviceCodeParser] linked device-libs (isa=" << IsaVer
             << " wave64=" << Wave64 << " cov=" << Cov << ")\n");
  return llvm::Error::success();
#endif
}
#endif

llvm::Expected<std::unique_ptr<llvm::Module>>
ToolDeviceCodeParser::translateSpirvFallback(
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
             << "[ToolDeviceCodeParser] SPIR-V JIT fallback for [" << Key
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

  // Link against device libs
  if (llvm::Error E = linkDeviceLibs(*M, CPU, Features))
    return std::move(E);

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
             << "[ToolDeviceCodeParser] SPIR-V JIT produced + cached " << Key
             << " (" << BcRef.getBufferSize() << " bytes)\n");
  return M;
#endif
}

llvm::Expected<std::unique_ptr<llvm::Module>>
ToolDeviceCodeParser::parseModule(const llvm::Triple &T, llvm::StringRef CPU,
                                  const llvm::SubtargetFeatures &Features,
                                  llvm::LLVMContext &Ctx,
                                  llvm::OptimizationLevel OptLevel) {
  std::lock_guard Lock(Mutex);
  std::string Key = canonicalLLVMISAKey(T, CPU, Features);
  LLVM_DEBUG(luthier::dbgs()
             << "[ToolDeviceCodeParser] parseModule key=[" << Key << "]\n");
  auto It = Slices.find(Key);
  if (It != Slices.end()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[ToolDeviceCodeParser]   matched slice [" << It->first()
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
              << "[ToolDeviceCodeParser] WARNING: slice keyed by its bundle "
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
