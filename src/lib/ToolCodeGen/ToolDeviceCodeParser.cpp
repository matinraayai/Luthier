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
#include <Utils/AMDGPUBaseInfo.h>
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
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/OffloadBundle.h>
#include <llvm/Support/CrashRecoveryContext.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
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

llvm::Error ToolDeviceCodeParser::addSlice(llvm::MemoryBufferRef Slice,
                                           llvm::StringRef ID) {
  const llvm::file_magic Magic = llvm::identify_magic(Slice.getBuffer());
  if (Magic == llvm::file_magic::bitcode) {
    // Read the slice's LLVM ISA straight from its bitcode rather than trusting
    // the bundle entry ID label: the triple from the module header, and the
    // CPU/features from the per-function target-cpu/target-features attributes
    // clang stamps. Parse into a throwaway local context so the module's reader
    // does not outlive this scope / alias the retained bundle buffers.
    llvm::Expected<std::string> TripleOrErr =
        llvm::getBitcodeTargetTriple(Slice);
    if (!TripleOrErr)
      return TripleOrErr.takeError();
    llvm::Triple TT(*TripleOrErr);

    std::string CPU;
    llvm::SubtargetFeatures Features;
    {
      llvm::LLVMContext ScanCtx;
      llvm::Expected<std::unique_ptr<llvm::Module>> MOrErr =
          llvm::parseBitcodeFile(Slice, ScanCtx);
      if (!MOrErr)
        return MOrErr.takeError();
      const llvm::Module &M = **MOrErr;
      for (const llvm::Function &F : M) {
        if (F.hasFnAttribute("target-cpu")) {
          CPU = F.getFnAttribute("target-cpu").getValueAsString().str();
          if (F.hasFnAttribute("target-features"))
            Features = llvm::SubtargetFeatures(
                F.getFnAttribute("target-features").getValueAsString());
          break;
        }
      }
    }
    if (CPU.empty())
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Bitcode slice '" + ID.str() +
          "' carries no function with a target-cpu attribute; cannot determine "
          "its ISA.");

    LLVM_DEBUG(luthier::dbgs() << "[ToolDeviceCodeParser] addBitcodeSlice id=["
                               << ID << "] isa=[" << TT.str() << "-" << CPU
                               << ":" << Features.getString()
                               << "] bcSize=" << Slice.getBufferSize() << "\n");

    Slices.push_back(
        {std::move(TT), std::move(CPU), std::move(Features), Slice});
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

llvm::Expected<const ToolDeviceCodeParser::SliceInfo *>
ToolDeviceCodeParser::findCompatibleSlice(
    const llvm::Triple &T, llvm::StringRef CPU,
    const llvm::SubtargetFeatures &Features) {
  std::string TgtErr;
  const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget(T, TgtErr);
  if (TheTarget == nullptr)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "TargetRegistry::lookupTarget failed for triple {0}: {1} (is the "
        "AMDGPU target registered?)",
        T.str(), TgtErr));

  const std::string ReqFS = Features.getString();
  std::unique_ptr<llvm::MCSubtargetInfo> ReqSTI(
      TheTarget->createMCSubtargetInfo(T, CPU, ReqFS));
  if (!ReqSTI)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "createMCSubtargetInfo returned nullptr for the requested ISA.");

  namespace IsaInfo = llvm::AMDGPU::IsaInfo;
  IsaInfo::AMDGPUTargetID ReqID(*ReqSTI);
  ReqID.setTargetIDFromFeaturesString(ReqFS);

  /// lambda used to check for compatibility between the ISAs regarding xnack
  /// and sramecc features
  auto featureCompat = [](IsaInfo::TargetIDSetting Req,
                          IsaInfo::TargetIDSetting Slice) {
    using S = IsaInfo::TargetIDSetting;
    if (Req == S::Unsupported || Slice == S::Unsupported || Req == S::Any ||
        Slice == S::Any)
      return true;
    return Req == Slice;
  };

  /// Features that must exactly match
  static constexpr llvm::StringLiteral BinaryFeatures[] = {
      "+wavefrontsize64", "+cumode", "+tgsplit"};

  for (const SliceInfo &S : Slices) {
    if (S.TT != T || S.CPU != CPU)
      continue;

    const std::string SliceFS = S.Features.getString();
    std::unique_ptr<llvm::MCSubtargetInfo> SliceSTI(
        TheTarget->createMCSubtargetInfo(S.TT, S.CPU, SliceFS));
    if (!SliceSTI)
      continue;

    IsaInfo::AMDGPUTargetID SliceID(*SliceSTI);
    SliceID.setTargetIDFromFeaturesString(SliceFS);
    if (!featureCompat(ReqID.getXnackSetting(), SliceID.getXnackSetting()))
      continue;
    if (!featureCompat(ReqID.getSramEccSetting(), SliceID.getSramEccSetting()))
      continue;

    bool BinaryMismatch = false;
    for (llvm::StringRef Feat : BinaryFeatures) {
      if (ReqSTI->checkFeatures(Feat) != SliceSTI->checkFeatures(Feat)) {
        BinaryMismatch = true;
        break;
      }
    }
    if (BinaryMismatch)
      continue;

    return &S;
  }
  return nullptr;
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
    const llvm::SubtargetFeatures &Features, llvm::LLVMContext &Ctx,
    llvm::OptimizationLevel OptLevel) {
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
             << "[ToolDeviceCodeParser] SPIR-V JIT fallback for [" << T.str()
             << "-" << CPU << ":" << Features.getString() << "]\n");

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
      std::move(BcBuf), "luthier.spirv.jit",
      /*RequiresNullTerminator=*/false);
  llvm::MemoryBufferRef BcRef = Owned->getMemBufferRef();
  RetainedBuffers.push_back(std::move(Owned));

  /// Put the JIT-compiled slice at the front since it is likely to be asked for
  /// again
  Slices.insert(Slices.begin(), SliceInfo{T, CPU.str(), Features, BcRef});

  LLVM_DEBUG(luthier::dbgs()
             << "[ToolDeviceCodeParser] SPIR-V JIT produced + cached "
             << T.str() << "-" << CPU << ":" << Features.getString() << " ("
             << BcRef.getBufferSize() << " bytes)\n");
  return M;
#endif
}

llvm::Expected<std::unique_ptr<llvm::Module>>
ToolDeviceCodeParser::parseModule(const llvm::Triple &T, llvm::StringRef CPU,
                                  const llvm::SubtargetFeatures &Features,
                                  llvm::LLVMContext &Ctx,
                                  llvm::OptimizationLevel OptLevel) {
  std::lock_guard Lock(Mutex);
  LLVM_DEBUG(luthier::dbgs()
             << "[ToolDeviceCodeParser] parseModule ISA=[" << T.str() << "-"
             << CPU << ":" << Features.getString() << "]\n");

  // Otherwise look for a precompiled slice whose bitcode ISA is compatible with
  // the requested one.
  llvm::Expected<const SliceInfo *> SliceOrErr =
      findCompatibleSlice(T, CPU, Features);
  if (!SliceOrErr)
    return SliceOrErr.takeError();
  if (const SliceInfo *S = *SliceOrErr) {
    LLVM_DEBUG(luthier::dbgs()
               << "[ToolDeviceCodeParser]   matched slice [" << S->TT.str()
               << "-" << S->CPU << ":" << S->Features.getString()
               << "], parsing " << S->Bitcode.getBufferSize()
               << " bytes of bitcode\n");
    return llvm::parseBitcodeFile(S->Bitcode, Ctx);
  }

  // No compatible slice. Try the SPIR-V JIT fallback
  if (SpirvSlice)
    return translateSpirvFallback(T, CPU, Features, Ctx, OptLevel);

  std::string AvailKeys;
  llvm::raw_string_ostream OS(AvailKeys);
  for (const SliceInfo &S : Slices)
    OS << "  [" << S.TT.str() << "-" << S.CPU << ":" << S.Features.getString()
       << "]\n";
  return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
      "No embedded bitcode compatible with the requested LLVM ISA tuple, and "
      "no SPIR-V slice for the JIT fallback. Requested: [{0}-{1}:{2}]. "
      "Available ({3} "
      "slices):\n{4}",
      T.str(), CPU, Features.getString(), Slices.size(), AvailKeys));
}

} // namespace luthier
