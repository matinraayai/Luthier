//===-- ToolDeviceCodeOffloadParserTest.hip ---------------------*- C++ -*-===//
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
/// Unit test for the \c ToolDeviceCodeParser and its
/// \c ToolDeviceCodeOffloadParserTrait. Tests every ISA backed into the bundle
/// by the CMake offload bundle utility. When the AMD SPIR-V translator is
/// available, the test also requests ISAs not precompiled in the bundle to
/// test the SPIR-V translation path.
//===----------------------------------------------------------------------===//
#undef SPIRV

#include <hip/hip_runtime.h>

#include <luthier/Common/GenericLuthierError.h>
#include <luthier/ToolCodeGen/ToolDeviceCodeOffloadParser.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>

#include <memory>
#include <string>
#include <utility>

template <typename T, int Dim>
struct __attribute__((device_builtin_surface_type)) LuthierSurface {};

class EmbeddedBitcodeLoaderTestTool
    : public luthier::ToolDeviceCodeOffloadParserTrait<
          EmbeddedBitcodeLoaderTestTool> {

  __global__ static void testKernel() { DeviceVar = ManagedVar; }

  __attribute__((device)) static int DeviceVar;

  __managed__ static int ManagedVar;

  static texture<float, 1, hipReadModeElementType> TexRef;

  __attribute__((device)) static LuthierSurface<void, 2> SurfRef;

  __attribute__((device, used)) static void testDeviceFunc();

public:
  using Base =
      luthier::ToolDeviceCodeOffloadParserTrait<EmbeddedBitcodeLoaderTestTool>;

  explicit EmbeddedBitcodeLoaderTestTool(llvm::Error &Err) : Base(Err) {}

  /// Load the embedded bitcode for the requested ISA into \p Ctx and confirm
  /// that every registered device handle resolves to a symbol present in the
  /// loaded module.
  ///
  /// Every handle must always \e resolve (the host handle table is populated at
  /// compile time, independent of the requested ISA). \param RequireAllHandles
  /// additionally requires each resolved symbol to be \e present in the loaded
  /// module. It is \c true on the precompiled-slice path; on the SPIR-V JIT
  /// fallback path it is \c false, since that path internalizes and optimizes
  /// the module, so only the \c used device function is guaranteed to survive
  /// (an unreferenced kernel / global / texture / surface may be eliminated).
  llvm::Error loadAndVerify(const llvm::Triple &T, llvm::StringRef CPU,
                            const llvm::SubtargetFeatures &Features,
                            llvm::LLVMContext &Ctx, bool RequireAllHandles);
};

//===----------------------------------------------------------------------===//
// Out-of-line device definitions of the static members.
//===----------------------------------------------------------------------===//
LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES(EmbeddedBitcodeLoaderTestTool)

__attribute__((device)) int EmbeddedBitcodeLoaderTestTool::DeviceVar = 0;

__managed__ int EmbeddedBitcodeLoaderTestTool::ManagedVar = 0;

texture<float, 1, hipReadModeElementType> EmbeddedBitcodeLoaderTestTool::TexRef;

__attribute__((device))
LuthierSurface<void, 2> EmbeddedBitcodeLoaderTestTool::SurfRef;

__attribute__((device, used)) void
EmbeddedBitcodeLoaderTestTool::testDeviceFunc() {
  DeviceVar += 1;
}

llvm::Error EmbeddedBitcodeLoaderTestTool::loadAndVerify(
    const llvm::Triple &T, llvm::StringRef CPU,
    const llvm::SubtargetFeatures &Features, llvm::LLVMContext &Ctx,
    bool RequireAllHandles) {
  auto ModOrErr = parseModule(T, CPU, Features, Ctx);
  if (!ModOrErr)
    return ModOrErr.takeError();
  std::unique_ptr<llvm::Module> M = std::move(*ModOrErr);

  std::string Failures;
  llvm::raw_string_ostream OS(Failures);

  // Resolve one host shadow handle to its device symbol name (always required)
  // and, when \p RequirePresence is set, confirm the symbol exists in the
  // loaded module.
  auto checkHandle = [&](auto *Handle, llvm::StringRef Label,
                         bool RequirePresence) {
    auto NameOrErr = this->lookupHandleName(Handle);
    if (!NameOrErr) {
      OS << "  - " << Label << ": no device-side handle registered ("
         << llvm::toString(NameOrErr.takeError()) << ")\n";
      return;
    }
    llvm::StringRef Name = *NameOrErr;
    if (RequirePresence && !M->getNamedValue(Name))
      OS << "  - " << Label << ": device symbol '" << Name
         << "' is not present in the loaded module\n";
  };

  // The device function is marked `used`, so it survives the SPIR-V fallback's
  // internalize + optimize pipeline: its presence is required on both paths.
  checkHandle(&EmbeddedBitcodeLoaderTestTool::testDeviceFunc, "device function",
              true);
  // Kernel / globals / texture / surface: presence required only on the
  // precompiled path (they may be eliminated on the SPIR-V fallback path).
  checkHandle(&EmbeddedBitcodeLoaderTestTool::testKernel, "kernel",
              RequireAllHandles);
  checkHandle(&EmbeddedBitcodeLoaderTestTool::DeviceVar, "__device__ variable",
              RequireAllHandles);
  checkHandle(&EmbeddedBitcodeLoaderTestTool::ManagedVar, "__managed__ variable",
              RequireAllHandles);
  checkHandle(&EmbeddedBitcodeLoaderTestTool::TexRef, "texture",
              RequireAllHandles);
  checkHandle(&EmbeddedBitcodeLoaderTestTool::SurfRef, "surface",
              RequireAllHandles);

  if (!Failures.empty())
    return LUTHIER_MAKE_GENERIC_ERROR("Handle verification failed:\n" +
                                      Failures);
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Test harness. Host-only: skipped during the HIP device-compilation passes so
// the device slices never need the GoogleTest headers.
//===----------------------------------------------------------------------===//
#ifndef __HIP_DEVICE_COMPILE__
#include <gtest/gtest.h>
#include <llvm/Support/TargetSelect.h>

namespace {

/// The ISAs precompiled into the bundle by the CMake \c TARGET_ISAS list. Keep
/// this in lockstep with tooldevicecodeparser/CMakeLists.txt.
const std::tuple<llvm::Triple, std::string, llvm::SubtargetFeatures> PrecompiledISAs[] = {
    {llvm::Triple{"amdgcn-amd-amdhsa-"}, "gfx900", llvm::SubtargetFeatures{"-xnack"}},
    {llvm::Triple{"amdgcn-amd-amdhsa-"}, "gfx908", llvm::SubtargetFeatures{"+xnack,-sramecc"}},
    {llvm::Triple{"amdgcn-amd-amdhsa-"}, "gfx90a", llvm::SubtargetFeatures{"-xnack,+sramecc"}},
    {llvm::Triple{"amdgcn-amd-amdhsa-"}, "gfx942", llvm::SubtargetFeatures{"+xnack,+sramecc"}},
    {llvm::Triple{"amdgcn-amd-amdhsa-"}, "gfx1030",
     llvm::SubtargetFeatures{"+wavefrontsize32,-cumode"}},
    {llvm::Triple{"amdgcn-amd-amdhsa-"}, "gfx1100",
     llvm::SubtargetFeatures{"+wavefrontsize64,+cumode"}},
};

class EmbeddedBitcodeLoaderTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
  }
};

TEST_F(EmbeddedBitcodeLoaderTest, LoadsEveryPrecompiledISA) {
  llvm::Error CtorErr = llvm::Error::success();
  EmbeddedBitcodeLoaderTestTool Tool(CtorErr);
  if (CtorErr)
    FAIL() << "Tool construction failed: "
           << llvm::toString(std::move(CtorErr));

  for (const auto &[TT, CPU, STF] : PrecompiledISAs) {
    llvm::LLVMContext Ctx;
    if (llvm::Error Err = Tool.loadAndVerify(TT, CPU, STF, Ctx,
                                             /*RequireAllHandles=*/true))
      ADD_FAILURE() << "ISA " << CPU << ": "
                    << llvm::toString(std::move(Err));
  }
}

#ifdef LUTHIER_HAS_SPIRV_TRANSLATOR
TEST_F(EmbeddedBitcodeLoaderTest, SpirvFallbackForNonPrecompiledISA) {
  llvm::Error CtorErr = llvm::Error::success();
  EmbeddedBitcodeLoaderTestTool Tool(CtorErr);
  if (CtorErr)
    FAIL() << "Tool construction failed: "
           << llvm::toString(std::move(CtorErr));

  // Compile an ISA not present in the pre-compiled slices.
  llvm::LLVMContext Ctx;
  llvm::Triple T{"amdgcn-amd-amdhsa-"};
  llvm::SubtargetFeatures Features{};
  if (llvm::Error Err = Tool.loadAndVerify(T, "gfx906", Features, Ctx,
                                           /*RequireAllHandles=*/false))
    FAIL() << "SPIR-V fallback failed for gfx906: "
           << llvm::toString(std::move(Err));
}
#endif

} // namespace
#endif // __HIP_DEVICE_COMPILE__
