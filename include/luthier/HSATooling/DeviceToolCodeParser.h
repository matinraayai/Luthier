//===-- DeviceToolCodeParser.h -----------------------------------*- C++-*-===//
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
/// Defines the \c DeviceToolCodeParser, which parses a FAT binary bundle
/// containing the device-side logic of a single luthier tool in LLVM IR bitcode
/// or SPIR-V format and loads them into \c llvm::Module instances for use in
/// the instrumentation pipeline.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_DEVICE_TOOL_CODE_PARSER_H
#define LUTHIER_TOOLING_DEVICE_TOOL_CODE_PARSER_H
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <llvm/TargetParser/Triple.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace luthier {

/// \brief Parses a FAT binary bundle containing the device-side logic of a
/// single Luthier tool and serves a slice with the matching target ISA as a
/// \c llvm::Module for the instrumentation pipeline.
///
/// \details The bundle contains the associated LLVM IR bitcode for each full
/// ISA string to be targeted for instrumentation. If SPIR-V translation support
/// is enabled, the bundle can also contain the SPIR-V file of the translation
/// unit to fallback on if the exact target ISA is absent in the bundle.
///
/// Each slice is put in a cache keyed by canonical LLVM ISA tuple (ISA with
/// its subtarget feature flags sorted). On request, this class loads the
/// matching slice's bitcode as an \c llvm::Module. In cases where no
/// precompiled slice matches the requested ISA and Luthier is compiled with
/// AMD SPIR-V translation support, presence of an AMD-flavored SPIR-V slice is
/// queried in the bundle cache. If present, the parser will JIT-translate the
/// SPIR-V to LLVM IR for the requested target, runs the Luthier device tool
/// compilation passes on it, and caches the result before returning it.
///
/// \note This class only handles the device code of one source (TU) compiled
/// for multiple GPU targets. For multiple TUs, use multiple instances.
class DeviceToolCodeParser {
protected:
  /// Guards \c Slices and the SPIR-V JIT cache insertion. Recursive so a
  /// derived class can re-enter through \c getEmbeddedModule.
  std::recursive_mutex Mutex;

  /// All slices, keyed by canonical LLVM ISA string. Populated at construction;
  /// SPIR-V JIT fallbacks insert additional entries lazily.
  llvm::StringMap<llvm::MemoryBufferRef> Slices;

  /// AMD-flavored SPIR-V slice (\c hip-spirv64-amd-amdhsa--amdgcnspirv), if the
  /// bundle carried one. Used by the SPIR-V → AMDGCN JIT fallback. A non-owning
  /// view into \c RetainedBuffers.
  std::optional<llvm::MemoryBufferRef> SpirvSlice;

  /// Bundles, decompressed payloads, and JIT-produced bitcode owned for this
  /// object's lifetime so \c Slices' / \c SpirvSlice's views stay valid.
  llvm::SmallVector<std::unique_ptr<llvm::MemoryBuffer>, 2> RetainedBuffers;

  /// Canonical hashable key for an LLVM ISA tuple. Deterministic: features are
  /// sorted before stringification.
  static std::string canonicalLLVMISAKey(const llvm::Triple &T,
                                         llvm::StringRef CPU,
                                         const llvm::SubtargetFeatures &F);

  /// Determine a Clang offload bundle's total byte extent from its in-memory
  /// header alone, handling both the uncompressed (\c __CLANG_OFFLOAD_BUNDLE__)
  /// and compressed (\c CCOB) formats via LLVM's offload-bundle parsers
  static uint64_t discoverBundleSize(const void *Bundle);

  /// Insert a new \c Slices entry (or stash the SPIR-V slice) for one
  /// fat-binary slice, dispatching on its leading magic (LLVM bitcode or
  /// SPIR-V). \p ID is the slice's Clang offload-bundle entry ID, from which a
  /// bitcode slice's LLVM ISA key is derived (the bitcode is not parsed here).
  /// Errors on a malformed ID, duplicate ISA, or an unrecognized slice.
  llvm::Error addSlice(llvm::MemoryBufferRef Slice, llvm::StringRef ID);

  /// SPIR-V → AMDGCN JIT fallback. Translates \c SpirvSlice to LLVM IR for the
  /// requested ISA, runs an O3 default pipeline + the Luthier device tool
  /// passes, caches the serialized bitcode under \p Key, and returns the
  /// freshly built module. Caller must hold \c Mutex. Errors if no SPIR-V slice
  /// is present or the translator was not built into this binary.
  llvm::Expected<std::unique_ptr<llvm::Module>> translateSpirvFallback(
      const llvm::Triple &T, llvm::StringRef CPU,
      const llvm::SubtargetFeatures &Features, llvm::StringRef Key,
      llvm::LLVMContext &Ctx,
      llvm::OptimizationLevel OptLevel = llvm::OptimizationLevel::O3);

public:
  /// Bundle-path constructor: takes ownership of \p Bundle and parses it as a
  /// Clang FAT binary offload bundle. Sets \p Err on parse failure or if two
  /// slices share the same LLVM ISA. A null \p Bundle is a legitimate
  /// "host-only tool" (no slices).
  DeviceToolCodeParser(std::unique_ptr<llvm::MemoryBuffer> Bundle,
                       llvm::Error &Err);

  DeviceToolCodeParser(const DeviceToolCodeParser &) = delete;
  DeviceToolCodeParser &operator=(const DeviceToolCodeParser &) = delete;

  ~DeviceToolCodeParser() = default;

  /// Parse the embedded tool bitcode for the requested LLVM ISA tuple into
  /// \p Ctx. On an exact-key miss, falls back to a SPIR-V → AMDGCN JIT
  /// translation for the requested ISA (if a SPIR-V slice is present and the
  /// translator is available). \p OptLevel indicates the optimization level
  /// used to compile the SPIR-V slice
  llvm::Expected<std::unique_ptr<llvm::Module>>
  parseModule(const llvm::Triple &T, llvm::StringRef CPU,
              const llvm::SubtargetFeatures &Features, llvm::LLVMContext &Ctx,
              llvm::OptimizationLevel OptLevel = llvm::OptimizationLevel::O3);
};

} // namespace luthier

#endif // LUTHIER_TOOLING_DEVICE_TOOL_CODE_PARSER_H
