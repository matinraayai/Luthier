//===-- ToolDeviceCodeParser.h -----------------------------------*- C++-*-===//
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
/// Defines the \c ToolDeviceCodeParser class, in charge of parsing and loading
/// a collection of device logic in LLVM IR bitcode or SPIR-V format that
/// belong to a single tool translation unit.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODEGEN_TOOL_DEVICE_CODE_PARSER_H
#define LUTHIER_TOOL_CODEGEN_TOOL_DEVICE_CODE_PARSER_H
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
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
#include <vector>

namespace luthier {

/// \brief A class in charge of parsing and loading a collection of device
/// logic in LLVM IR bitcode or SPIR-V format that belong to a single tool
/// translation unit.
///
/// \details As of right now, this class only accepts clang offload bundle files
/// (i.e. both compressed and uncompressed FAT binaries). Each bitcode entry's
/// LLVM ISA (triple, CPU, and subtarget features) is read directly from the
/// bitcode itself at construction.
///
/// When requested, this class returns loads the bitcode of a slice that is
/// compatible with the requested ISA into an \c llvm::Module. In cases
/// where no precompiled slice is compatible and Luthier is compiled with AMD
/// SPIR-V translation support, presence of an AMD-flavored SPIR-V slice is
/// queried in the bundle cache. If present, the parser will JIT-translate the
/// SPIR-V to LLVM IR for the requested target, runs the Luthier device tool
/// compilation passes on it, and caches the result before returning the
/// materialized module.
///
/// TODO: Support managing separately provided files for tools that don't use
/// HIP
/// TODO: Flesh out the SPIR-V support for graphics AMD triples as well
class ToolDeviceCodeParser {
protected:
  /// Mutex to protect internal state of slices.
  std::recursive_mutex Mutex;

  struct SliceInfo {
    llvm::Triple TT;
    std::string CPU;
    llvm::SubtargetFeatures Features;
    llvm::MemoryBufferRef Bitcode;
  };

  /// All precompiled bitcode slices, along with their parsed ISAs
  llvm::SmallVector<SliceInfo, 4> Slices;

  /// AMD-flavored SPIR-V slice (\c hip-spirv64-amd-amdhsa--amdgcnspirv), if the
  /// bundle carried one. Used by the SPIR-V → AMDGCN JIT fallback.
  std::optional<llvm::MemoryBufferRef> SpirvSlice;

  /// Bundles, decompressed payloads, and JIT-produced bitcode owned for this
  /// object's lifetime
  llvm::SmallVector<std::unique_ptr<llvm::MemoryBuffer>, 2> RetainedBuffers;

  /// Register one fat-binary slice, dispatching on its leading magic. For an
  /// LLVM bitcode slice, the triple/CPU/features are read from the bitcode
  /// itself and a \c SliceInfo is appended to \c Slices; a SPIR-V slice is
  /// stashed in \c SpirvSlice.
  /// \p ID is the slice's Clang offload-bundle entry ID.
  /// \returns \c llvm::Error when the bitcode carries no target-cpu or the
  /// slice is neither bitcode nor SPIR-V.
  llvm::Error addSlice(llvm::MemoryBufferRef Slice, llvm::StringRef ID);

  /// Find a precompiled slice whose ISA is compatible with the requested one.
  /// Returns \c nullptr when no slice is compatible. Caller must hold \c Mutex.
  llvm::Expected<const SliceInfo *>
  findCompatibleSlice(const llvm::Triple &T, llvm::StringRef CPU,
                      const llvm::SubtargetFeatures &Features);

  /// SPIR-V -> AMDGCN JIT fallback. Translates \c SpirvSlice to LLVM IR for the
  /// requested ISA, runs an O3 default pipeline + the Luthier device tool
  /// passes, caches the serialized bitcode under \p Key, and returns the
  /// freshly built module. Caller must hold \c Mutex. Errors if no SPIR-V slice
  /// is present or the translator was not built into this binary.
  llvm::Expected<std::unique_ptr<llvm::Module>> translateSpirvFallback(
      const llvm::Triple &T, llvm::StringRef CPU,
      const llvm::SubtargetFeatures &Features, llvm::LLVMContext &Ctx,
      llvm::OptimizationLevel OptLevel = llvm::OptimizationLevel::O3);

public:
  /// Owning buffer constructor: Call this when the parser should take over the
  /// lifetime of \p Bundle
  ToolDeviceCodeParser(std::unique_ptr<llvm::MemoryBuffer> Bundle,
                       llvm::Error &Err);

  /// Non-owning buffer constructor: Call this when the \p Bundle's lifetime
  /// is externally managed
  ToolDeviceCodeParser(llvm::MemoryBufferRef Bundle, llvm::Error &Err);

  ToolDeviceCodeParser(const ToolDeviceCodeParser &) = delete;
  ToolDeviceCodeParser &operator=(const ToolDeviceCodeParser &) = delete;

  ~ToolDeviceCodeParser() = default;

  /// Parse the embedded tool bitcode for the requested LLVM ISA tuple into
  /// \p Ctx. Returns the bitcode of a slice compatible with the requested ISA
  /// (see \c findCompatibleSlice). When no slice is compatible, falls back to a
  /// SPIR-V → AMDGCN JIT translation for the requested ISA (if a SPIR-V slice
  /// is present and the translator is available). \p OptLevel indicates the
  /// optimization level used to compile the SPIR-V slice
  llvm::Expected<std::unique_ptr<llvm::Module>>
  parseModule(const llvm::Triple &T, llvm::StringRef CPU,
              const llvm::SubtargetFeatures &Features, llvm::LLVMContext &Ctx,
              llvm::OptimizationLevel OptLevel = llvm::OptimizationLevel::O3);
};

} // namespace luthier

#endif // LUTHIER_TOOL_CODEGEN_DEVICE_TOOL_CODE_PARSER_H
