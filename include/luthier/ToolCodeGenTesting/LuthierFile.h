//===-- LuthierFile.h -------------------------------------------*- C++ -*-===//
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
/// Defines \c LuthierFileParser — the class responsible for deserializing
/// \c .luthier files into a \c luthier::InstrumentPrototype
/// — together with the \c writeLuthierFile helper for serialization.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TESTING_LUTHIER_FILE_H
#define LUTHIER_TOOL_CODE_GEN_TESTING_LUTHIER_FILE_H

#include "luthier/ToolCodeGen/InstrumentPrototype.h"
#include <functional>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/CodeGen/MIRParser/MIRParser.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class Function;
class LLVMContext;
class Module;
class raw_ostream;
} // namespace llvm

namespace luthier {

/// Result of parsing a \c .luthier file: the assembled
/// \c InstrumentPrototype together with the \c MIRParser instances used to
/// build each module (non-null only for modules stored in MIR form).
/// Callers hold on to the parsers until \c MachineModuleAnalysis has been
/// wired up so they can call \c MIRParser::parseMachineFunctions.
struct LoadedInstrumentPrototype {
  std::unique_ptr<InstrumentPrototype> IP;
  std::unique_ptr<llvm::MIRParser> TargetMIRParser;
  std::unique_ptr<llvm::MIRParser> IModuleMIRParser;
};

/// Parses a \c .luthier YAML file and provides typed access to its contents.
class LuthierFileParser {
public:
  /// One entry in the cross-module metadata slot map embedded in a
  /// \c .luthier file.  Maps a metadata slot number in the instrumentation
  /// module to the slot number of the same \c MDNode in the target module.
  struct MDSlotEntry {
    unsigned IModuleSlot = 0;
    unsigned TargetSlot = 0;

    bool operator==(const MDSlotEntry &) const = default;
  };

  /// Encoding of a module field in a \c .luthier file.
  enum class ModuleFormat {
    IR,      ///< LLVM IR text (.ll)
    Bitcode, ///< LLVM bitcode, base64-encoded in the YAML block scalar
    MIR,     ///< Machine IR text (.mir)
  };

  //===--------------------------------------------------------------------===//
  // Factory
  //===--------------------------------------------------------------------===//

  /// Parses a \c .luthier file from \p Buffer.  The buffer identifier is used
  /// in error messages.
  static llvm::Expected<LuthierFileParser> create(llvm::MemoryBufferRef Buffer);

  /// Parses the \c .luthier file at \p Path.
  static llvm::Expected<LuthierFileParser> create(llvm::StringRef Path);

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  [[nodiscard]] llvm::StringRef getTargetModule() const {
    return TargetModuleText;
  }
  [[nodiscard]] llvm::StringRef getInstrumentationModule() const {
    return InstrumentationModuleText;
  }
  [[nodiscard]] ModuleFormat getTargetModuleFormat() const {
    return TargetModuleFormat;
  }
  [[nodiscard]] ModuleFormat getInstrumentationModuleFormat() const {
    return InstrumentationModuleFormat;
  }
  llvm::ArrayRef<MDSlotEntry> getMDSlotMap() const { return MDSlotMap; }

  //===--------------------------------------------------------------------===//
  // Prototype loading
  //===--------------------------------------------------------------------===//

  /// Parse both modules of the \c .luthier file into a single
  /// \c InstrumentPrototype.
  ///
  /// \p Ctx is the \c LLVMContext both parsed modules share.  \p IPAM is
  /// threaded through for future use (analyses that need to associate MIR
  /// parsing state with the returned prototype); the reader itself does
  /// not currently register anything on it.
  ///
  /// \p SetDataLayout is forwarded to the target module's MIR parser (used
  /// by \c luthier-llc to override the data layout and initialize a
  /// \c TargetMachine).  \p SetMIRFunctionAttributes is applied to every
  /// \c Function in a module parsed from MIR.  Both callbacks default to
  /// no-ops if omitted.
  ///
  /// The instrumentation module's metadata is patched so that cross-module
  /// \c MDNode references point back into the live target module.
  llvm::Expected<LoadedInstrumentPrototype>
  load(llvm::LLVMContext &Ctx, InstrumentPrototypeAnalysisManager &IPAM,
       std::function<std::optional<std::string>(llvm::StringRef,
                                                llvm::StringRef)>
           SetDataLayout = nullptr,
       std::function<void(llvm::Function &)> SetMIRFunctionAttributes =
           nullptr) const;

  /// Parse only the instrumentation-module half of the \c .luthier file
  /// against the caller's live \p TargetModule.  The embedded MDNode slot
  /// map is applied to \p TargetModule (not to the file's serialized
  /// target module), so \p TargetModule must have the same metadata layout
  /// the file was written against.  Returns the parsed instrumentation
  /// module together with its \c MIRParser (non-null iff the module was
  /// stored in MIR form).
  llvm::Expected<std::pair<std::unique_ptr<llvm::Module>,
                           std::unique_ptr<llvm::MIRParser>>>
  loadIModule(llvm::LLVMContext &Ctx, llvm::Module &TargetModule) const;

private:
  std::string TargetModuleText;
  std::string InstrumentationModuleText;
  ModuleFormat TargetModuleFormat = ModuleFormat::MIR;
  ModuleFormat InstrumentationModuleFormat = ModuleFormat::IR;
  std::vector<MDSlotEntry> MDSlotMap;
};

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

/// Serializes \p IP as a \c .luthier YAML file, writing the result to \p OS.
/// For each module, the writer picks a format automatically: if any of the
/// module's \c Function s has a cached \c llvm::MachineFunctionAnalysis
/// result on the \c FunctionAnalysisManager reachable from \p IPAM, the
/// module is written as MIR; otherwise it is written as LLVM IR text.
llvm::Error writeLuthierFile(llvm::raw_ostream &OS, InstrumentPrototype &IP,
                             InstrumentPrototypeAnalysisManager &IPAM);

/// Convenience overload that opens \p Path and delegates to the stream-based
/// \c writeLuthierFile.
llvm::Error writeLuthierFile(llvm::StringRef Path, InstrumentPrototype &IP,
                             InstrumentPrototypeAnalysisManager &IPAM);

/// Compatibility shim for legacy callers that hold the two modules
/// separately and do not have an \c InstrumentPrototypeAnalysisManager.
/// Both modules are written as IR text; the MDNode slot map is still
/// computed against the live \p TargetModule so that reloaded imodule
/// metadata can be rewired.
llvm::Error writeLuthierFile(llvm::StringRef Path, llvm::Module &TargetModule,
                             llvm::Module &IModule);

} // namespace luthier

#endif
