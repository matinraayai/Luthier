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
/// \c .luthier files into a \c luthier::Prototype
/// — together with the \c writeLuthierFile helper for serialization.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TESTING_LUTHIER_FILE_H
#define LUTHIER_TOOL_CODE_GEN_TESTING_LUTHIER_FILE_H

#include "luthier/ToolCodeGen/Prototype.h"
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

/// \brief Parses a \c .luthier file and provides typed access to its
/// contents.
class LuthierFileParser {
public:
  /// One entry in the cross-module metadata slot map embedded in a
  /// \c .luthier file. Maps a metadata slot number in the instrumentation
  /// module to the slot number of the same \c MDNode in the target module.
  struct MDSlotEntry {
    unsigned IModuleSlot = 0;
    unsigned TargetSlot = 0;

    bool operator==(const MDSlotEntry &) const = default;
  };

  /// Encoding of a module field in a \c .luthier file.
  enum class ModuleFormat {
    IR,  ///< LLVM IR text (.ll)
    MIR, ///< Machine IR text (.mir)
  };

private:
  std::string Identifier;

  std::string TargetModuleText;

  std::string InstrumentationModuleText;

  ModuleFormat TargetModuleFormat = ModuleFormat::MIR;

  ModuleFormat InstrumentationModuleFormat = ModuleFormat::IR;

  std::vector<MDSlotEntry> MDSlotMap;

  std::unique_ptr<llvm::MIRParser> TargetMIRParser{nullptr};

  std::unique_ptr<llvm::MIRParser> InstrumentationMIRParser{nullptr};

public:
  //===--------------------------------------------------------------------===//
  // Factory
  //===--------------------------------------------------------------------===//

  /// Parses a \c .luthier file from \p Buffer.
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

  /// Parse the .luthier file into a prototype backed by the \p Ctx
  /// \p SetDataLayout and \p SetMIRFunctionAttributes are forwarded to the MIR
  /// parser if MIR is used to encode any of the modules.
  llvm::Expected<std::unique_ptr<Prototype>> loadPrototype(
      llvm::LLVMContext &Ctx,
      const std::function<std::optional<std::string>(
          llvm::StringRef, llvm::StringRef)> &SetDataLayout = nullptr,
      const std::function<void(llvm::Function &)> &SetMIRFunctionAttributes =
          nullptr);

  /// Loads the target and instrumentation module's MIR, if the .luthier file
  /// indicates they are present
  llvm::Error loadMIR(Prototype &P, PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

/// Serializes \p IP as a \c .luthier YAML file, writing the result to \p OS.
/// For each module, the writer picks a format automatically: if any of the
/// module's \c Function s has a cached \c llvm::MachineFunctionAnalysis
/// result on the \c FunctionAnalysisManager reachable from \p IPAM, the
/// module is written as MIR; otherwise it is written as LLVM IR text.
llvm::Error writeLuthierFile(llvm::raw_ostream &OS, Prototype &IP,
                             PrototypeAnalysisManager &IPAM);

/// Convenience overload that opens \p Path and delegates to the stream-based
/// \c writeLuthierFile.
llvm::Error writeLuthierFile(llvm::StringRef Path, Prototype &IP,
                             PrototypeAnalysisManager &IPAM);

} // namespace luthier

#endif
