//===-- InstrumentationModule.h ---------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// Describes the \c InstrumentationModule class, which encapsulates a single
/// instrumentation shared object file with its instrumentation LLVM bitcode
/// embedded inside it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INSTRUMENTATION_MODULE_H
#define LUTHIER_TOOLING_INSTRUMENTATION_MODULE_H
#include <llvm/ADT/StringMap.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/Error.h>
#include <luthier/object/ELFObjectUtils.h>
#include <luthier/object/ObjectUtils.h>

namespace luthier {

/// \brief Encapsulates a Luthier instrumentation module, consisting of
/// a single instrumentation shared object file with its instrumentation
/// LLVM bitcode embedded inside its \c IModuleBCSectionName section
class InstrumentationModule {
  /// A host copy of the code object used to load the module
  const std::vector<uint8_t> CodeObject;

  /// Parsed representation of the \c CodeObject
  const std::unique_ptr<llvm::object::ObjectFile> ObjectFile;

  /// The LLVM bitcode buffer inside the \c CodeObject
  const std::unique_ptr<llvm::MemoryBuffer> BCBuffer;

  InstrumentationModule(std::vector<uint8_t> CodeObject,
                        std::unique_ptr<llvm::object::ObjectFile> ObjectFile,
                        std::unique_ptr<llvm::MemoryBuffer> BCBuffer)
      : CodeObject(std::move(CodeObject)), ObjectFile(std::move(ObjectFile)),
        BCBuffer(std::move(BCBuffer)) {}

public:
  /// Creates an instance of \c InstrumentationModule from the passed Luthier
  /// instrumentation \p CodeObject
  /// \param CodeObject a Luthier instrumentation module's shared object with
  /// its instrumentation bitcode embedded inside it
  /// \returns Expects a new instance of \c InstrumentationModule on success;
  /// \c llvm::Error if \p CodeObject is not a valid instrumentation module
  static llvm::Expected<std::unique_ptr<InstrumentationModule>>
  create(std::vector<uint8_t> CodeObject);

  /// \returns the parsed object file representation of the instrumentation
  /// module
  [[nodiscard]] const llvm::object::ObjectFile &getObject() const {
    return *ObjectFile;
  }

  /// Reads the bitcode of this InstrumentationModule into a new
  /// \c llvm::Module backed by the \p Ctx
  /// \param Ctx an \c LLVMContext of the returned Module
  /// \return Expects the \c llvm::Module of the loaded instrumentation module
  /// on success
  llvm::Expected<std::unique_ptr<llvm::Module>>
  readBitcodeIntoContext(llvm::LLVMContext &Ctx) const {
    return llvm::parseBitcodeFile(*BCBuffer, Ctx);
  }

  /// \return Expects the symbol load offsets of this module
  llvm::Expected<llvm::StringMap<uint64_t>> getSymbolLoadOffsetsMap() const {
    return object::getSymbolLoadOffsetsMap(*ObjectFile);
  }
};

} // namespace luthier
#endif