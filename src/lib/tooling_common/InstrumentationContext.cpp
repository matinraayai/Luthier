//===-- InstrumentationContext.cpp ----------------------------------------===//
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
/// This file implements the <tt>InstrumentationContext</tt> class.
//===----------------------------------------------------------------------===//
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/LuthierError.h>
#include <luthier/llvm/EagerManagedStatic.h>
#include <luthier/tooling/InstrumentationContext.h>

namespace luthier {

EagerManagedStatic<llvm::cl::OptionCategory>
    InstContextOptionsCat("Luthier Instrumentation Context Options");

EagerManagedStatic<llvm::cl::opt<llvm::CodeGenOptLevel>> DefaultCodeGenOptLevel(
    "luthier-codegen-opt-level",
    llvm::cl::desc(
        "Set the default optimization level for code generation in Luthier:"),
    llvm::cl::values(
        llvm::cl::OptionEnumValue{
            "O0", static_cast<int>(llvm::CodeGenOptLevel::None), "None"},
        llvm::cl::OptionEnumValue{
            "O1", static_cast<int>(llvm::CodeGenOptLevel::Less), "Less"},
        llvm::cl::OptionEnumValue{
            "O2", static_cast<int>(llvm::CodeGenOptLevel::Default), "Default"},
        llvm::cl::OptionEnumValue{
            "O3", static_cast<int>(llvm::CodeGenOptLevel::Aggressive),
            "Aggressive"}),
    llvm::cl::cat(*InstContextOptionsCat));

static llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(const llvm::Triple &TT,
                    const std::optional<llvm::StringRef> &CPUName,
                    const llvm::SubtargetFeatures &SF) {
  std::string Error;
  const llvm::Target *Target =
      llvm::TargetRegistry::lookupTarget(TT.normalize(), Error);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      Target,
      "Failed to get target {0} from LLVM. Error according to LLVM: {1}.",
      TT.normalize(), Error));

  /// TODO: If other targets end up being supported by Luthier, we must make
  /// sure reloc and code models are set correctly for the target being
  /// instrumented
  return std::unique_ptr<llvm::TargetMachine>(Target->createTargetMachine(
      llvm::Triple(TT).normalize(), CPUName.has_value() ? *CPUName : "unknown",
      SF.getString(), {}, llvm::Reloc::PIC_, std::nullopt,
      *DefaultCodeGenOptLevel));
}

llvm::Expected<llvm::TargetMachine &>
InstrumentationContext::getOrCreateTargetMachine(
    const llvm::Triple &TT, llvm::StringRef CPU,
    const llvm::SubtargetFeatures &STF) {
  auto Lock = getLock();

  std::string ISA = (TT.str() + CPU + STF.getString()).str();
  if (!TMs.contains(ISA)) {
    llvm::Expected<std::unique_ptr<llvm::TargetMachine>> NewTM =
        createTargetMachine(TT, CPU, STF);
    LUTHIER_RETURN_ON_ERROR(NewTM.takeError());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        *NewTM != nullptr, "Failed to create the target machine"));
    TMs.insert({ISA, std::move(*NewTM)});
  }
  return *TMs[ISA];
}

llvm::TargetMachine *
InstrumentationContext::getTargetMachine(const llvm::Triple &TT,
                                         llvm::StringRef CPU,
                                         const llvm::SubtargetFeatures &STF) {
  auto Lock = getLock();
  std::string ISA = (TT.str() + CPU + STF.getString()).str();
  if (TMs.contains(ISA))
    return TMs[ISA].get();
  return nullptr;
}

} // namespace luthier