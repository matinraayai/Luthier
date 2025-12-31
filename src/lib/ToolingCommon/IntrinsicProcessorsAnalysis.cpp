//===-- IntrinsicProcessorsAnalysis.cpp -----------------------------------===//
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
/// Implements the intrinsics processors analysis.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IntrinsicProcessorsAnalysis.h"
#include "luthier/Tooling/IntrinsicProcessorRegistry.h"

namespace luthier {
llvm::AnalysisKey IntrinsicsProcessorsAnalysis::Key;

std::optional<IntrinsicProcessor>
IntrinsicsProcessorsAnalysis::Result::getProcessorIfRegistered(
    llvm::StringRef Name) const {
  return IntrinsicProcessorRegistry::instance()
      .getIntrinsicProcessorIfRegistered(Name);
}

bool IntrinsicsProcessorsAnalysis::Result::isProcessorRegistered(
    llvm::StringRef Name) const {
  return IntrinsicProcessorRegistry::instance().isIntrinsicProcessorRegistered(
      Name);
}

void IntrinsicsProcessorsAnalysis::Result::registerIntrinsicProcessor(
    llvm::StringRef Name, IntrinsicProcessor Processor) {
  return IntrinsicProcessorRegistry::instance().registerIntrinsicProcessor(
      Name, std::move(Processor));
}

void IntrinsicsProcessorsAnalysis::Result::unregisterIntrinsicProcessor(
    llvm::StringRef Name) {
  return IntrinsicProcessorRegistry::instance().unregisterIntrinsicProcessor(
      Name);
}
} // namespace luthier