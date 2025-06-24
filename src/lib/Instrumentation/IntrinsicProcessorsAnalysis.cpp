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
/// This file implements a set of analysis passes that wrap around data
/// structures commonly used by the instrumentation passes in Luthier.
//===----------------------------------------------------------------------===//
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/Instrumentation/IntrinsicProcessorsAnalysis.h>
#include <luthier/Instrumentation/Intrinsics/ImplicitArgPtr.h>
#include <luthier/Instrumentation/Intrinsics/ReadReg.h>
#include <luthier/Instrumentation/Intrinsics/SAtomicAdd.h>
#include <luthier/Instrumentation/Intrinsics/WriteExec.h>
#include <luthier/Instrumentation/Intrinsics/WriteReg.h>

namespace luthier {

llvm::AnalysisKey IntrinsicsProcessorsAnalysis::Key;

std::optional<IntrinsicProcessor>
IntrinsicsProcessorsAnalysis::Result::getProcessor(
    llvm::StringRef IntrinsicName) const {
  auto It = IntrinsicProcessors.find(IntrinsicName);
  return It != IntrinsicProcessors.end() ? It->second : std::nullopt;
}

llvm::Error IntrinsicsProcessorsAnalysis::Result::defineIntrinsic(
    llvm::StringRef IntrinsicName, IntrinsicIRProcessorFunc IRProcessor,
    IntrinsicMIRProcessorFunc MIRProcessor) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !IntrinsicProcessors.contains(IntrinsicName),
      llvm::formatv("Intrinsic {0} is already defined", IntrinsicName)));
  IntrinsicProcessors.insert({IntrinsicName, {IRProcessor, MIRProcessor}});
  return llvm::Error::success();
}

IntrinsicsProcessorsAnalysis::IntrinsicsProcessorsAnalysis()
    : IntrinsicProcessors(
          {{"luthier::readReg", {readRegIRProcessor, readRegMIRProcessor}},
           {"luthier::writeReg", {writeRegIRProcessor, writeRegMIRProcessor}},
           {"luthier::writeExec",
            {writeExecIRProcessor, writeExecMIRProcessor}},
           {"luthier::implicitArgPtr",
            {implicitArgPtrIRProcessor, implicitArgPtrMIRProcessor}},
           {"luthier::sAtomicAdd",
            {sAtomicAddIRProcessor, sAtomicAddMIRProcessor}}}) {}

} // namespace luthier