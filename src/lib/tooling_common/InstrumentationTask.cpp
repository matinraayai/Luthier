//===-- InstrumentationTask.cpp - Instrumentation Task --------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file implements the instrumentation task class.
//===----------------------------------------------------------------------===//
#include <luthier/InstrumentationTask.h>

#include "tooling_common/CodeGenerator.hpp"
#include "tooling_common/CodeLifter.hpp"
#include "tooling_common/ToolExecutableLoader.hpp"

namespace luthier {

llvm::Error luthier::InstrumentationTask::insertHookBefore(
    llvm::MachineInstr &MI, const void *Hook,
    llvm::ArrayRef<std::variant<llvm::Constant *, llvm::MCRegister>> Args) {
  const auto *SIM = llvm::dyn_cast<StaticInstrumentationModule>(&IM);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ARGUMENT_ERROR_CHECK(SIM != nullptr));
  auto HookName = SIM->convertHookHandleToHookName(Hook);
  LUTHIER_RETURN_ON_ERROR(HookName.takeError());
  // Check if the passed MI belongs to the LiftedRepresentation being
  // worked on
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ARGUMENT_ERROR_CHECK(LR.getHSAInstr(MI) != nullptr));
  if (!HookInsertionTasks.contains(&MI)) {
    HookInsertionTasks.insert({&MI, {}});
  }
  HookInsertionTasks[&MI].emplace_back(
      *HookName,
      llvm::SmallVector<std::variant<llvm::Constant *, llvm::MCRegister>>(
          Args));
  return llvm::Error::success();
}

InstrumentationTask::InstrumentationTask(LiftedRepresentation &LR)
    : LR(LR),
      IM(ToolExecutableLoader::instance().getStaticInstrumentationModule()){};

} // namespace luthier