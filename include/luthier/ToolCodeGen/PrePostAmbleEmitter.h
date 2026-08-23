//===-- PrePostAmbleEmitter.h -----------------------------------*- C++ -*-===//
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
///
/// \file
/// This file describes the Pre and post amble emitter,
/// which will emits code before and after
/// using the information gathered from code gen passes when generating
/// the hooks.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_PRE_POST_AMBLE_EMITTER_H
#define LUTHIER_TOOL_CODE_GEN_PRE_POST_AMBLE_EMITTER_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/Support/Error.h>

namespace luthier {

class SVStorageAndLoadLocations;

} // namespace luthier

#endif