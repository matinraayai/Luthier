//===-- IntrinsicProcessorRegistry.cpp ------------------------------------===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// Implements Luthier's Intrinsic Processor registry singleton.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IntrinsicProcessorRegistry.h"
#include "luthier/Intrinsic/ImplicitArgPtr.h"
#include "luthier/Intrinsic/ReadReg.h"
#include "luthier/Intrinsic/SAtomicAdd.h"
#include "luthier/Intrinsic/WriteExec.h"
#include "luthier/Intrinsic/WriteReg.h"

namespace luthier {

template <>
IntrinsicProcessorRegistry *Singleton<IntrinsicProcessorRegistry>::Instance{
    nullptr};

IntrinsicProcessorRegistry::IntrinsicProcessorRegistry()
    : Singleton<luthier::IntrinsicProcessorRegistry>() {
  /// Register built-in Luthier intrinsics
#define REGISTER_INTRINSIC(NAME, IR_PROCESSOR, MIR_PROCESSOR)                  \
  Processors.insert({NAME, {IR_PROCESSOR, MIR_PROCESSOR}});
#include "luthier/Intrinsic/IntrinsicRegistry.def"
}

} // namespace luthier