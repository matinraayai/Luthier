//===-- Executable.cpp - HSA Executable Wrapper----------------------------===//
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
/// This file implements the concrete portions of the \c hsa::Executable
/// interface.
//===----------------------------------------------------------------------===//
#include "hsa/Executable.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"

namespace luthier::hsa {

llvm::Error Executable::create() {
  return create(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);
}

llvm::Expected<hsa::LoadedCodeObject>
Executable::loadAgentCodeObject(const CodeObjectReader &Reader,
                                const GpuAgent &Agent) {
  return loadAgentCodeObject(Reader, Agent, "");
}

} // namespace luthier::hsa
