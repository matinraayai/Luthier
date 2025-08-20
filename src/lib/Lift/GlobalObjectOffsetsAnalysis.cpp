//===-- GlobalObjectOffsetsAnalysis.cpp -----------------------------------===//
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
/// Implements the \c GlobalObjectOffsetsAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/Lift/GlobalObjectOffsetsAnalysis.h"

namespace luthier {

llvm::AnalysisKey GlobalObjectOffsetsAnalysis::Key;

const llvm::GlobalObject *
GlobalObjectOffsetsAnalysis::Result::getGlobalObject(uint64_t Offset) const {
  auto It = OffsetToObjectMap.find(Offset);
  return It != OffsetToObjectMap.end() ? &It->second : nullptr;
}

uint64_t GlobalObjectOffsetsAnalysis::Result::getOffset(
    const llvm::GlobalObject &GO) const {
  auto It = ObjectToOffsetMap.find(const_cast<llvm::GlobalObject &>(GO));
  return It != ObjectToOffsetMap.end() ? It->second : 0;
}

} // namespace luthier