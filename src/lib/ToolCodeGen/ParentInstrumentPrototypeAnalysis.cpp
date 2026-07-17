//===-- ParentInstrumentPrototypeAnalysis.cpp -----------------------------===//
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
/// Implements \c ParentInstrumentPrototypeAnalysis and its backing
/// \c ModuleToInstrumentPrototypeMap.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/ParentInstrumentPrototypeAnalysis.h"

namespace luthier {

llvm::AnalysisKey ParentInstrumentPrototypeAnalysis::Key;

void ModuleToInstrumentPrototypeMap::registerInstrumentPrototype(
    InstrumentPrototype &IP) {
  ModuleToIP[&IP.getTargetModule()] = &IP;
  ModuleToIP[&IP.getInstrumentationModule()] = &IP;
}

void ModuleToInstrumentPrototypeMap::unregisterInstrumentPrototype(
    InstrumentPrototype &IP) {
  ModuleToIP.erase(&IP.getTargetModule());
  ModuleToIP.erase(&IP.getInstrumentationModule());
}

InstrumentPrototype *
ModuleToInstrumentPrototypeMap::lookup(const llvm::Module &M) const {
  auto It = ModuleToIP.find(&M);
  return It == ModuleToIP.end() ? nullptr : It->second;
}

} // namespace luthier
