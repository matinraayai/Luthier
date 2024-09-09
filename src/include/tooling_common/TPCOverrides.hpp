//===-- TPCOverrides.hpp --------------------------------------------------===//
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
///
/// \file
/// This file describes member functions in the target pass config that
/// need to be overridden for Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_TPC_OVERRIDES_HPP
#define LUTHIER_TOOLING_COMMON_TPC_OVERRIDES_HPP
#include "tooling_common/HookPEIPass.hpp"


namespace llvm {

class TargetPassConfig;

}


namespace luthier {

void addMachinePassesToTPC(llvm::TargetPassConfig &TPC, HookPEIPass &PEIPass);


}

#endif