//===-- PreKernelEmitter.hpp ----------------------------------------------===//
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
/// This file describes the Pre-kernel emitter, which will emit a pre-kernel
/// using the information gathered from code gen passes when generating
/// the hooks.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_PRE_KERNEL_DESCRIPTOR_HPP
#define LUTHIER_TOOLING_COMMON_PRE_KERNEL_DESCRIPTOR_HPP
#include <luthier/LiftedRepresentation.h>
#include "tooling_common/LRStateValueLocations.hpp"

namespace luthier {

/// \brief a struct which aggregates information about the pre-kernel's
/// specifications
struct PreKernelEmissionDescriptor {
  /// Whether or not the kernels requires a pre-kernel to be emitted
  /// This is set to true if any of the hooks make use of the state value
  /// VGPR, or use the stack in any shape or form
  bool DoesNeedPreKernel{false};

  /// Whether or not the kernels require to enable scratch (if not already
  /// enabled) and the pre-kernel requires the stack information to be stored
  /// inside the state value register
  bool EnableScratchAndStoreStackInfo{false};

  /// The maximum amount of scratch memory per work-item requested by the
  /// hooks
  uint32_t AmountOfScratchRequested{0};
};

class PreKernelEmitter {
private:
  LiftedRepresentation &LR;

  LRStateValueLocations & SVLocations;

  PreKernelEmissionDescriptor PKInfo;

public:
  explicit PreKernelEmitter(PreKernelEmissionDescriptor Info,
                            LiftedRepresentation &LR, LRStateValueLocations &SVLocs)
      : PKInfo(Info), LR(LR), SVLocations(SVLocs) {};

  llvm::Error emitPreKernel();
};

} // namespace luthier

#endif