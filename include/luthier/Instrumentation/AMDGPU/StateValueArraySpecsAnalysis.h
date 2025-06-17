//===-- StateValueArraySpecsAnalysis.h --------------------------*- C++ -*-===//
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
/// Provides an analysis that can be used by other passes in the AMDGPU
/// instrumentation pipline to query specifications of the state
/// value array (e.g. frame spill slots, where the kernel arguments are
/// stored, etc).
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_AMDGPU_STATE_VALUE_ARRAY_SPECS_H
#define LUTHIER_INSTRUMENTATION_AMDGPU_STATE_VALUE_ARRAY_SPECS_H
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCRegister.h>
#include <llvm/Support/Error.h>
#include <luthier/Instrumentation/IntrinsicProcessor.h>

namespace luthier {

class StateValueArraySpecsAnalysis
    : public llvm::AnalysisInfoMixin<StateValueArraySpecsAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {

    /// A mapping between the application registers that need to be spilled
    /// before
    /// the instrumentation frame is loaded, and their spill lane IDs in the
    /// state value array
    const llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>
        FrameSpillSlots;

    /// A mapping between stack frame registers of the instrumentation function
    /// and the lane IDs of where they will be stored in the state value array
    const llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>
        InstrumentationStackFrameStoreSlots;

    /// A mapping between the kernel arguments and the lane ID of where they
    /// will be stored in the state value array, as well as
    /// Intended for use for when the kernel's wavefront size is 64
    const llvm::SmallDenseMap<KernelArgumentType, std::pair<short, short>, 32>
        WaveFront64KernelArgumentStoreSlots;

    Result();

  public:
    /// \return \c true if \p Reg belongs to a spill slot on the state value
    /// array,
    /// \c false otherwise
    bool isFrameSpillSlot(llvm::MCRegister Reg) const;

    llvm::iterator_range<decltype(FrameSpillSlots)::const_iterator>
    getFrameSpillSlots() const;

    /// \param Reg SGPRs that clobber the frame of an AMD GPU device function
    /// with the C-calling convention, i.e. s0, s1, s2, s3, s32, s33, FS_LO, and
    /// FS_HI
    /// \return the lane ID in the state value array where the SGPR is spilled,
    /// or 255 if the register doesn't get clobbered by a device function's
    /// stack frame
    unsigned short getFrameSpillSlotLaneId(llvm::MCRegister Reg) const;

    unsigned short
    getInstrumentationStackFrameLaneIdStoreSlot(llvm::MCRegister Reg) const;

    llvm::iterator_range<
        decltype(InstrumentationStackFrameStoreSlots)::const_iterator>
    getFrameStoreSlots() const;

    llvm::Expected<unsigned short>
    getKernelArgumentLaneIdStoreSlotBeginForWave64(
        KernelArgumentType Arg) const;

    llvm::Expected<unsigned short>
    getKernelArgumentStoreSlotSizeForWave64(KernelArgumentType Arg) const;

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  Result SVASpecs;

  StateValueArraySpecsAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) { return SVASpecs; }
};

} // namespace luthier

#endif