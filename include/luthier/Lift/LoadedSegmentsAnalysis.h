//===-- LoadedSegmentsAnalysis.h --------------------------------*- C++ -*-===//
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
/// \file
/// Defines the \c LoadedSegmentsAnalysis class, which provides a
/// host-accessible copy of the currently loaded memory segments on a
/// target GPU agent for inspection by other lifting passees.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LIFT_LOADED_CONTENTS_ANALYSIS_H
#define LUTHIER_LIFT_LOADED_CONTENTS_ANALYSIS_H
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Error.h>

namespace luthier {

/// \brief Provides host-accessible views of the loaded memory
/// segments (allocations) on a target device for inspection by other
/// lifting passes. The underlying buffers of the host-accessible segment views
/// can be either managed by the target GPU runtime or be directly managed by
/// the \c LoadedSegmentsAnalysis::Result itself in case the runtime
/// doesn't have a host-accessible address or copy of the requested allocation.
/// This analysis is runtime agnostic. Any target runtime has to extend this
/// class to provide the concrete way to obtain and manage the host-accessible
/// views.
/// If this analysis is absent from the pass pipeline, it indicates that
/// the currently inspected binary has no loaded contents; In which case,
/// the \c AMDGCNObjectFileAnalysis will be required, and the contents of the
/// object file will be inspected instead.
class LoadedSegmentsAnalysis
    : public llvm::AnalysisInfoMixin<LoadedSegmentsAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  struct SegmentDescriptor {
    /// Size of the segment in bytes
    size_t Size;
    /// Start address of the segment on the device
    void *DeviceAddress;
    /// Start address of the segment on the host
    void *HostAddress;
  };

  class Result {
    friend LoadedSegmentsAnalysis;

    const LoadedSegmentsAnalysis &Parent;

    explicit Result(const LoadedSegmentsAnalysis &Parent) : Parent(Parent) {};

  public:
    llvm::Expected<SegmentDescriptor>
    getSegmentDescriptor(uint64_t Addr) const {
      return getSegmentDescriptor(reinterpret_cast<const void *>(Addr));
    }

    llvm::Expected<SegmentDescriptor>
    getSegmentDescriptor(const void *Addr) const {
      return Parent.getSegmentDescriptor(Addr);
    }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  virtual llvm::Expected<SegmentDescriptor>
  getSegmentDescriptor(const void *Addr) const = 0;

  explicit LoadedSegmentsAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result{*this};
  }

  virtual ~LoadedSegmentsAnalysis() = default;
};

} // namespace luthier

#endif
