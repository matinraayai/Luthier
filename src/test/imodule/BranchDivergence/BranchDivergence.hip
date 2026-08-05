//===-- BranchDivergence.hip - Divergence counter ---------------------===//
//
// Hook intended to be inserted before branch instructions. Computes whether
// the wave is diverged at this point (some lanes inactive) and increments
// matching counters. Tests payload injection at branch boundaries.
//
//===----------------------------------------------------------------------===//
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

__attribute__((device)) uint64_t ConvergedBranches;

__attribute__((device)) uint64_t DivergedBranches;

__attribute__((device, used)) void recordBranchDivergence() {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  uint32_t NumActive = __popcll(ExecMask);
  if (FirstActiveThreadId == LaneId) {
    if (NumActive == 64)
      atomicAdd(&ConvergedBranches, 1UL);
    else
      atomicAdd(&DivergedBranches, 1UL);
  }
}
