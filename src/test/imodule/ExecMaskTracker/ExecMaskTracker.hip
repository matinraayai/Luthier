//===-- ExecMaskTracker.hip - EXEC + lane built-ins --------------------===//
//
// Tests instrumentation that introspects the EXEC mask via __builtin_amdgcn_*
// and computes lane-relative quantities. Records the number of fully-active
// wavefronts vs. partially-active wavefronts encountered at the
// instrumentation point.
//
//===----------------------------------------------------------------------===//
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

__attribute__((device)) uint64_t FullWaves;

__attribute__((device)) uint64_t PartialWaves;

__attribute__((device, used)) void classifyWave() {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  if (FirstActiveThreadId == LaneId) {
    if (ExecMask == 0xFFFFFFFFFFFFFFFFULL)
      atomicAdd(&FullWaves, 1UL);
    else
      atomicAdd(&PartialWaves, 1UL);
  }
}
