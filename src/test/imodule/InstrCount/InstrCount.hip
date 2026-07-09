//===-- InstrCount.hip - Instruction Count IModule ----------------------===//
//
// Device-only instrumentation module for the instruction counter tool.
// Stripped from examples/InstrCount/InstrCount.hip.
//
//===----------------------------------------------------------------------===//
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

__attribute__((device)) uint64_t Counter;

__attribute__((device, used)) void
countInstructionsVector(bool CountWaveFrontLevel) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  uint32_t NumActiveThreads = __popcll(ExecMask);

  if (FirstActiveThreadId == LaneId) {
    if (CountWaveFrontLevel) {
      if (NumActiveThreads > 0) {
        atomicAdd(&Counter, 1);
      }
    } else {
      atomicAdd(&Counter, NumActiveThreads);
    }
  }
}

__attribute__((device, used)) void countInstructionsScalar() {
  (void)luthier::sAtomicAdd(&Counter, 1UL);
}
