//===-- WriteExecGuarded.hip - writeExec test --------------------------===//
//
// Exercises luthier::writeExec by serializing the hook body to a single lane
// (lane 0) and restoring EXEC on the way out. Useful for validating that
// writeExec is correctly lowered and that the surrounding code restores
// EXEC properly across the hook boundary.
//
//===----------------------------------------------------------------------===//
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

__attribute__((device)) uint64_t SerializedBumps;

__attribute__((device, used)) void serializedBump() {
  unsigned long long int Saved = __builtin_amdgcn_read_exec();
  luthier::writeExec(1ULL);
  atomicAdd(&SerializedBumps, 1UL);
  luthier::writeExec(Saved);
}
