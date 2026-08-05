//===-- TrivialCounter.hip - Smallest possible IModule ----------------===//
//
// Smallest possible Luthier instrumentation module: a single zero-argument
// hook that bumps a global counter via an unconditional atomicAdd. Useful as
// a smoke test for payload injection plumbing — no intrinsics, no EXEC
// inspection, no inline asm.
//
//===----------------------------------------------------------------------===//
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

__attribute__((device)) uint64_t TrivialCounter;

__attribute__((device, used)) void bumpCounter() {
  atomicAdd(&TrivialCounter, 1UL);
}
