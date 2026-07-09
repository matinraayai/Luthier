//===-- OpcodeHistogram.hip - Opcode Histogram IModule -----------------===//
//
// Device-only instrumentation module for the opcode histogram tool.
// Stripped from examples/OpcodeHistogram/OpcodeHistogram.hip.
//
// NOTE: The host side reads llvm::AMDGPU::INSTRUCTION_LIST_END from LLVM
// headers; the device array sized identically must agree at runtime. We
// hard-code an upper bound here (any value safely above current
// INSTRUCTION_LIST_END for AMDGPU). Adjust if backend grows.
//
//===----------------------------------------------------------------------===//
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

constexpr unsigned LUTHIER_TEST_OPCODE_LIST_END = 32768;

__attribute__((device)) uint64_t KernelHistogram[LUTHIER_TEST_OPCODE_LIST_END];

__attribute__((device, used)) void
countOpcodeHistogramVector(bool CountWaveFrontLevel, int OpcodeNumber) {
  unsigned long long int ExecMask = __builtin_amdgcn_read_exec();
  const uint32_t LaneId = __lane_id() + 1;
  uint32_t FirstActiveThreadId = __ffsll(ExecMask);
  uint32_t NumActiveThreads = __popcll(ExecMask);

  if (FirstActiveThreadId == LaneId) {
    if (CountWaveFrontLevel) {
      if (NumActiveThreads > 0) {
        atomicAdd(&KernelHistogram[OpcodeNumber], 1);
      }
    } else {
      atomicAdd(&KernelHistogram[OpcodeNumber], NumActiveThreads);
    }
  }
}

__attribute__((device, used)) void
countOpcodeHistogramScalar(int OpcodeNumber) {
  luthier::sAtomicAdd(&KernelHistogram[OpcodeNumber], 1UL);
}
