//===-- RegLogger.hip - readReg + payload arg test ---------------------===//
//
// Tests the readReg intrinsic with a payload-injected MCRegister argument.
// Logs the value of the register passed in by the host instrumentation pass
// into a per-register slot. Intended to validate that:
//   - readReg works for both 32-bit and 64-bit reads
//   - payload args (MCRegister + a uint32_t slot index) survive lowering
//   - return values from readReg participate correctly in atomicAdd
//
//===----------------------------------------------------------------------===//
#include <luthier/Intrinsic/Intrinsics.h>
#include <luthier/ToolCodeGen/FunctionAnnotations.h>

__attribute__((device)) uint64_t RegSum32[16];

__attribute__((device)) uint64_t RegSum64[16];

__attribute__((device, used)) void logReg32(const llvm::MCRegister Reg,
                                            uint32_t Slot) {
  uint32_t Val = luthier::readReg<uint32_t>(Reg);
  atomicAdd(&RegSum32[Slot & 0xF], static_cast<uint64_t>(Val));
}

__attribute__((device, used)) void logReg64(const llvm::MCRegister Reg,
                                            uint32_t Slot) {
  uint64_t Val = luthier::readReg<uint64_t>(Reg);
  atomicAdd(&RegSum64[Slot & 0xF], Val);
}
