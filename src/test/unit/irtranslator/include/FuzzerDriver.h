//===-- FuzzerDriver.h ------------------------------------------*- C++ -*-===//
//
// Orchestrates one instruction test:
//   analyze -> build the MI + kernel MachineFunction -> emit reference ELF ->
//   fill random inputs -> dispatch on the GPU -> compare outputs.
//
// The translation (LLVM IR) path is not wired up yet; see testInstruction().
//
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_FUZZER_DRIVER_H
#define LUTHIER_TEST_FUZZER_DRIVER_H

#include "HSADispatcher.h"
#include "InstrDescriptor.h"
#include "MachineKernelBuilder.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Target/TargetMachine.h>

#include <cstdint>
#include <string>
#include <vector>

namespace luthier::test {

/// Result of testing a single instruction with one set of inputs.
struct TestResult {
  unsigned Opcode = 0;
  std::string InstrName;
  uint64_t Seed = 0;

  bool Passed = false;
  bool Skipped = false;
  std::string SkipReason;
  std::string ErrorMsg;

  /// The random input values written into the kernarg buffer, in operand order.
  std::vector<uint32_t> InputValues;

  /// Per-output comparison detail.
  struct OutputComparison {
    std::string Name;
    uint32_t Reference = 0;
    uint32_t Translated = 0;
    unsigned SizeBytes = 0;
    bool Matches = false;
  };
  std::vector<OutputComparison> Outputs;
};

class FuzzerDriver {
public:
  FuzzerDriver(HSADispatcher &Dispatcher, llvm::TargetMachine &TM,
               const InstrDescriptor &Desc)
      : Dispatcher(Dispatcher), TM(TM), Desc(Desc), MKBuilder(TM) {}

  /// Test a single instruction on \p Agent.
  ///
  /// Inputs are derived from \p Seed, unless \p FixedInputs is non-empty, in
  /// which case those raw 32-bit values are used verbatim (one per explicit
  /// input operand).
  ///
  /// When \p CompareTranslation is true, the second dispatch runs the
  /// translation path (the reference kernel raised back to LLVM IR by
  /// TraceFunctionTranslator and recompiled) instead of a second reference dispatch,
  /// so a mismatch flags a wrong/missing instruction semantic. Otherwise both
  /// dispatches run the reference kernel (a determinism / stability check).
  TestResult testInstruction(const GpuAgentInfo &Agent, unsigned Opcode,
                             uint64_t Seed,
                             llvm::ArrayRef<uint32_t> FixedInputs = {},
                             bool CompareTranslation = true);

private:
  HSADispatcher &Dispatcher;
  llvm::TargetMachine &TM;
  const InstrDescriptor &Desc;
  MachineKernelBuilder MKBuilder;
};

/// Format a human-readable report of \p R.
std::string formatResult(const TestResult &R);

} // namespace luthier::test

#endif // LUTHIER_TEST_FUZZER_DRIVER_H
