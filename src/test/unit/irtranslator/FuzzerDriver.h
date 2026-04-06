#ifndef LUTHIER_TEST_FUZZER_DRIVER_H
#define LUTHIER_TEST_FUZZER_DRIVER_H

#include "HSADispatcher.h"
#include "IRKernelGen.h"
#include "InstrDescriptor.h"
#include "MachineKernelBuilder.h"
#include "StandaloneMIBuilder.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Target/TargetMachine.h>

#include <cstdint>
#include <string>
#include <vector>

namespace luthier::test {

/// Result of testing a single instruction with one set of inputs.
struct TestResult {
  unsigned Opcode;
  std::string InstrName;
  uint64_t Seed;
  bool Passed = false;
  bool Skipped = false;
  std::string SkipReason;
  std::string ErrorMsg;

  /// Per-output comparison detail (on mismatch).
  struct OutputComparison {
    std::string Name;
    uint64_t Reference;
    uint64_t Translated;
    unsigned SizeBytes;
    bool Matches;
  };
  std::vector<OutputComparison> Outputs;
};

/// Orchestrates the end-to-end fuzzer loop for a single instruction:
///   analyze → build MachineFunction → emit ELF (reference) →
///   generate IR (translated) → compile → allocate memory →
///   fill random inputs → dispatch both → compare.
class FuzzerDriver {
public:
  /// \param Dispatcher  An initialized HSADispatcher.
  /// \param TM          An AMDGPU TargetMachine for the detected GPU.
  ///                    Must be non-const for MachineKernelBuilder.
  /// \param Desc        An InstrDescriptor initialized from \p TM.
  FuzzerDriver(HSADispatcher &Dispatcher, llvm::TargetMachine &TM,
               const InstrDescriptor &Desc)
      : Dispatcher(Dispatcher), TM(TM), Desc(Desc),
        SAMIBuilder(TM), MKBuilder(TM), IRGen(TM) {}

  /// Test a single instruction on a specific GPU agent with the given seed.
  TestResult testInstruction(const GpuAgentInfo &Agent, unsigned Opcode,
                             uint64_t Seed);

  /// Run \p NumIterations tests for the given opcode on the given agent.
  std::vector<TestResult> testInstructionMulti(const GpuAgentInfo &Agent,
                                               unsigned Opcode,
                                               unsigned NumIterations = 10,
                                               uint64_t BaseSeed = 42);

  /// Run tests on ALL discovered GPU agents.
  std::vector<TestResult> testInstructionAllAgents(unsigned Opcode,
                                                   unsigned NumIterations = 10,
                                                   uint64_t BaseSeed = 42);

private:
  HSADispatcher &Dispatcher;
  llvm::TargetMachine &TM;
  const InstrDescriptor &Desc;
  StandaloneMIBuilder SAMIBuilder;
  MachineKernelBuilder MKBuilder;
  IRKernelGen IRGen;

  /// Fill the input fields of a kernarg buffer with random data.
  void fillRandomInputs(void *KernargPtr, const KernargLayout &Layout,
                        uint64_t Seed);

  /// Compare two output buffers (register outputs) and optionally two data
  /// buffers (memory contents for store/atomic instructions).
  bool compareOutputs(const void *RefBuf, const void *TransBuf,
                      const void *RefDataBuf, const void *TransDataBuf,
                      const KernargLayout &Layout,
                      std::vector<TestResult::OutputComparison> &Out);

  /// Format a mismatch report string.
  static std::string formatMismatch(const TestResult &R);
};

} // namespace luthier::test

#endif // LUTHIER_TEST_FUZZER_DRIVER_H
