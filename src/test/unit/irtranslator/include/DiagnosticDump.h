//===-- DiagnosticDump.h ----------------------------------------*- C++ -*-===//
//
// Writes a per-opcode diagnostic report when the translation path disagrees
// with the reference path, or when it takes the process down with it.
//
// A report holds the two sides of the comparison in the form needed to see
// where a semantic went wrong: the reference kernel's MachineInstrs (what the
// translator was asked to model) and the lifted IR function (what it produced),
// plus the inputs, the per-field output diff, and the frame/ABI facts that do
// not survive the round trip through IR.
//
// Crash safety is the reason this is not simply a failure handler. A translated
// kernel that faults on the device aborts the process from inside the HSA
// runtime — no error is returned and no later code runs — so the report is
// written *before* each dispatch and deleted afterwards if the test passed. A
// report left on disk therefore means one of two things:
//
//   * STATUS: FAILED  — the comparison ran and the outputs differed.
//   * STATUS: DISPATCHING — the process died mid-dispatch; the kernel named in
//     the report is the one that killed it.
//
// Controlled by the environment:
//   LUTHIER_FUZZER_DUMP=0     disable reports entirely
//   LUTHIER_FUZZER_DUMP_ALL=1 keep reports for passing opcodes too
//   LUTHIER_FUZZER_DUMP_DIR   output directory (default ./luthier-fuzzer-dumps)
//
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_DIAGNOSTIC_DUMP_H
#define LUTHIER_TEST_DIAGNOSTIC_DUMP_H

#include "MachineKernelBuilder.h" // KernargLayout, TranslationArtifacts

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

#include <cstdint>
#include <string>

namespace luthier::test {

/// Whether diagnostic reports are produced at all (LUTHIER_FUZZER_DUMP).
bool isDumpingEnabled();

/// Whether reports are kept for passing opcodes (LUTHIER_FUZZER_DUMP_ALL).
bool isDumpAllEnabled();

/// Directory reports are written to (LUTHIER_FUZZER_DUMP_DIR).
llvm::StringRef getDumpDir();

/// A pending diagnostic report for one instruction under test.
///
/// One instance corresponds to one FuzzerDriver::testInstruction call and owns
/// a single file, rewritten in place as the test progresses. Nothing here can
/// fail a test: when reporting is disabled every method is a no-op, and I/O
/// errors are reported to stderr and otherwise ignored.
class DiagnosticDump {
public:
  DiagnosticDump(llvm::StringRef InstrName, uint64_t Seed);

  /// False when reporting is disabled; callers can skip capturing artifacts.
  bool enabled() const { return Enabled; }

  /// The artifact buffer to hand to MachineKernelBuilder::emitTranslatedToELF.
  TranslationArtifacts &artifacts() { return Artifacts; }

  /// Record the shared kernarg/output layout and the values fed to both paths.
  void setLayout(const KernargLayout &Layout);
  void setInputs(llvm::ArrayRef<uint32_t> Values);

  /// Write the report, replacing any previous content, with \p Status as its
  /// verdict and \p Detail as the human-readable explanation. Safe to call
  /// repeatedly.
  void write(llvm::StringRef Status, llvm::StringRef Detail = "");

  /// Delete a previously written report because the test passed. Honours
  /// LUTHIER_FUZZER_DUMP_ALL, which keeps it instead.
  void discard();

  /// Absolute path of the report file.
  llvm::StringRef path() const { return Path; }

private:
  bool Enabled;
  std::string InstrName;
  uint64_t Seed;
  std::string Path;
  TranslationArtifacts Artifacts;
  std::string LayoutText;
  std::string InputText;
  /// Set once a file exists on disk, so discard() does not stat a missing path.
  bool Written = false;
};

} // namespace luthier::test

#endif // LUTHIER_TEST_DIAGNOSTIC_DUMP_H
