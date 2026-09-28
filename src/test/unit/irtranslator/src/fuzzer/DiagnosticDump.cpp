//===-- DiagnosticDump.cpp --------------------------------------*- C++ -*-===//
#include "DiagnosticDump.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>

namespace luthier::test {

namespace {

/// Read \p Name as a boolean, treating "0", "false", and "off" as false.
bool envFlag(const char *Name, bool Default) {
  const char *V = std::getenv(Name);
  if (!V || !*V)
    return Default;
  llvm::StringRef S(V);
  return !(S == "0" || S.equals_insensitive("false") ||
           S.equals_insensitive("off") || S.equals_insensitive("no"));
}

/// Number of non-overlapping occurrences of \p Needle in \p Haystack.
unsigned countOccurrences(llvm::StringRef Haystack, llvm::StringRef Needle) {
  unsigned N = 0;
  for (size_t P = Haystack.find(Needle); P != llvm::StringRef::npos;
       P = Haystack.find(Needle, P + Needle.size()))
    ++N;
  return N;
}

/// Derive what is provably wrong from the captured text alone.
///
/// These checks exist because the two most common translation failures are
/// invisible in the pass/fail output but obvious in the raised IR:
///
///   * An opcode the translator has no semantic for does not fail loudly — it
///     falls through to TraceFunctionTranslator's default case, which re-emits the
///     instruction as inline asm. The recompiled kernel then runs *something*,
///     so the only trace is the `asm` in the IR.
///   * A kernel whose scratch backing was established directly on the reference
///     MachineFunction (CreateStackObject) loses it here, because the
///     translated kernel is rebuilt from IR and a stack object is not IR. The
///     descriptor then reports private_segment_fixed_size = 0, HSA allocates no
///     scratch, and the flat-scratch base is null — a page fault at address
///     (nil) on the first private access.
void appendAnalysis(llvm::raw_ostream &OS, const TranslationArtifacts &A) {
  llvm::StringRef IR = A.RaisedIR;
  if (IR.empty())
    return;

  llvm::SmallVector<std::string, 4> Findings;

  unsigned InlineAsm = countOccurrences(IR, " asm ");
  if (InlineAsm)
    Findings.push_back(llvm::formatv(
        "The raised IR contains {0} inline-asm call(s). TraceFunctionTranslator "
        "re-emits an instruction as inline asm when it has NO semantic for "
        "that opcode, so at least one instruction here is unmodelled.",
        InlineAsm));

  unsigned Private = countOccurrences(IR, "addrspace(5)");
  unsigned Allocas = countOccurrences(IR, " alloca ");
  if (Private && !Allocas)
    Findings.push_back(llvm::formatv(
        "The raised IR performs {0} private/scratch (addrspace(5)) access(es) "
        "but declares no alloca. The translated kernel is rebuilt from this IR "
        "alone, so its descriptor will report private_segment_fixed_size = 0 "
        "and HSA will back it with no scratch: the flat-scratch base is null "
        "and the first private access faults at address (nil). Compare the "
        "'reference frame facts' section below — if the reference kernel has "
        "stack objects and this IR has no alloca, the scratch backing was lost "
        "in the round trip.",
        Private));

  bool HasMemOp = IR.contains(" load ") || IR.contains(" store ") ||
                  IR.contains("atomicrmw") || IR.contains("cmpxchg");
  if (!HasMemOp && A.ReferenceMIR.contains("mem:"))
    Findings.push_back(
        "The reference MachineFunction has memory operands but the raised IR "
        "has no load/store/atomic. The instruction's memory effect was dropped "
        "entirely.");

  if (Findings.empty())
    return;

  OS << "\n================================================================\n"
     << " AUTOMATED ANALYSIS\n"
     << "================================================================\n";
  for (unsigned I = 0; I < Findings.size(); ++I)
    OS << "\n[" << (I + 1) << "] " << Findings[I] << "\n";
}

} // namespace

bool isDumpingEnabled() {
  static const bool Enabled = envFlag("LUTHIER_FUZZER_DUMP", true);
  return Enabled;
}

bool isDumpAllEnabled() {
  static const bool All = envFlag("LUTHIER_FUZZER_DUMP_ALL", false);
  return All;
}

llvm::StringRef getDumpDir() {
  static const std::string Dir = [] {
    if (const char *V = std::getenv("LUTHIER_FUZZER_DUMP_DIR"); V && *V)
      return std::string(V);
    // Default to the working directory, which is the build directory when the
    // suite is run under ctest.
    llvm::SmallString<128> P;
    if (llvm::sys::fs::current_path(P))
      return std::string("luthier-fuzzer-dumps");
    llvm::sys::path::append(P, "luthier-fuzzer-dumps");
    return std::string(P);
  }();
  return Dir;
}

DiagnosticDump::DiagnosticDump(llvm::StringRef InstrName, uint64_t Seed)
    : Enabled(isDumpingEnabled()), InstrName(InstrName.str()), Seed(Seed) {
  if (!Enabled)
    return;
  std::string Safe = this->InstrName;
  for (char &C : Safe)
    if (!llvm::isAlnum(C) && C != '_' && C != '-' && C != '.')
      C = '_';
  llvm::SmallString<192> P(getDumpDir());
  llvm::sys::path::append(P, Safe + ".txt");
  Path = std::string(P);
}

void DiagnosticDump::setLayout(const KernargLayout &Layout) {
  if (!Enabled)
    return;
  LayoutText.clear();
  llvm::raw_string_ostream OS(LayoutText);
  OS << "kernarg total size   : " << Layout.TotalSize << "\n"
     << "output buffer size   : " << Layout.OutputBufSize << "\n"
     << "grid / workgroup     : " << Layout.GridSizeX << " / "
     << Layout.WorkgroupSizeX << "\n"
     << "group segment (LDS)  : " << Layout.GroupSegmentSize << "\n"
     << "private segment      : " << Layout.PrivateSegmentSize << "\n";
  if (Layout.DataBufPtrOffset != UINT32_MAX)
    OS << "data buffer          : " << Layout.DataBufSizeBytes
       << " bytes, pointer at kernarg+" << Layout.DataBufPtrOffset << "\n";
  OS << "\ninput fields:\n";
  for (const auto &F : Layout.Inputs)
    OS << llvm::formatv("  +{0,-5} {1,-4}B  {2}\n", F.Offset, F.SizeBytes,
                        F.Name);
  OS << "\noutput fields:\n";
  for (const auto &F : Layout.Outputs)
    OS << llvm::formatv("  +{0,-5} {1,-4}B  {2}\n", F.Offset, F.SizeBytes,
                        F.Name);
}

void DiagnosticDump::setInputs(llvm::ArrayRef<uint32_t> Values) {
  if (!Enabled)
    return;
  InputText.clear();
  llvm::raw_string_ostream OS(InputText);
  for (unsigned I = 0; I < Values.size(); ++I)
    OS << llvm::formatv("  src{0} = 0x{1:x-8}\n", I, Values[I]);
}

void DiagnosticDump::write(llvm::StringRef Status, llvm::StringRef Detail) {
  if (!Enabled)
    return;

  if (auto EC = llvm::sys::fs::create_directories(getDumpDir())) {
    llvm::errs() << "[fuzzer] cannot create dump directory " << getDumpDir()
                 << ": " << EC.message() << "\n";
    Enabled = false;
    return;
  }

  std::error_code EC;
  llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "[fuzzer] cannot write dump " << Path << ": "
                 << EC.message() << "\n";
    Enabled = false;
    return;
  }

  OS << "================================================================\n"
     << " " << InstrName << "\n"
     << "================================================================\n"
     << "STATUS : " << Status << "\n"
     << "SEED   : " << Seed << "\n";
  if (!Detail.empty())
    OS << "\n" << Detail << "\n";

  appendAnalysis(OS, Artifacts);

  if (!InputText.empty())
    OS << "\n---------------- inputs ----------------\n" << InputText;
  if (!LayoutText.empty())
    OS << "\n---------------- kernarg / output layout ----------------\n"
       << LayoutText;

  if (!Artifacts.ReferenceFrameFacts.empty())
    OS << "\n---------------- reference frame facts ----------------\n"
       << "(state held on the reference MachineFunction; anything the raised "
          "IR\n does not re-encode is lost when the translated kernel is "
          "rebuilt)\n\n"
       << Artifacts.ReferenceFrameFacts;

  OS << "\n================================================================\n"
     << " REFERENCE MACHINE FUNCTION (the MI the translator was given)\n"
     << "================================================================\n"
     << Artifacts.ReferenceMIR;

  OS << "\n================================================================\n"
     << " LIFTED IR FUNCTION (what TraceFunctionTranslator produced)\n"
     << "================================================================\n"
     << Artifacts.RaisedIR;

  OS << "\n================================================================\n"
     << " REFERENCE IR SHELL (before the body was replaced)\n"
     << "================================================================\n"
     << Artifacts.ReferenceIR;

  OS.close();
  Written = true;

  static bool Announced = false;
  if (!Announced) {
    Announced = true;
    llvm::errs() << "[fuzzer] diagnostic dumps -> " << getDumpDir() << "\n";
  }
}

void DiagnosticDump::discard() {
  if (!Enabled || !Written || isDumpAllEnabled())
    return;
  llvm::sys::fs::remove(Path);
  Written = false;
}

} // namespace luthier::test
