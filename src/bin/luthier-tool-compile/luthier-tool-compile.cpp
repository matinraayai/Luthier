//===-- luthier-tool-compile.cpp ----------------------------------*-C++-*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file luthier-tool-compile.cpp
/// Custom Luthier tool device HIP code compiler driver; For each GPU
/// architecture passed, the driver creates a combination of all possible
/// sub-target features supported by that target, invokes the clang driver for
/// each, and bundles the result of all clang driver invocations into a FAT
/// binary.
///
/// This driver exists because the clang driver, due to its argument
/// parsing design for the AMD GPU backend, is not capable of compiling for the
/// same target with multiple sub-target features both enabled and disabled
/// at the same time.
//===----------------------------------------------------------------------===//
#include <array>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/AMDGPUTargetParser.h>
#include <llvm/TargetParser/Host.h>
#include <optional>
#include <string>

using namespace llvm;

//===----------------------------------------------------------------------===//
// CLI
//===----------------------------------------------------------------------===//

static cl::list<std::string>
    Archs("arch", cl::desc("AMDGCN arch to compile for (e.g. gfx906, gfx1201). "
                           "Repeatable."));

static cl::list<std::string> VaryArgs(
    "vary",
    cl::desc("Opt a knob into the matrix. Knob name without sign "
             "(xnack | sramecc | wavefrontsize64 | cumode). Defaults: "
             "wavefrontsize64 and cumode are already in the matrix when "
             "the arch supports them; xnack and sramecc are not."));

static cl::list<std::string>
    PinArgs("pin",
            cl::desc("Pin a knob to one value (e.g. wavefrontsize64+, "
                     "cumode-, xnack+). Removes that knob from the matrix."));

static cl::opt<std::string> HipClang("hip-clang", cl::desc("Path to clang++."),
                                     cl::Required);

static cl::opt<std::string>
    PluginIR("plugin-ir",
             cl::desc("Path to LuthierToolIRCompilationPlugin .so for "
                      "`-fpass-plugin=`."),
             cl::Required);

static cl::opt<std::string>
    BundlerArg("bundler",
               cl::desc("Path to clang-offload-bundler. Defaults to "
                        "<dirname(hip-clang)>/clang-offload-bundler."));

static cl::list<std::string>
    IncludeDirs("include-dir",
                cl::desc("Path passed via -isystem to every clang invocation. "
                         "Repeatable."));

static cl::list<std::string>
    Defines("define",
            cl::desc("`NAME[=VALUE]` passed via -D to every clang invocation. "
                     "Repeatable."));

static cl::opt<std::string> HipIncludeDir(
    "hip-include-dir",
    cl::desc("Path passed via -idirafter for HIP runtime headers."),
    cl::Required);

static cl::opt<unsigned>
    HipStdVal("hip-std",
              cl::desc("C++ standard to compile with (passed as "
                       "-std=c++<N>). Default: 20."),
              cl::init(20));

static cl::list<std::string> ExtraClangFlags(
    "extra-clang-flag",
    cl::desc("Extra flag forwarded verbatim to every per-slice clang "
             "invocation. Repeatable. Placed after the driver's "
             "foundational flags (-x hip, --cuda-device-only, "
             "--offload-arch=, -m{wave,cumode}, -std=, -O3, "
             "-fpass-plugin=) and before the -D / -isystem / "
             "-idirafter block — so the user can extend includes / "
             "defines / arbitrary clang knobs without quietly "
             "disabling the foundational ones."));

static cl::opt<std::string> Output("output", cl::desc("Output `.hipfb` path."),
                                   cl::Required);

static cl::list<std::string>
    Sources(cl::Positional, cl::desc("<HIP source files>"), cl::OneOrMore);

//===----------------------------------------------------------------------===//
// Knob model
//===----------------------------------------------------------------------===//

enum class Knob : unsigned {
  WavefrontSize64 = 0,
  CuMode = 1,
  Xnack = 2,
  SramEcc = 3,
};
static constexpr unsigned KnobCount = 4;

static StringRef knobName(Knob K) {
  switch (K) {
  case Knob::WavefrontSize64:
    return "wavefrontsize64";
  case Knob::CuMode:
    return "cumode";
  case Knob::Xnack:
    return "xnack";
  case Knob::SramEcc:
    return "sramecc";
  }
  llvm_unreachable("invalid Knob");
}

static std::optional<Knob> knobFromName(StringRef N) {
  if (N == "wavefrontsize64")
    return Knob::WavefrontSize64;
  if (N == "cumode")
    return Knob::CuMode;
  if (N == "xnack")
    return Knob::Xnack;
  if (N == "sramecc")
    return Knob::SramEcc;
  return std::nullopt;
}

/// Maps each knob to the AMDGCN `FEATURE_*` bit that gates whether
/// the knob is variable on a given arch.
static unsigned knobFeatureBit(Knob K) {
  switch (K) {
  case Knob::WavefrontSize64:
    return AMDGPU::FEATURE_WAVE32;
  case Knob::CuMode:
    return AMDGPU::FEATURE_WGP;
  case Knob::Xnack:
    return AMDGPU::FEATURE_XNACK;
  case Knob::SramEcc:
    return AMDGPU::FEATURE_SRAMECC;
  }
  llvm_unreachable("invalid Knob");
}

static bool archSupportsKnob(StringRef Arch, Knob K) {
  auto GK = AMDGPU::parseArchAMDGCN(Arch);
  if (GK == AMDGPU::GK_NONE)
    return false;
  return (AMDGPU::getArchAttrAMDGCN(GK) & knobFeatureBit(K)) != 0;
}

//===----------------------------------------------------------------------===//
// ISASpec — one slice to compile
//===----------------------------------------------------------------------===//

struct ISASpec {
  std::string Arch;
  std::array<std::optional<bool>, KnobCount> Vals{std::nullopt, std::nullopt,
                                                  std::nullopt, std::nullopt};

  std::optional<bool> get(Knob K) const {
    return Vals[static_cast<unsigned>(K)];
  }
  void set(Knob K, std::optional<bool> V) {
    Vals[static_cast<unsigned>(K)] = V;
  }
};

/// `gfxN[:xnack±][:sramecc±][:wavefrontsize64±][:cumode±]`. Only knobs
/// with an explicit value appear. Used for the outer bundle label.
static std::string targetIDFor(const ISASpec &S) {
  std::string Out = S.Arch;
  auto append = [&](Knob K) {
    if (auto V = S.get(K)) {
      Out += ':';
      Out += knobName(K).str();
      Out += *V ? '+' : '-';
    }
  };
  // AMDGPU convention: xnack/sramecc first, then our Luthier additions.
  append(Knob::Xnack);
  append(Knob::SramEcc);
  append(Knob::WavefrontSize64);
  append(Knob::CuMode);
  return Out;
}

/// Per-spec clang flags. xnack/sramecc go into `--offload-arch=`
/// (clang's whitelist); wave/cumode go as standalone `-m` flags
/// (clang rejects them in the target ID).
static SmallVector<std::string> clangFlagsFor(const ISASpec &S) {
  SmallVector<std::string> Out;
  std::string OffloadArch = "--offload-arch=" + S.Arch;
  if (auto V = S.get(Knob::Xnack))
    OffloadArch += *V ? ":xnack+" : ":xnack-";
  if (auto V = S.get(Knob::SramEcc))
    OffloadArch += *V ? ":sramecc+" : ":sramecc-";
  Out.push_back(std::move(OffloadArch));
  if (auto V = S.get(Knob::WavefrontSize64))
    Out.emplace_back(*V ? "-mwavefrontsize64" : "-mno-wavefrontsize64");
  if (auto V = S.get(Knob::CuMode))
    Out.emplace_back(*V ? "-mcumode" : "-mno-cumode");
  return Out;
}

//===----------------------------------------------------------------------===//
// Policy + matrix expansion
//===----------------------------------------------------------------------===//

enum class KnobPolicy { Default, Matrix, Plus, Minus };

struct PolicySet {
  std::array<KnobPolicy, KnobCount> P{KnobPolicy::Default, KnobPolicy::Default,
                                      KnobPolicy::Default, KnobPolicy::Default};
  KnobPolicy &operator[](Knob K) { return P[static_cast<unsigned>(K)]; }
  KnobPolicy operator[](Knob K) const { return P[static_cast<unsigned>(K)]; }
};

/// Parses one `--pin=` argument like `cumode-` into (knob, plus). Returns
/// false on syntactic failure.
static bool parsePin(StringRef Arg, Knob &K, bool &Plus) {
  if (Arg.empty())
    return false;
  char Sign = Arg.back();
  if (Sign != '+' && Sign != '-')
    return false;
  auto N = knobFromName(Arg.drop_back());
  if (!N)
    return false;
  K = *N;
  Plus = (Sign == '+');
  return true;
}

static int buildPolicyFromArgs(PolicySet &Policy) {
  for (auto &V : VaryArgs) {
    auto K = knobFromName(V);
    if (!K) {
      errs() << "luthier-tool-compile: --vary='" << V
             << "': unknown knob name\n";
      return 1;
    }
    Policy[*K] = KnobPolicy::Matrix;
  }
  for (auto &Arg : PinArgs) {
    Knob K;
    bool Plus;
    if (!parsePin(Arg, K, Plus)) {
      errs() << "luthier-tool-compile: --pin='" << Arg
             << "': expected <knob>+ or <knob>-\n";
      return 1;
    }
    Policy[K] = Plus ? KnobPolicy::Plus : KnobPolicy::Minus;
  }
  return 0;
}

/// Append every ISASpec implied by \p Policy on \p Arch to \p Out.
/// Cartesian product across the four knobs walked in enum order so
/// output is deterministic.
static int expandMatrixFor(StringRef Arch, const PolicySet &Policy,
                           SmallVectorImpl<ISASpec> &Out) {
  if (AMDGPU::parseArchAMDGCN(Arch) == AMDGPU::GK_NONE) {
    errs() << "luthier-tool-compile: unknown AMDGCN arch '" << Arch << "'\n";
    return 1;
  }

  // Per-knob value list.
  std::array<SmallVector<std::optional<bool>, 2>, KnobCount> Vals;
  for (unsigned I = 0; I < KnobCount; ++I) {
    Knob K = static_cast<Knob>(I);
    bool Sup = archSupportsKnob(Arch, K);
    KnobPolicy P = Policy[K];
    if (!Sup && P != KnobPolicy::Default) {
      errs() << "luthier-tool-compile: arch '" << Arch
             << "' does not support knob '" << knobName(K) << "'\n";
      return 1;
    }
    if (!Sup) {
      Vals[I] = {std::nullopt};
      continue;
    }
    switch (P) {
    case KnobPolicy::Default:
      // wave/cumode default = matrix; xnack/sramecc default = unspecified
      // (clang emits `_ANY_V4` ELF flag bits → one slice covers both
      // hardware states).
      if (K == Knob::WavefrontSize64 || K == Knob::CuMode)
        Vals[I] = {true, false};
      else
        Vals[I] = {std::nullopt};
      break;
    case KnobPolicy::Matrix:
      Vals[I] = {true, false};
      break;
    case KnobPolicy::Plus:
      Vals[I] = {true};
      break;
    case KnobPolicy::Minus:
      Vals[I] = {false};
      break;
    }
  }

  SmallVector<ISASpec> Cur;
  Cur.emplace_back();
  Cur.back().Arch = Arch.str();
  for (unsigned I = 0; I < KnobCount; ++I) {
    SmallVector<ISASpec> Next;
    Next.reserve(Cur.size() * Vals[I].size());
    for (const ISASpec &Base : Cur) {
      for (auto V : Vals[I]) {
        ISASpec Copy = Base;
        Copy.set(static_cast<Knob>(I), V);
        Next.push_back(std::move(Copy));
      }
    }
    Cur = std::move(Next);
  }
  for (auto &S : Cur)
    Out.push_back(std::move(S));
  return 0;
}

//===----------------------------------------------------------------------===//
// Subprocess execution
//===----------------------------------------------------------------------===//

/// Forks `Prog` with `Args`, waits, returns the exit code. Writes the
/// full argv to stderr first so build logs reproduce what happened.
static int runCmd(StringRef Prog, ArrayRef<std::string> Args) {
  SmallVector<StringRef> Refs;
  Refs.reserve(Args.size() + 1);
  Refs.push_back(Prog);
  for (const auto &A : Args)
    Refs.push_back(A);

  errs() << "+";
  for (auto R : Refs)
    errs() << ' ' << R;
  errs() << '\n';

  std::string ErrMsg;
  int RC = sys::ExecuteAndWait(Prog, Refs, /*Env=*/std::nullopt,
                               /*Redirects=*/{}, /*SecondsToWait=*/0,
                               /*MemoryLimit=*/0, &ErrMsg);
  if (RC != 0) {
    errs() << "luthier-tool-compile: command '" << Prog << "' failed (rc=" << RC
           << ")";
    if (!ErrMsg.empty())
      errs() << ": " << ErrMsg;
    errs() << '\n';
  }
  return RC;
}

//===----------------------------------------------------------------------===//
// Phase 1: compile one .co per ISA spec
//===----------------------------------------------------------------------===//

/// On success: writes the produced `.co` path to \p PerCoOut.
static int compileSpec(const ISASpec &Spec, StringRef OutDir, size_t Idx,
                       std::string &PerCoOut) {
  SmallString<256> CoPath(OutDir);
  sys::path::append(CoPath, llvm::Twine(Idx) + "." + Spec.Arch + ".co");
  PerCoOut = std::string(CoPath);

  SmallVector<std::string> Argv;
  Argv.emplace_back("-x");
  Argv.emplace_back("hip");
  Argv.emplace_back("--cuda-device-only");
  for (auto &F : clangFlagsFor(Spec))
    Argv.push_back(std::move(F));
  Argv.emplace_back("-std=c++" + std::to_string(HipStdVal.getValue()));
  // -O3 is non-optional: without it the IR plugin's TargetModulePatcherPass
  // hits AMDGPUResourceUsageAnalysisImpl::analyzeResourceUsage on pre-RA
  // virtual registers cloned from out-of-line device helpers.
  Argv.emplace_back("-O3");
  Argv.emplace_back("-fpass-plugin=" + PluginIR.getValue());
  // User pass-through. After the foundational flags so they take
  // precedence on `-m`/`-O` style toggles; before the include/define
  // block so a user `-D` or `-isystem` is "later" than the helper's
  // and wins on duplicate-key collisions.
  for (auto &F : ExtraClangFlags)
    Argv.push_back(F);
  for (auto &D : Defines)
    Argv.emplace_back("-D" + D);
  for (auto &I : IncludeDirs) {
    Argv.emplace_back("-isystem");
    Argv.push_back(I);
  }
  Argv.emplace_back("-idirafter");
  Argv.push_back(HipIncludeDir.getValue());
  Argv.emplace_back("-c");
  for (auto &S : Sources)
    Argv.push_back(S);
  Argv.emplace_back("-o");
  Argv.push_back(PerCoOut);

  return runCmd(HipClang, Argv);
}

//===----------------------------------------------------------------------===//
// Phase 2: unbundle the AMDGCN slice from each per-spec .co
//===----------------------------------------------------------------------===//

static int unbundleSlice(StringRef Bundler, StringRef Co, StringRef Arch,
                         StringRef OutSlice) {
  SmallVector<std::string> Argv;
  Argv.emplace_back("--type=o");
  Argv.emplace_back("--unbundle");
  Argv.emplace_back(("--targets=hipv4-amdgcn-amd-amdhsa--" + Arch).str());
  Argv.emplace_back("--input=" + Co.str());
  Argv.emplace_back("--output=" + OutSlice.str());
  return runCmd(Bundler, Argv);
}

//===----------------------------------------------------------------------===//
// Phase 3: rebundle the extracted slices + host placeholder
//===----------------------------------------------------------------------===//

static int rebundle(StringRef Bundler, ArrayRef<ISASpec> Specs,
                    ArrayRef<std::string> SlicePaths, StringRef OutFatbin) {
  // Host placeholder label uses the build host's triple.
  std::string Targets = "--targets=host-" + sys::getDefaultTargetTriple();
  for (const auto &S : Specs) {
    Targets += ",hipv4-amdgcn-amd-amdhsa--";
    Targets += targetIDFor(S);
  }

  SmallVector<std::string> Argv;
  Argv.emplace_back("--type=o");
  Argv.push_back(std::move(Targets));
  Argv.emplace_back("--input=/dev/null");
  for (const auto &SP : SlicePaths)
    Argv.emplace_back("--input=" + SP);
  Argv.emplace_back("--output=" + OutFatbin.str());
  /// Pad each slice's offset to 8 bytes so LLVM's ELF parser, which
  /// requires \c alignof(uint64_t) at the buffer start, can read the
  /// AMDGCN slice in-place without an aligned-copy in the loader. The
  /// default bundler packs slices immediately after the metadata, which
  /// frequently lands them at non-8 offsets (e.g. 0x8d when the two
  /// triples sum to 141 metadata bytes).
  Argv.emplace_back("--bundle-align=8");

  return runCmd(Bundler, Argv);
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "luthier-tool-compile: build a Luthier tool fat binary across an "
      "(arch, wave, cumode, xnack, sramecc) matrix.\n");

  if (Archs.empty()) {
    errs() << "luthier-tool-compile: at least one --arch= is required\n";
    return 1;
  }

  // Resolve bundler path. Defaults to the clang++'s sibling.
  std::string Bundler;
  if (!BundlerArg.empty()) {
    Bundler = BundlerArg.c_str();
  } else {
    SmallString<256> P(HipClang.getValue());
    sys::path::remove_filename(P);
    sys::path::append(P, "clang-offload-bundler");
    Bundler = std::string(P);
  }

  // Resolve output's parent dir; intermediates land there.
  SmallString<256> OutDir(Output.getValue());
  sys::path::remove_filename(OutDir);
  if (OutDir.empty())
    OutDir = ".";

  // Apply --vary / --pin to the default policy.
  PolicySet Policy;
  if (int RC = buildPolicyFromArgs(Policy))
    return RC;

  // Expand the matrix across all requested arches.
  SmallVector<ISASpec> Specs;
  for (const auto &A : Archs) {
    if (int RC = expandMatrixFor(A, Policy, Specs))
      return RC;
  }
  if (Specs.empty()) {
    errs() << "luthier-tool-compile: expansion produced 0 slices\n";
    return 1;
  }
  errs() << "luthier-tool-compile: " << Specs.size()
         << " slice(s) to compile\n";

  // Compile + extract per-spec.
  SmallVector<std::string> StrippedSlices;
  StrippedSlices.reserve(Specs.size());
  for (size_t I = 0; I < Specs.size(); ++I) {
    std::string PerCo;
    if (int RC = compileSpec(Specs[I], OutDir, I, PerCo))
      return RC;
    std::string Stripped = PerCo + ".amdgcn.o";
    if (int RC = unbundleSlice(Bundler, PerCo, Specs[I].Arch, Stripped))
      return RC;
    StrippedSlices.push_back(std::move(Stripped));
  }

  // Final bundle.
  return rebundle(Bundler, Specs, StrippedSlices, Output);
}
