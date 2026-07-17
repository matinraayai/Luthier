//===-- llc.cpp - Luthier's LLVM LLC Fork ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// A fork of LLVM's llc code generator driver modified for running Luthier's
/// instrument prototype passes with support for plugins.
///
//===----------------------------------------------------------------------===//

#include "luthier/Common/Debug.h"
#include "luthier/LLVM/streams.h"
#include "luthier/PassPlugin/LuthierPassPlugin.h"
#include "luthier/ToolCodeGen/InstrumentPrototype.h"
#include "luthier/ToolCodeGen/InstrumentPrototypePassBuilder.h"
#include "luthier/ToolCodeGenTesting/LuthierFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineVerifier.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PGOOptions.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>
#include <memory>
#include <optional>
using namespace llvm;

namespace {

enum class VerifierKind { None, InputOutput, EachPass };

struct LLCDiagnosticHandler : public DiagnosticHandler {
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    DiagnosticHandler::handleDiagnostics(DI);
    if (DI.getKind() == DK_SrcMgr) {
      const auto &DISM = cast<DiagnosticInfoSrcMgr>(DI);
      const SMDiagnostic &SMD = DISM.getSMDiag();

      SMD.print(nullptr, errs());

      if (DISM.isInlineAsmDiag() && DISM.getLocCookie())
        WithColor::note() << "!srcloc = " << DISM.getLocCookie() << "\n";

      return true;
    }

    if (auto *Remark = dyn_cast<DiagnosticInfoOptimizationBase>(&DI))
      if (!Remark->isEnabled())
        return true;

    DiagnosticPrinterRawOStream DP(errs());
    errs() << LLVMContext::getDiagnosticMessagePrefix(DI.getSeverity()) << ": ";
    DI.print(DP);
    errs() << "\n";
    return true;
  }
};

} // namespace

static cl::opt<RegAllocType, false, RegAllocTypeParser>
    RegAlloc("regalloc-npm",
             cl::desc("Register allocator to use for new pass manager"),
             cl::Hidden, cl::init(RegAllocType::Unset));

static cl::opt<bool>
    DebugPM("debug-pass-manager", cl::Hidden,
            cl::desc("Print pass management debugging information"));

static codegen::RegisterCodeGenFlags CGF;
static codegen::RegisterSaveStatsFlag SSF;

// General options for llc.  Other pass-specific options are specified
// within the corresponding llc passes, and target-specific options
// and back-end code generation options are specified with the target machine.
//
static cl::opt<std::string>
    InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init(""));

static cl::list<std::string>
    InstPrinterOptions("M", cl::desc("InstPrinter options"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"));

static cl::opt<std::string>
    SplitDwarfOutputFile("split-dwarf-output", cl::desc(".dwo output filename"),
                         cl::value_desc("filename"));

static cl::opt<unsigned>
    TimeCompilations("time-compilations", cl::Hidden, cl::init(1u),
                     cl::value_desc("N"),
                     cl::desc("Repeat compilation N times for timing"));

static cl::opt<bool> TimeTrace("time-trace", cl::desc("Record time trace"));

static cl::opt<unsigned> TimeTraceGranularity(
    "time-trace-granularity",
    cl::desc(
        "Minimum time granularity (in microseconds) traced by time profiler"),
    cl::init(500), cl::Hidden);

static cl::opt<std::string>
    TimeTraceFile("time-trace-file",
                  cl::desc("Specify time trace file destination"),
                  cl::value_desc("filename"));

static cl::opt<std::string>
    BinutilsVersion("binutils-version", cl::Hidden,
                    cl::desc("Produced object files can use all ELF features "
                             "supported by this binutils version and newer."
                             "If -no-integrated-as is specified, the generated "
                             "assembly will consider GNU as support."
                             "'none' means that all ELF features can be used, "
                             "regardless of binutils support"));

static cl::opt<bool>
    PreserveComments("preserve-as-comments", cl::Hidden,
                     cl::desc("Preserve Comments in outputted assembly"),
                     cl::init(true));

// Determine optimization level.
static cl::opt<char>
    OptLevel("O",
             cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                      "(default = '-O2')"),
             cl::Prefix, cl::init('2'));

static cl::opt<std::string>
    TargetTriple("mtriple", cl::desc("Override target triple for module"));

static cl::opt<std::string> SplitDwarfFile(
    "split-dwarf-file",
    cl::desc(
        "Specify the name of the .dwo file to encode in the DWARF output"));

static cl::opt<bool> NoVerify("disable-verify", cl::Hidden,
                              cl::desc("Do not verify input module"));

static cl::opt<bool> VerifyEach("verify-each",
                                cl::desc("Verify after each transform"));

static cl::opt<bool>
    DisableSimplifyLibCalls("disable-simplify-libcalls",
                            cl::desc("Disable simplify-libcalls"));

static cl::opt<bool> ShowMCEncoding("show-mc-encoding", cl::Hidden,
                                    cl::desc("Show encoding in .s output"));

static cl::opt<unsigned>
    OutputAsmVariant("output-asm-variant",
                     cl::desc("Syntax variant to use for output printing"));

static cl::opt<bool>
    DwarfDirectory("dwarf-directory", cl::Hidden,
                   cl::desc("Use .file directives with an explicit directory"),
                   cl::init(true));

static cl::opt<bool> AsmVerbose("asm-verbose",
                                cl::desc("Add comments to directives."),
                                cl::init(true));

static cl::opt<bool> DiscardValueNames(
    "discard-value-names",
    cl::desc("Discard names from Value (other than GlobalValue)."),
    cl::init(false), cl::Hidden);

static cl::opt<bool>
    PrintMIR2VecVocab("print-mir2vec-vocab", cl::Hidden,
                      cl::desc("Print MIR2Vec vocabulary contents"),
                      cl::init(false));

static cl::opt<bool>
    PrintMIR2Vec("print-mir2vec", cl::Hidden,
                 cl::desc("Print MIR2Vec embeddings for functions"),
                 cl::init(false));

static cl::list<std::string> IncludeDirs("I", cl::desc("include search path"));

static cl::opt<bool> RemarksWithHotness(
    "pass-remarks-with-hotness",
    cl::desc("With PGO, include profile count in optimization remarks"),
    cl::Hidden);

static cl::opt<std::optional<uint64_t>, false, remarks::HotnessThresholdParser>
    RemarksHotnessThreshold(
        "pass-remarks-hotness-threshold",
        cl::desc("Minimum profile count required for "
                 "an optimization remark to be output. "
                 "Use 'auto' to apply the threshold from profile summary."),
        cl::value_desc("N or 'auto'"), cl::init(0), cl::Hidden);

static cl::opt<std::string>
    RemarksFilename("pass-remarks-output",
                    cl::desc("Output filename for pass remarks"),
                    cl::value_desc("filename"));

static cl::opt<std::string>
    RemarksPasses("pass-remarks-filter",
                  cl::desc("Only record optimization remarks from passes whose "
                           "names match the given regular expression"),
                  cl::value_desc("regex"));

static cl::opt<std::string> RemarksFormat(
    "pass-remarks-format",
    cl::desc("The format used for serializing remarks (default: YAML)"),
    cl::value_desc("format"), cl::init("yaml"));

static cl::list<std::string> PassPlugins("load-pass-plugin",
                                         cl::desc("Load plugin library"));

// This flag specifies a textual description of the optimization pass pipeline
// to run over the Instrument Prototype. It requires explicit target(...) or
// instrumentation(...) wrapping of the inner pipeline.
static cl::opt<std::string> PassPipeline(
    "passes",
    cl::desc(
        "A textual description of the pass pipeline for TAIM. "
        "Requires explicit 'target(...)' or 'instrumentation(...)' wrapping."));
static cl::alias PassPipeline2("p", cl::aliasopt(PassPipeline),
                               cl::desc("Alias for -passes"));

// PGO command line options
enum PGOKind {
  NoPGO,
  SampleUse,
};

static cl::opt<PGOKind>
    PGOKindFlag("pgo-kind", cl::init(NoPGO), cl::Hidden,
                cl::desc("The kind of profile guided optimization"),
                cl::values(clEnumValN(NoPGO, "nopgo", "Do not use PGO."),
                           clEnumValN(SampleUse, "pgo-sample-use-pipeline",
                                      "Use sampled profile to guide PGO.")));

// Function to set PGO options on TargetMachine based on command line flags.
static void setPGOOptions(TargetMachine &TM) {
  std::optional<PGOOptions> PGOOpt;

  switch (PGOKindFlag) {
  case SampleUse:
    // Use default values for other PGOOptions parameters. This parameter
    // is used to test that PGO data is preserved at -O0.
    PGOOpt = PGOOptions("", "", "", "", PGOOptions::SampleUse,
                        PGOOptions::NoCSAction);
    break;
  case NoPGO:
    PGOOpt = std::nullopt;
    break;
  }

  if (PGOOpt)
    TM.setPGOOption(PGOOpt);
}

static int compileModule(char **argv,
                         SmallVectorImpl<luthier::PassPlugin> &,
                         LLVMContext &Context, std::string &OutputFilename);

[[noreturn]] static void reportError(Twine Msg, StringRef Filename = "") {
  SmallString<256> Prefix;
  if (!Filename.empty()) {
    if (Filename == "-")
      Filename = "<stdin>";
    ("'" + Twine(Filename) + "': ").toStringRef(Prefix);
  }
  WithColor::error(errs(), "llc") << Prefix << Msg << "\n";
  exit(1);
}

[[noreturn]] static void reportError(Error Err, StringRef Filename) {
  assert(Err);
  handleAllErrors(createFileError(Filename, std::move(Err)),
                  [&](const ErrorInfoBase &EI) { reportError(EI.message()); });
  llvm_unreachable("reportError() should not return");
}

static std::unique_ptr<ToolOutputFile> GetOutputStream(Triple::OSType OS) {
  // If we don't yet have an output filename, make one.
  if (OutputFilename.empty()) {
    if (InputFilename.empty() || InputFilename == "-")
      OutputFilename = "-";
    else {
      // Strip a recognized input suffix and append one based on
      // codegen::getFileType() (only .s / .o / - reachable here — .luthier
      // output requires an explicit -o foo.luthier).
      StringRef IFN = InputFilename;
      if (IFN.ends_with(".luthier"))
        OutputFilename = std::string(IFN.drop_back(8));
      else
        OutputFilename = std::string(IFN);

      switch (codegen::getFileType()) {
      case CodeGenFileType::AssemblyFile:
        OutputFilename += ".s";
        break;
      case CodeGenFileType::ObjectFile:
        if (OS == Triple::Win32)
          OutputFilename += ".obj";
        else
          OutputFilename += ".o";
        break;
      case CodeGenFileType::Null:
        OutputFilename = "-";
        break;
      }
    }
  }

  bool EmitsLuthier = StringRef(OutputFilename).ends_with(".luthier");

  // Decide if we need "binary" output.  .luthier is always text (YAML).
  bool Binary = false;
  if (!EmitsLuthier) {
    switch (codegen::getFileType()) {
    case CodeGenFileType::AssemblyFile:
      break;
    case CodeGenFileType::ObjectFile:
    case CodeGenFileType::Null:
      Binary = true;
      break;
    }
  }

  // Open the file.
  std::error_code EC;
  sys::fs::OpenFlags OpenFlags = sys::fs::OF_None;
  if (!Binary)
    OpenFlags |= sys::fs::OF_TextWithCRLF;
  auto FDOut = std::make_unique<ToolOutputFile>(OutputFilename, EC, OpenFlags);
  if (EC)
    reportError(EC.message());
  return FDOut;
}

// main - Entry point for the llc compiler.
//
int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Enable debug stream buffering.
  EnableDebugBuffering = true;

  // Initialize targets first, so that --version shows registered targets.
  InitializeAllTargets();
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();
  InitializeAllTargetMCAs();

  // Initialize codegen and IR passes used by llc so that the -print-after,
  // -print-before, and -stop-after options work.
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializePostInlineEntryExitInstrumenterPass(*Registry);
  initializeUnreachableBlockElimLegacyPassPass(*Registry);
  initializeConstantHoistingLegacyPassPass(*Registry);
  initializeScalarOpts(*Registry);
  initializeVectorization(*Registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
  initializeExpandReductionsPass(*Registry);
  initializeHardwareLoopsLegacyPass(*Registry);
  initializeTransformUtils(*Registry);
  initializeReplaceWithVeclibLegacyPass(*Registry);

  // Initialize debugging passes.
  initializeScavengerTestPass(*Registry);

  SmallVector<luthier::PassPlugin, 1> PluginList;
  PassPlugins.setCallback([&](const std::string &PluginPath) {
    auto Plugin = luthier::PassPlugin::Load(PluginPath);
    if (!Plugin)
      reportFatalUsageError(Plugin.takeError());
    PluginList.emplace_back(std::move(*Plugin));
  });

  // Register the Target and CPU printer for --version.
  cl::AddExtraVersionPrinter(sys::printDefaultTargetAndDetectedCPU);
  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  luthier::registerDebugCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "llvm system compiler\n");

  if (TimeTrace)
    timeTraceProfilerInitialize(TimeTraceGranularity, argv[0]);
  auto TimeTraceScopeExit = make_scope_exit([]() {
    if (TimeTrace) {
      if (auto E = timeTraceProfilerWrite(TimeTraceFile, OutputFilename)) {
        handleAllErrors(std::move(E), [&](const StringError &SE) {
          errs() << SE.getMessage() << "\n";
        });
        return;
      }
      timeTraceProfilerCleanup();
    }
  });

  LLVMContext Context;
  Context.setDiscardValueNames(DiscardValueNames);

  // Set a diagnostic handler that doesn't exit on the first error
  Context.setDiagnosticHandler(std::make_unique<LLCDiagnosticHandler>());

  Expected<LLVMRemarkFileHandle> RemarksFileOrErr =
      setupLLVMOptimizationRemarks(Context, RemarksFilename, RemarksPasses,
                                   RemarksFormat, RemarksWithHotness,
                                   RemarksHotnessThreshold);
  if (Error E = RemarksFileOrErr.takeError())
    reportError(std::move(E), RemarksFilename);
  LLVMRemarkFileHandle RemarksFile = std::move(*RemarksFileOrErr);

  codegen::MaybeEnableStatistics();
  std::string OutputFilename;

  // Compile the module TimeCompilations times to give better compile time
  // metrics.
  for (unsigned I = TimeCompilations; I; --I)
    if (int RetVal = compileModule(argv, PluginList, Context, OutputFilename))
      return RetVal;

  if (RemarksFile)
    RemarksFile->keep();

  return codegen::MaybeSaveStatistics(OutputFilename, "llc");
}

static int compileModule(char **argv,
                         SmallVectorImpl<luthier::PassPlugin> &PluginList,
                         LLVMContext &Context, std::string &OutputFilename) {
  Triple TheTriple;
  std::string CPUStr = codegen::getCPUStr(),
              FeaturesStr = codegen::getFeaturesStr();

  // Set attributes on functions as loaded from MIR from command line arguments.
  auto setMIRFunctionAttributes = [&CPUStr, &FeaturesStr](Function &F) {
    codegen::setFunctionAttributes(F, CPUStr, FeaturesStr);
  };

  auto MAttrs = codegen::getMAttrs();
  bool SkipModule =
      CPUStr == "help" || (!MAttrs.empty() && MAttrs.front() == "help");

  CodeGenOptLevel OLvl;
  if (auto Level = CodeGenOpt::parseLevel(OptLevel)) {
    OLvl = *Level;
  } else {
    WithColor::error(errs(), argv[0]) << "invalid optimization level.\n";
    return 1;
  }

  // Parse 'none' or '$major.$minor'. Disallow -binutils-version=0 because we
  // use that to indicate the MC default.
  if (!BinutilsVersion.empty() && BinutilsVersion != "none") {
    StringRef V = BinutilsVersion.getValue();
    unsigned Num;
    if (V.consumeInteger(10, Num) || Num == 0 ||
        !(V.empty() ||
          (V.consume_front(".") && !V.consumeInteger(10, Num) && V.empty()))) {
      WithColor::error(errs(), argv[0])
          << "invalid -binutils-version, accepting 'none' or major.minor\n";
      return 1;
    }
  }
  TargetOptions Options;
  auto InitializeOptions = [&](const Triple &TheTriple) {
    Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);

    if (Options.XCOFFReadOnlyPointers) {
      if (!TheTriple.isOSAIX())
        reportError("-mxcoff-roptr option is only supported on AIX",
                    InputFilename);

      // Since the storage mapping class is specified per csect,
      // without using data sections, it is less effective to use read-only
      // pointers. Using read-only pointers may cause other RO variables in the
      // same csect to become RW when the linker acts upon `-bforceimprw`;
      // therefore, we require that separate data sections are used in the
      // presence of ReadOnlyPointers. We respect the setting of data-sections
      // since we have not found reasons to do otherwise that overcome the user
      // surprise of not respecting the setting.
      if (!Options.DataSections)
        reportError("-mxcoff-roptr option must be used with -data-sections",
                    InputFilename);
    }

    if (TheTriple.isX86() &&
        codegen::getFuseFPOps() != FPOpFusion::FPOpFusionMode::Standard)
      WithColor::warning(errs(), argv[0])
          << "X86 backend ignores --fp-contract setting; use IR fast-math "
             "flags instead.";

    Options.BinutilsVersion =
        TargetMachine::parseBinutilsVersion(BinutilsVersion);
    Options.MCOptions.ShowMCEncoding = ShowMCEncoding;
    Options.MCOptions.AsmVerbose = AsmVerbose;
    Options.MCOptions.PreserveAsmComments = PreserveComments;
    if (OutputAsmVariant.getNumOccurrences())
      Options.MCOptions.OutputAsmVariant = OutputAsmVariant;
    Options.MCOptions.IASSearchPaths = IncludeDirs;
    Options.MCOptions.InstPrinterOptions = InstPrinterOptions;
    Options.MCOptions.SplitDwarfFile = SplitDwarfFile;
    if (DwarfDirectory.getPosition()) {
      Options.MCOptions.MCUseDwarfDirectory =
          DwarfDirectory ? MCTargetOptions::EnableDwarfDirectory
                         : MCTargetOptions::DisableDwarfDirectory;
    } else {
      // -dwarf-directory is not set explicitly. Some assemblers
      // (e.g. GNU as or ptxas) do not support `.file directory'
      // syntax prior to DWARFv5. Let the target decide the default
      // value.
      Options.MCOptions.MCUseDwarfDirectory =
          MCTargetOptions::DefaultDwarfDirectory;
    }
  };

  std::optional<Reloc::Model> RM = codegen::getExplicitRelocModel();
  std::optional<CodeModel::Model> CM = codegen::getExplicitCodeModel();

  const Target *TheTarget = nullptr;
  std::unique_ptr<TargetMachine> Target;
  std::unique_ptr<luthier::InstrumentPrototype> IP;
  std::unique_ptr<MIRParser> TargetMIRParser;
  std::unique_ptr<MIRParser> IModuleMIRParser;

  auto SetDataLayout = [&](StringRef DataLayoutTargetTriple,
                           StringRef OldDLStr) -> std::optional<std::string> {
    std::string IRTargetTriple = DataLayoutTargetTriple.str();
    if (!TargetTriple.empty())
      IRTargetTriple = Triple::normalize(TargetTriple);
    TheTriple = Triple(IRTargetTriple);
    if (TheTriple.getTriple().empty())
      TheTriple.setTriple(sys::getDefaultTargetTriple());

    std::string Error;
    TheTarget =
        TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) {
      WithColor::error(errs(), argv[0]) << Error << "\n";
      exit(1);
    }

    InitializeOptions(TheTriple);
    Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
        TheTriple, CPUStr, FeaturesStr, Options, RM, CM, OLvl));
    assert(Target && "Could not allocate target machine!");
    setPGOOptions(*Target);
    return Target->createDataLayout().getStringRepresentation();
  };

  // Only two supported input shapes: an empty input (synthesize an empty
  // InstrumentPrototype from -mtriple) or a .luthier file.
  if (InputFilename.empty()) {
    if (TargetTriple.empty()) {
      WithColor::error(errs(), argv[0])
          << "no input file: -mtriple is required\n";
      return 1;
    }
    TheTriple = Triple(Triple::normalize(TargetTriple));
    std::string Error;
    TheTarget =
        TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) {
      WithColor::error(errs(), argv[0]) << Error << "\n";
      return 1;
    }
    InitializeOptions(TheTriple);
    Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
        TheTriple, CPUStr, FeaturesStr, Options, RM, CM, OLvl));
    assert(Target && "Could not allocate target machine!");
    setPGOOptions(*Target);

    auto TargetM = std::make_unique<Module>("", Context);
    TargetM->setTargetTriple(TheTriple);
    TargetM->setDataLayout(Target->createDataLayout());
    auto IModuleM = std::make_unique<Module>("luthier-instrumentation", Context);
    IModuleM->setTargetTriple(TheTriple);
    IModuleM->setDataLayout(Target->createDataLayout());
    IP = std::make_unique<luthier::InstrumentPrototype>(std::move(TargetM),
                                                        std::move(IModuleM));
  } else if (SkipModule) {
    // -mcpu=help / -mattr=help: don't parse the module.
    TheTriple = Triple(Triple::normalize(TargetTriple));
    if (TheTriple.getTriple().empty())
      TheTriple.setTriple(sys::getDefaultTargetTriple());

    std::string Error;
    TheTarget =
        TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
    if (!TheTarget) {
      WithColor::error(errs(), argv[0]) << Error << "\n";
      return 1;
    }
    InitializeOptions(TheTriple);
    Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
        TheTriple, CPUStr, FeaturesStr, Options, RM, CM, OLvl));
    assert(Target && "Could not allocate target machine!");
    setPGOOptions(*Target);
    return 0;
  } else if (StringRef(InputFilename).ends_with(".luthier")) {
    auto ParserOrErr = luthier::LuthierFileParser::create(InputFilename);
    if (!ParserOrErr)
      reportError(ParserOrErr.takeError(), InputFilename);

    // A throw-away IPAM used only to satisfy the parser signature.  The
    // driver builds its own IPAM later and rebinds analyses to it.
    luthier::InstrumentPrototypeAnalysisManager ParserIPAM;
    auto LoadedOrErr = ParserOrErr->load(Context, ParserIPAM, SetDataLayout,
                                         setMIRFunctionAttributes);
    if (!LoadedOrErr)
      reportError(LoadedOrErr.takeError(), InputFilename);

    IP = std::move(LoadedOrErr->IP);
    TargetMIRParser = std::move(LoadedOrErr->TargetMIRParser);
    IModuleMIRParser = std::move(LoadedOrErr->IModuleMIRParser);

    if (!TargetTriple.empty())
      IP->getTargetModule().setTargetTriple(
          Triple(Triple::normalize(TargetTriple)));

    std::optional<CodeModel::Model> CM_IR = IP->getTargetModule().getCodeModel();
    if (!CM && CM_IR)
      Target->setCodeModel(*CM_IR);
    if (std::optional<uint64_t> LDT = codegen::getExplicitLargeDataThreshold())
      Target->setLargeDataThreshold(*LDT);
  } else {
    WithColor::error(errs(), argv[0])
        << "unsupported input file '" << InputFilename
        << "': expected a '.luthier' file or no input\n";
    return 1;
  }

  assert(IP && "should have constructed an InstrumentPrototype above");
  if (codegen::getFloatABIForCalls() != FloatABI::Default)
    Target->Options.FloatABIType = codegen::getFloatABIForCalls();

  // Figure out where we are going to send the output.
  std::unique_ptr<ToolOutputFile> Out = GetOutputStream(TheTriple.getOS());
  if (!Out)
    return 1;

  // Ensure the filename is passed down to CodeViewDebug.
  Target->Options.ObjectFilenameForDebug = Out->outputFilename();

  // Return a copy of the output filename via the output param
  OutputFilename = Out->outputFilename();

  bool EmitLuthierFile = StringRef(Out->outputFilename()).ends_with(".luthier");

  // Tell target that this tool is not necessarily used with argument ABI
  // compliance (i.e. narrow integer argument extensions).
  Target->Options.VerifyArgABICompliance = 0;

  std::unique_ptr<ToolOutputFile> DwoOut;
  if (!SplitDwarfOutputFile.empty()) {
    std::error_code EC;
    DwoOut = std::make_unique<ToolOutputFile>(SplitDwarfOutputFile, EC,
                                              sys::fs::OF_None);
    if (EC)
      reportError(EC.message(), SplitDwarfOutputFile);
  }

  // Add an appropriate TargetLibraryInfo pass for the target module's triple.
  TargetLibraryInfoImpl TLII(IP->getTargetModule().getTargetTriple(),
                             Target->Options.VecLib);

  // The -disable-simplify-libcalls flag actually disables all builtin optzns.
  if (DisableSimplifyLibCalls)
    TLII.disableAllFunctions();

  // Verify target module immediately to catch problems before
  // doInitialization() is called on any passes.
  if (!NoVerify && verifyModule(IP->getTargetModule(), &errs()))
    reportError("input module cannot be verified", InputFilename);

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  codegen::setFunctionAttributes(IP->getTargetModule(), CPUStr, FeaturesStr);

  if (mc::getExplicitRelaxAll() &&
      codegen::getFileType() != CodeGenFileType::ObjectFile)
    WithColor::warning(errs(), argv[0])
        << ": warning: ignoring -mc-relax-all because filetype != obj";

  VerifierKind VK = VerifierKind::InputOutput;
  if (NoVerify)
    VK = VerifierKind::None;
  else if (VerifyEach)
    VK = VerifierKind::EachPass;

  CodeGenFileType FileType = codegen::getFileType();
  StringRef Arg0 = argv[0];

  if (!PassPipeline.empty() && TargetPassConfig::hasLimitedCodeGenPipeline()) {
    WithColor::error(errs(), Arg0)
        << "--passes cannot be used with "
        << TargetPassConfig::getLimitedCodeGenPipelineReason() << ".\n";
    return 1;
  }

  raw_pwrite_stream *OS = &Out->os();

  std::unique_ptr<buffer_ostream> BOS;
  if (!EmitLuthierFile && FileType != CodeGenFileType::AssemblyFile &&
      !Out->os().supportsSeeking()) {
    BOS = std::make_unique<buffer_ostream>(Out->os());
    OS = BOS.get();
  }

  CGPassBuilderOption Opt = getCGPassBuilderOption();
  Opt.DisableVerify = VK != VerifierKind::InputOutput;
  Opt.DebugPM = DebugPM;
  Opt.RegAlloc = RegAlloc;

  MachineModuleInfo MMI(Target.get());

  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(Context, Opt.DebugPM,
                              VK == VerifierKind::EachPass);
  registerCodeGenCallback(PIC, *Target);

  MachineFunctionAnalysisManager MFAM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  luthier::InstrumentPrototypeAnalysisManager IPAM;

  PassBuilder PB(Target.get(), PipelineTuningOptions(), std::nullopt, &PIC);

  luthier::InstrumentPrototypePassBuilder IPPB(PB);

  for (const auto &Plugin : PluginList)
    Plugin.registerInstrumentPrototypePassBuilderCallback(IPPB);

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerMachineFunctionAnalyses(MFAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);
  SI.registerCallbacks(PIC, &MAM);

  IPPB.crossRegisterProxies(MAM, FAM, MFAM, IPAM);

  FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  MAM.registerPass([&] {
    const TargetOptions &Opts = Target->Options;
    return RuntimeLibraryAnalysis(
        IP->getTargetModule().getTargetTriple(), Target->Options.ExceptionModel,
        Target->Options.FloatABIType, Target->Options.EABIVersion,
        Opts.MCOptions.ABIName, Target->Options.VecLib);
  });
  MAM.registerPass([&] { return LibcallLoweringModuleAnalysis(); });
  MAM.registerPass([&] { return MachineModuleAnalysis(MMI); });

  luthier::InstrumentPrototypePassManager IPPM;

  if (!PassPipeline.empty()) {
    if (!IP->getTargetModule().empty() && !TargetMIRParser) {
      WithColor::error(errs(), Arg0)
          << "-passes requires a .luthier or empty input.\n";
      return 1;
    }

    if (auto Err = IPPB.parsePipeline(IPPM, PassPipeline)) {
      logAllUnhandledErrors(std::move(Err), errs(), "error: ");
      return 1;
    }

    if (!EmitLuthierFile) {
      // The user's -passes covers pipeline shape; append a MIR/AsmPrinter
      // tail so the tool still produces the requested .s / .o output.
      ModulePassManager TargetMPM;
      TargetMPM.addPass(PrintMIRPreparePass(*OS));
      MachineFunctionPassManager MFPM;
      if (VK == VerifierKind::InputOutput)
        MFPM.addPass(MachineVerifierPass());
      MFPM.addPass(PrintMIRPass(*OS));
      FunctionPassManager FPM;
      FPM.addPass(createFunctionToMachineFunctionPassAdaptor(std::move(MFPM)));
      TargetMPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

      IPPB.addTargetModulePass(IPPM, std::move(TargetMPM));
    }
  } else if (!EmitLuthierFile) {
    // No -passes and non-luthier output: run the full default target-side
    // codegen pipeline that ends in AsmPrinter/ObjectFile emission.
    ModulePassManager TargetMPM;
    ExitOnError ExitOnErr;
    ExitOnErr(Target->buildCodeGenPipeline(TargetMPM, MAM, *OS,
                                           DwoOut ? &DwoOut->os() : nullptr,
                                           FileType, Opt, MMI.getContext(),
                                           &PIC));
    IPPB.addTargetModulePass(IPPM, std::move(TargetMPM));
  }
  // When EmitLuthierFile is set and -passes is empty, the driver runs no
  // default pipeline: the tool re-emits the InstrumentPrototype as-is.

  if (PrintPipelinePasses) {
    std::string PipelineStr;
    raw_string_ostream PSO(PipelineStr);
    IPPM.printPipeline(PSO, [&PIC](StringRef ClassName) {
      auto PassName = PIC.getPassNameForClassName(ClassName);
      return PassName.empty() ? ClassName : PassName;
    });
    outs() << PipelineStr << '\n';
    return 0;
  }

  if (TargetMIRParser &&
      TargetMIRParser->parseMachineFunctions(IP->getTargetModule(), MAM))
    return 1;
  if (IModuleMIRParser &&
      IModuleMIRParser->parseMachineFunctions(IP->getInstrumentationModule(),
                                              MAM))
    return 1;

  cl::PrintOptionValues();

  IPPM.run(*IP, IPAM);

  if (Context.getDiagHandlerPtr()->HasErrors)
    return 1;

  if (EmitLuthierFile) {
    if (auto Err = luthier::writeLuthierFile(Out->os(), *IP, IPAM)) {
      logAllUnhandledErrors(std::move(Err), errs(), "error: ");
      return 1;
    }
  }

  Out->keep();
  if (DwoOut)
    DwoOut->keep();

  return 0;
}
