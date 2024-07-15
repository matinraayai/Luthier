#include "tooling_common/target_manager.hpp"

#include <AMDGPUTargetMachine.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCAsmLexer.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include "common/error.hpp"
#include "hsa/hsa_agent.hpp"

namespace luthier {

template <> TargetManager *Singleton<TargetManager>::Instance{nullptr};

TargetManager::TargetManager() : Singleton<TargetManager>() {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUTargetMCA();
  // TODO: make LLVM args separate from Luthier arguments
  auto Argv = "";
  llvm::cl::ParseCommandLineOptions(
      0, &Argv, "Luthier, An AMD GPU Binary Instrumentation Tool",
      &llvm::errs(), "LUTHIER_ARGS");
}

TargetManager::~TargetManager() {
  for (auto &It : LLVMTargetInfo) {
    delete It.second.MRI;
    delete It.second.MAI;
    delete It.second.MII;
    delete It.second.MIA;
    delete It.second.STI;
    delete It.second.IP;
    delete It.second.TargetOptions;
  }
  LLVMTargetInfo.clear();
  llvm::llvm_shutdown();
  Singleton<TargetManager>::~Singleton();
}

llvm::Expected<const TargetInfo &>
TargetManager::getTargetInfo(const hsa::ISA &Isa) const {
  if (!LLVMTargetInfo.contains(Isa)) {
    auto Info = LLVMTargetInfo.insert({Isa, TargetInfo()}).first;

    auto TT = Isa.getTargetTriple();
    LUTHIER_RETURN_ON_ERROR(TT.takeError());

    std::string Error;

    auto Target = llvm::TargetRegistry::lookupTarget(TT->normalize(), Error);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Target));

    auto MRI = Target->createMCRegInfo(TT->getTriple());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MRI));

    auto TargetOptions = new llvm::TargetOptions();

    TargetOptions->MCOptions.AsmVerbose = true;

    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(TargetOptions));

    auto MAI = Target->createMCAsmInfo(*MRI, TT->getTriple(),
                                       TargetOptions->MCOptions);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MAI));

    auto MII = Target->createMCInstrInfo();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MII));

    auto MIA = Target->createMCInstrAnalysis(MII);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MIA));

    auto CPU = Isa.getProcessor();
    LUTHIER_RETURN_ON_ERROR(CPU.takeError());

    auto FeatureString = Isa.getSubTargetFeatures();
    LUTHIER_RETURN_ON_ERROR(FeatureString.takeError());

    auto STI = Target->createMCSubtargetInfo(TT->getTriple(), *CPU,
                                             FeatureString->getString());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(STI));

    auto IP = Target->createMCInstPrinter(
        llvm::Triple(*TT), MAI->getAssemblerDialect(), *MAI, *MII, *MRI);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(IP));

    Info->second.Target = Target;
    Info->second.MRI = MRI;
    Info->second.MAI = MAI;
    Info->second.MII = MII;
    Info->second.MIA = MIA;
    Info->second.STI = STI;
    Info->second.IP = IP;
    Info->second.TargetOptions = TargetOptions;
  }
  return LLVMTargetInfo[Isa];
}
llvm::Expected<std::unique_ptr<llvm::GCNTargetMachine>>
TargetManager::createTargetMachine(
    const hsa::ISA &ISA, const llvm::TargetOptions &TargetOptions) const {
  auto TT = ISA.getTargetTriple();
  LUTHIER_RETURN_ON_ERROR(TT.takeError());
  std::string Error;
  auto Target = llvm::TargetRegistry::lookupTarget(TT->normalize(), Error);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Target));
  auto CPU = ISA.getProcessor();
  LUTHIER_RETURN_ON_ERROR(CPU.takeError());

  auto FeatureString = ISA.getSubTargetFeatures();
  LUTHIER_RETURN_ON_ERROR(FeatureString.takeError());
  return std::unique_ptr<llvm::GCNTargetMachine>(
      reinterpret_cast<llvm::GCNTargetMachine *>(Target->createTargetMachine(
          llvm::Triple(*TT).normalize(), *CPU, FeatureString->getString(),
          TargetOptions, llvm::Reloc::PIC_)));
}

} // namespace luthier
