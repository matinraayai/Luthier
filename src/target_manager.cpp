#include "target_manager.hpp"

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
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include "error.hpp"
#include "hsa_agent.hpp"

namespace luthier {

TargetManager::TargetManager() {
  // TODO: When the target application is multi-threaded, this might need to be
  // in a scoped lock
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUTargetMCA();
}

TargetManager::~TargetManager() {
  for (auto &it : llvmTargetInfo_) {
    delete it.second.MRI_;
    delete it.second.MAI_;
    delete it.second.MII_;
    delete it.second.MIA_;
    delete it.second.STI_;
    delete it.second.IP_;
    delete it.second.targetOptions_;
    delete it.second.targetMachine_;
    delete it.second.llvmContext_;
  }
  llvmTargetInfo_.clear();
}

llvm::Expected<const TargetInfo &>
TargetManager::getTargetInfo(const hsa::ISA &Isa) const {
  if (!llvmTargetInfo_.contains(Isa)) {
    auto Info = llvmTargetInfo_.insert({Isa, TargetInfo()}).first;

    auto TT = Isa.getLLVMTargetTriple();
    LUTHIER_RETURN_ON_ERROR(TT.takeError());

    std::string Error;

    auto Target = llvm::TargetRegistry::lookupTarget(
        llvm::Triple(*TT).normalize(), Error);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Target));

    auto MRI = Target->createMCRegInfo(*TT);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MRI));

    auto TargetOptions = new llvm::TargetOptions();

    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(TargetOptions));

    auto MAI = Target->createMCAsmInfo(*MRI, *TT, TargetOptions->MCOptions);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MAI));

    auto MII = Target->createMCInstrInfo();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MII));

    auto MIA = Target->createMCInstrAnalysis(MII);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MIA));

    auto CPU = Isa.getProcessor();
    LUTHIER_RETURN_ON_ERROR(CPU.takeError());

    auto FeatureString = Isa.getFeatureString();
    LUTHIER_RETURN_ON_ERROR(FeatureString.takeError());

    auto STI = Target->createMCSubtargetInfo(*TT, *CPU, *FeatureString);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(STI));

    auto IP = Target->createMCInstPrinter(
        llvm::Triple(*TT), MAI->getAssemblerDialect(), *MAI, *MII, *MRI);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(IP));

    auto TM =
        reinterpret_cast<llvm::GCNTargetMachine *>(Target->createTargetMachine(
            llvm::Triple(*TT).normalize(), *CPU, *FeatureString, *TargetOptions,
            llvm::Reloc::PIC_));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(TM));

    auto LLVMContext = new llvm::LLVMContext();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(LLVMContext));

    Info->second.target_ = Target;
    Info->second.MRI_ = MRI;
    Info->second.MAI_ = MAI;
    Info->second.MII_ = MII;
    Info->second.MIA_ = MIA;
    Info->second.STI_ = STI;
    Info->second.IP_ = IP;
    Info->second.targetOptions_ = TargetOptions;
    Info->second.targetMachine_ = TM;
    Info->second.llvmContext_ = LLVMContext;
  }
  return llvmTargetInfo_[Isa];
}

} // namespace luthier
