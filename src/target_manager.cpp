#include "target_manager.hpp"

#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrDesc.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCAsmLexer.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <AMDGPUTargetMachine.h>

#include "error.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"

namespace luthier {

TargetManager::TargetManager() {
    //TODO: When the target application is multi-threaded, this might need to be in a scoped lock
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTargetMCA();
}

TargetManager::~TargetManager() {
    for (auto &it: llvmTargetInfo_) {
        delete it.second.MRI_;
        delete it.second.MAI_;
        delete it.second.MII_;
        delete it.second.MIA_;
        delete it.second.STI_;
        delete it.second.IP_;
        delete it.second.targetOptions_;
        delete it.second.targetMachine_;
    }
    llvmTargetInfo_.clear();
}

const TargetInfo &TargetManager::getTargetInfo(const hsa::Isa &isa) const {
    if (!llvmTargetInfo_.contains(isa)) {
        auto info = llvmTargetInfo_.insert({isa, TargetInfo()}).first;

        std::string targetTriple = llvm::Triple(isa.getLLVMTargetTriple()).normalize();
        std::string error;

        auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
        LUTHIER_CHECK(target && error.c_str());

        auto mri = target->createMCRegInfo(targetTriple);
        LUTHIER_CHECK(mri);

        auto targetOptions = new llvm::TargetOptions();

        auto mai = target->createMCAsmInfo(*mri, targetTriple, targetOptions->MCOptions);
        LUTHIER_CHECK(mai);

        auto mii = target->createMCInstrInfo();
        LUTHIER_CHECK(mii);

        auto mia = target->createMCInstrAnalysis(mii);
        LUTHIER_CHECK(mia);

        std::string cpu = isa.getProcessor();
        std::string featureString = isa.getFeatureString();

        auto sti = target->createMCSubtargetInfo(targetTriple, cpu, featureString);
        LUTHIER_CHECK(sti);

        auto ip = target->createMCInstPrinter(llvm::Triple(targetTriple), mai->getAssemblerDialect(), *mai, *mii, *mri);
        LUTHIER_CHECK(ip);

        auto targetMachine = reinterpret_cast<llvm::GCNTargetMachine *>(target->createTargetMachine(
            llvm::Triple(targetTriple).normalize(), cpu, featureString, *targetOptions, llvm::Reloc::PIC_));

        info->second.target_ = target;
        info->second.MRI_ = mri;
        info->second.MAI_ = mai;
        info->second.MII_ = mii;
        info->second.MIA_ = mia;
        info->second.STI_ = sti;
        info->second.IP_ = ip;
        info->second.targetOptions_ = targetOptions;
        info->second.targetMachine_ = targetMachine;
    }
    return llvmTargetInfo_.at(isa);
}

}// namespace luthier
