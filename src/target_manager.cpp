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

#include <atomic>
#include <memory>
#include <mutex>

#include "error.hpp"
#include "hsa.hpp"
#include "hsa_agent.hpp"

namespace luthier {

TargetManager::TargetManager() {
    static std::mutex llvm_init_mutex;
    {
        std::scoped_lock llvm_init_lock(llvm_init_mutex);
        static bool LLVMInitialized = false;
        if (LLVMInitialized) { return; }
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUDisassembler();
        LLVMInitializeAMDGPUAsmParser();
        LLVMInitializeAMDGPUAsmPrinter();
        LLVMInitializeAMDGPUTargetMCA();
        LLVMInitialized = true;
    }
}

TargetManager::~TargetManager() {
    for (auto &it: llvmTargetInfo_) {
        delete it.second.MRI_;
        delete it.second.MAI_;
        delete it.second.MII_;
        delete it.second.MIA_;
        delete it.second.STI_;
        delete it.second.IP_;
        //        delete it.second.targetOptions_;
    }
    llvmTargetInfo_.clear();
}

const TargetInfo &TargetManager::getTargetInfo(const hsa::Isa &isa) const {
    if (!llvmTargetInfo_.contains(isa)) {
        auto info = llvmTargetInfo_.insert({isa, TargetInfo()}).first;
        std::string targetTriple = llvm::Triple(isa.getLLVMTargetTriple()).normalize();
        //        std::string targetIsa = isa.getLLVMTarget();
        std::string error;

        //        auto theTargetMachine = reinterpret_cast<llvm::LLVMTargetMachine *>(
        //            target->createTargetMachine(triple, cpu, features,
        //                                        targetInfo.getTargetOptions(), llvm::Reloc::PIC_));

        auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
        LUTHIER_CHECK(target && error.c_str());

        auto mri = target->createMCRegInfo(targetTriple);
        LUTHIER_CHECK(mri);

        //        auto targetOptions = new llvm::TargetOptions();

        auto mai = target->createMCAsmInfo(*mri, targetTriple, info->second.targetOptions_.MCOptions);
        LUTHIER_CHECK(mai);

        auto mii = target->createMCInstrInfo();
        LUTHIER_CHECK(mii);

        auto mia = target->createMCInstrAnalysis(mii);
        LUTHIER_CHECK(mia);

        auto sti = target->createMCSubtargetInfo(targetTriple, isa.getProcessor(), isa.getFeatureString());
        LUTHIER_CHECK(sti);

        auto ip = target->createMCInstPrinter(llvm::Triple(targetTriple), mai->getAssemblerDialect(), *mai, *mii, *mri);
        LUTHIER_CHECK(ip);

        info->second.target_ = target;
        info->second.MRI_ = mri;
        info->second.MAI_ = mai;
        info->second.MII_ = mii;
        info->second.MIA_ = mia;
        info->second.STI_ = sti;
        info->second.IP_ = ip;
        //        info->second.targetOptions_ = targetOptions;
    }
    return llvmTargetInfo_.at(isa);
}

}// namespace luthier
