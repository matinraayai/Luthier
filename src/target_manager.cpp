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
#include <llvm/MC/MCTargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>

#include <memory>

#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "target_manager.hpp"

namespace luthier {

TargetManager::TargetManager() {

    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUTargetMCA();

    llvm::SmallVector<hsa::GpuAgent, 8> agents;
    hsa::getGpuAgents(agents);
    for (const auto &agent: agents) {
        llvm::SmallVector<hsa::Isa, 1> isaList;
        agent.getIsa(isaList);

        for (const auto &isa: isaList) {
            auto info = llvmTargetInfo_.insert({isa, std::make_unique<TargetInfo>()}).first;
            std::string targetTriple = isa.getLLVMTargetTriple();
            std::string targetIsa = isa.getLLVMTarget();
            std::string error;

            auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
            assert(target);

            auto MRI = target->createMCRegInfo(targetTriple);
            assert(MRI);

            auto MAI = target->createMCAsmInfo(*MRI, targetTriple, info->second->targetOptions_.MCOptions);
            assert(MAI);

            auto MII = target->createMCInstrInfo();
            assert(MII);

            auto MIA = target->createMCInstrAnalysis(MII);
            assert(MIA);

            auto STI = target->createMCSubtargetInfo(targetTriple, isa.getProcessor(), isa.getFeatureString());
            assert(STI);

            auto IP =
                target->createMCInstPrinter(llvm::Triple(targetTriple), MAI->getAssemblerDialect(), *MAI, *MII, *MRI);
            assert(IP);

            info->second->target_ = target;
            info->second->MRI_.reset(MRI);
            info->second->MAI_.reset(MAI);
            info->second->MII_.reset(MII);
            info->second->MIA_.reset(MIA);
            info->second->STI_.reset(STI);
            info->second->IP_.reset(IP);
        }
    }
}

TargetManager::~TargetManager() { llvmTargetInfo_.clear(); }

}// namespace luthier
