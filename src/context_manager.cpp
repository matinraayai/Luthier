#include "context_manager.hpp"

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

#include "error.h"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"

namespace luthier {

hsa_status_t ContextManager::initGpuAgentList() {
    auto &coreTable = HsaInterceptor::instance().getSavedHsaTables().core;

    auto returnGpuAgentsCallback = [](hsa_agent_t agent, void *data) {
        auto agentMap = reinterpret_cast<std::vector<hsa::GpuAgent> *>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS) return stat;
        if (dev_type == HSA_DEVICE_TYPE_GPU) { agentMap->push_back(luthier::hsa::GpuAgent(agent)); }
        return stat;
    };
    return coreTable.hsa_iterate_agents_fn(returnGpuAgentsCallback, &agents_);
}

ContextManager::ContextManager() {
    // Initialize the GpuAgent list
    LUTHIER_HSA_CHECK(initGpuAgentList());

    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();

    for (const auto &agent: getHsaAgents()) {
        for (const auto &isa: agent.getIsa()) {

            std::string targetTriple = isa.getLLVMTargetTriple();
            std::string targetIsa = isa.getLLVMTarget();
            std::string error;

            const llvm::Target *theTarget = llvm::TargetRegistry::lookupTarget(targetTriple, error);
            assert(theTarget);
            std::unique_ptr<const llvm::MCRegisterInfo> mri(theTarget->createMCRegInfo(targetTriple));
            assert(mri);

            auto mcOptions = std::make_unique<const llvm::MCTargetOptions>();
            std::unique_ptr<const llvm::MCAsmInfo> mai(theTarget->createMCAsmInfo(*mri, targetTriple, *mcOptions));
            assert(mai);

            std::unique_ptr<const llvm::MCInstrInfo> mii(theTarget->createMCInstrInfo());
            assert(mii);

            std::unique_ptr<const llvm::MCInstrAnalysis> mia(theTarget->createMCInstrAnalysis(mii.get()));

            std::unique_ptr<const llvm::MCSubtargetInfo> sti(
                theTarget->createMCSubtargetInfo(targetTriple, isa.getProcessor(), isa.getFeatureString()));
            assert(sti);

            std::unique_ptr<llvm::MCInstPrinter> ip(theTarget->createMCInstPrinter(
                llvm::Triple(targetTriple), mai->getAssemblerDialect(), *mai, *mii, *mri));
            assert(ip);

            llvmContexts_.insert({isa,
                                  LLVMMCTargetInfo(theTarget, std::move(mri), std::move(mcOptions), std::move(mai),
                                                   std::move(mii), std::move(mia), std::move(sti), std::move(ip))});
        }
    }
}

std::vector<hsa::Executable> ContextManager::getHsaExecutables() const {
    const auto &loaderApi = HsaInterceptor::instance().getHsaVenAmdLoaderTable();
    std::vector<luthier::hsa::Executable> out;
    auto iterator = [](hsa_executable_t exec, void *data) {
        auto out = reinterpret_cast<std::vector<luthier::hsa::Executable> *>(data);
        out->emplace_back(exec);
        return HSA_STATUS_SUCCESS;
    };
    LUTHIER_HSA_CHECK(loaderApi.hsa_ven_amd_loader_iterate_executables(iterator, &out));
    return out;
}
ContextManager::~ContextManager() = default;

LLVMMCTargetInfo::LLVMMCTargetInfo(const llvm::Target *target, std::unique_ptr<const llvm::MCRegisterInfo> mri,
                                   std::unique_ptr<const llvm::MCTargetOptions> mcOptions,
                                   std::unique_ptr<const llvm::MCAsmInfo> mai,
                                   std::unique_ptr<const llvm::MCInstrInfo> mii,
                                   std::unique_ptr<const llvm::MCInstrAnalysis> mia,
                                   std::unique_ptr<const llvm::MCSubtargetInfo> sti,
                                   std::unique_ptr<llvm::MCInstPrinter> ip)
    : target_(target),
      MRI_(std::move(mri)),
      MCOptions_(std::move(mcOptions)),
      MAI_(std::move(mai)),
      MII_(std::move(mii)),
      MIA_(std::move(mia)),
      STI_(std::move(sti)),
      IP_(std::move(ip)) {
    assert(target_);
    assert(MRI_);
    assert(MCOptions_);
    assert(MAI_);
    assert(MII_);
    assert(MIA_);
    assert(STI_);
};
}// namespace luthier
