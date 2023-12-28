#ifndef CONTEXT_MANAGER_HPP
#define CONTEXT_MANAGER_HPP
#include "hsa_isa.hpp"
#include <unordered_map>
#include <vector>
#include <optional>

namespace llvm {

class Target;

class MCRegisterInfo;

class MCTargetOptions;

class MCAsmInfo;

class MCInstrInfo;

class MCInstrAnalysis;

class MCSubtargetInfo;

class MCInstPrinter;
}// namespace llvm

namespace luthier {

namespace hsa {

class GpuAgent;

class Isa;

class Executable;
}// namespace hsa

struct LLVMMCTargetInfo {
    friend class ContextManager;

 public:
    const llvm::Target *target_;
    std::unique_ptr<const llvm::MCRegisterInfo> MRI_;
    std::unique_ptr<const llvm::MCTargetOptions> MCOptions_;
    std::unique_ptr<const llvm::MCAsmInfo> MAI_;
    std::unique_ptr<const llvm::MCInstrInfo> MII_;
    std::unique_ptr<const llvm::MCInstrAnalysis> MIA_;
    std::unique_ptr<const llvm::MCSubtargetInfo> STI_;
    std::unique_ptr<llvm::MCInstPrinter> IP_;
    LLVMMCTargetInfo() = delete;

 private:
    LLVMMCTargetInfo(const llvm::Target *target, std::unique_ptr<const llvm::MCRegisterInfo> mri,
                     std::unique_ptr<const llvm::MCTargetOptions> mcOptions, std::unique_ptr<const llvm::MCAsmInfo> mai,
                     std::unique_ptr<const llvm::MCInstrInfo> mii, std::unique_ptr<const llvm::MCInstrAnalysis> mia,
                     std::unique_ptr<const llvm::MCSubtargetInfo> sti,
                     std::unique_ptr<llvm::MCInstPrinter> ip);
};

class ContextManager {
 private:
    std::vector<luthier::hsa::GpuAgent> agents_;
    std::unordered_map<hsa::Isa, LLVMMCTargetInfo> llvmContexts_;

    hsa_status_t initGpuAgentList();

    ContextManager();
    ~ContextManager();

 public:
    ContextManager(const ContextManager &) = delete;
    ContextManager &operator=(const ContextManager &) = delete;

    static inline ContextManager &instance() {
        static ContextManager instance;
        return instance;
    }

    const std::vector<hsa::GpuAgent> &getHsaAgents() const { return agents_; };

    std::vector<luthier::hsa::Executable> getHsaExecutables() const;

    LLVMMCTargetInfo& getLLVMTargetInfo(const hsa::Isa& isa) {return llvmContexts_.at(isa);}
};

}// namespace luthier

#endif//CONTEXT_MANAGER_HPP
