#ifndef TARGET_MANAGER_HPP
#define TARGET_MANAGER_HPP
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "hsa_isa.hpp"
#include "llvm/Target/TargetOptions.h"

namespace llvm {

class Target;

class MCRegisterInfo;

class TargetOptions;

class MCAsmInfo;

class MCInstrInfo;

class MCInstrAnalysis;

class MCSubtargetInfo;

class MCInstPrinter;
}// namespace llvm

namespace luthier {

class ContextManager;

struct LLVMMCTargetInfo {
    friend class ContextManager;

 private:
    const llvm::Target *target_{nullptr};
    std::unique_ptr<const llvm::MCRegisterInfo> MRI_{nullptr};
    std::unique_ptr<const llvm::MCAsmInfo> MAI_{nullptr};
    std::unique_ptr<const llvm::MCInstrInfo> MII_{nullptr};
    std::unique_ptr<const llvm::MCInstrAnalysis> MIA_{nullptr};
    std::unique_ptr<const llvm::MCSubtargetInfo> STI_{nullptr};
    std::unique_ptr<llvm::MCInstPrinter> IP_{nullptr};
    llvm::TargetOptions* targetOptions_{nullptr}; //TODO: FIX the issue with the destructor


 public:

    [[nodiscard]] const llvm::Target *getTarget() const { return target_; }

    [[nodiscard]] const llvm::MCRegisterInfo *getMCRegisterInfo() const { return MRI_.get(); }

    [[nodiscard]] const llvm::MCAsmInfo *getMCAsmInfo() const { return MAI_.get(); }

    [[nodiscard]] const llvm::MCInstrInfo *getMCInstrInfo() const { return MII_.get(); }

    [[nodiscard]] const llvm::MCInstrAnalysis *getMCInstrAnalysis() const { return MIA_.get(); }

    [[nodiscard]] const llvm::MCSubtargetInfo *getMCSubTargetInfo() const { return STI_.get(); }

    [[nodiscard]] llvm::MCInstPrinter *getMCInstPrinter() const { return IP_.get(); }

    [[nodiscard]] llvm::TargetOptions *getTargetOptions() const { return targetOptions_; }
};

class ContextManager {
 private:
    std::unordered_map<hsa::Isa, std::unique_ptr<LLVMMCTargetInfo>> llvmContexts_;

    ContextManager();
    ~ContextManager();

 public:
    ContextManager(const ContextManager &) = delete;
    ContextManager &operator=(const ContextManager &) = delete;

    static inline ContextManager &instance() {
        static ContextManager instance;
        return instance;
    }

    const LLVMMCTargetInfo &getLLVMTargetInfo(const hsa::Isa &isa) const { return *llvmContexts_.at(isa); }
};

}// namespace luthier

#endif//CONTEXT_MANAGER_HPP
