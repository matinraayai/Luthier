#ifndef TARGET_MANAGER_HPP
#define TARGET_MANAGER_HPP
#include <llvm/Target/TargetOptions.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "hsa_isa.hpp"

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

class TargetManager;

struct TargetInfo {
    friend class TargetManager;

 private:
    const llvm::Target *target_{nullptr};
    std::unique_ptr<const llvm::MCRegisterInfo> MRI_{nullptr};
    std::unique_ptr<const llvm::MCAsmInfo> MAI_{nullptr};
    std::unique_ptr<const llvm::MCInstrInfo> MII_{nullptr};
    std::unique_ptr<const llvm::MCInstrAnalysis> MIA_{nullptr};
    std::unique_ptr<const llvm::MCSubtargetInfo> STI_{nullptr};
    std::unique_ptr<llvm::MCInstPrinter> IP_{nullptr};
    const llvm::TargetOptions targetOptions_{};

 public:
    [[nodiscard]] const llvm::Target *getTarget() const { return target_; }

    [[nodiscard]] const llvm::MCRegisterInfo *getMCRegisterInfo() const { return MRI_.get(); }

    [[nodiscard]] const llvm::MCAsmInfo *getMCAsmInfo() const { return MAI_.get(); }

    [[nodiscard]] const llvm::MCInstrInfo *getMCInstrInfo() const { return MII_.get(); }

    [[nodiscard]] const llvm::MCInstrAnalysis *getMCInstrAnalysis() const { return MIA_.get(); }

    [[nodiscard]] const llvm::MCSubtargetInfo *getMCSubTargetInfo() const { return STI_.get(); }

    [[nodiscard]] llvm::MCInstPrinter *getMCInstPrinter() const { return IP_.get(); }

    [[nodiscard]] const llvm::TargetOptions& getTargetOptions() const { return targetOptions_; }
};

class TargetManager {
 private:
    std::unordered_map<hsa::Isa, std::unique_ptr<TargetInfo>> llvmTargetInfo_;

    TargetManager();
    ~TargetManager();

 public:
    TargetManager(const TargetManager &) = delete;
    TargetManager &operator=(const TargetManager &) = delete;

    static inline TargetManager &instance() {
        static TargetManager instance;
        return instance;
    }

    const TargetInfo &getTargetInfo(const hsa::Isa &isa) const { return *llvmTargetInfo_.at(isa); }
};

}// namespace luthier

#endif//CONTEXT_MANAGER_HPP
