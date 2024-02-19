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
    const llvm::MCRegisterInfo* MRI_{nullptr};
    const llvm::MCAsmInfo* MAI_{nullptr};
    const llvm::MCInstrInfo* MII_{nullptr};
    const llvm::MCInstrAnalysis* MIA_{nullptr};
    const llvm::MCSubtargetInfo* STI_{nullptr};
    llvm::MCInstPrinter* IP_{nullptr};
    llvm::TargetOptions* targetOptions_{nullptr};

 public:
    [[nodiscard]] const llvm::Target *getTarget() const { return target_; }

    [[nodiscard]] const llvm::MCRegisterInfo *getMCRegisterInfo() const { return MRI_; }

    [[nodiscard]] const llvm::MCAsmInfo *getMCAsmInfo() const { return MAI_; }

    [[nodiscard]] const llvm::MCInstrInfo *getMCInstrInfo() const { return MII_; }

    [[nodiscard]] const llvm::MCInstrAnalysis *getMCInstrAnalysis() const { return MIA_; }

    [[nodiscard]] const llvm::MCSubtargetInfo *getMCSubTargetInfo() const { return STI_; }

    [[nodiscard]] llvm::MCInstPrinter *getMCInstPrinter() const { return IP_; }

    [[nodiscard]] const llvm::TargetOptions& getTargetOptions() const { return *targetOptions_; }
};

class TargetManager {
 private:
    mutable std::unordered_map<hsa::Isa, TargetInfo> llvmTargetInfo_{};

    TargetManager();
    ~TargetManager();

 public:
    TargetManager(const TargetManager &) = delete;
    TargetManager &operator=(const TargetManager &) = delete;

    static inline TargetManager &instance() {
        static TargetManager instance;
        return instance;
    }

    const TargetInfo &getTargetInfo(const hsa::Isa &isa) const;
};

}// namespace luthier

#endif//TARGET_MANAGER_HPP
