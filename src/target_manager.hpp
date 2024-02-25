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

class GCNTargetMachine;

class LLVMContext;
} // namespace llvm

namespace luthier {

class TargetManager;

struct TargetInfo {
  friend class TargetManager;

private:
  const llvm::Target *target_{nullptr};
  llvm::LLVMContext *llvmContext_{nullptr};
  llvm::GCNTargetMachine *targetMachine_{nullptr};
  const llvm::MCRegisterInfo *MRI_{nullptr};
  const llvm::MCAsmInfo *MAI_{nullptr};
  const llvm::MCInstrInfo *MII_{nullptr};
  const llvm::MCInstrAnalysis *MIA_{nullptr};
  const llvm::MCSubtargetInfo *STI_{nullptr};
  llvm::MCInstPrinter *IP_{nullptr};
  llvm::TargetOptions *targetOptions_{nullptr};

public:
  [[nodiscard]] const llvm::Target *getTarget() const { return target_; }

  [[nodiscard]] llvm::LLVMContext *getLLVMContext() { return llvmContext_; }

  [[nodiscard]] llvm::GCNTargetMachine *getTargetMachine() const {
    return targetMachine_;
  }

  [[nodiscard]] const llvm::MCRegisterInfo *getMCRegisterInfo() const {
    return MRI_;
  }

  [[nodiscard]] const llvm::MCAsmInfo *getMCAsmInfo() const { return MAI_; }

  [[nodiscard]] const llvm::MCInstrInfo *getMCInstrInfo() const { return MII_; }

  [[nodiscard]] const llvm::MCInstrAnalysis *getMCInstrAnalysis() const {
    return MIA_;
  }

  [[nodiscard]] const llvm::MCSubtargetInfo *getMCSubTargetInfo() const {
    return STI_;
  }

  [[nodiscard]] llvm::MCInstPrinter *getMCInstPrinter() const { return IP_; }

  [[nodiscard]] const llvm::TargetOptions &getTargetOptions() const {
    return *targetOptions_;
  }
};

/**
 * \brief in charge of creating and managing LLVM constructs that are shared
 * among different components of Luthier (e.g. Disassembler, CodeGenerator)
 * Initializes the AMDGPU LLVM target upon construction
 */
class TargetManager {
private:
  mutable std::unordered_map<hsa::ISA, TargetInfo> llvmTargetInfo_{};

  TargetManager();
  ~TargetManager();

public:
  TargetManager(const TargetManager &) = delete;
  TargetManager &operator=(const TargetManager &) = delete;

  static inline TargetManager &instance() {
    static TargetManager instance;
    return instance;
  }

  llvm::Expected<const TargetInfo &> getTargetInfo(const hsa::ISA &Isa) const;
};

} // namespace luthier

#endif // TARGET_MANAGER_HPP
