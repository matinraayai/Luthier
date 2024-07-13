#ifndef TARGET_MANAGER_HPP
#define TARGET_MANAGER_HPP
#include <llvm/Target/TargetOptions.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "hsa/hsa_isa.hpp"
#include "common/singleton.hpp"

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
  const llvm::Target *Target{nullptr};
  llvm::GCNTargetMachine *TargetMachine{nullptr};
  const llvm::MCRegisterInfo *MRI{nullptr};
  const llvm::MCAsmInfo *MAI{nullptr};
  const llvm::MCInstrInfo *MII{nullptr};
  const llvm::MCInstrAnalysis *MIA{nullptr};
  const llvm::MCSubtargetInfo *STI{nullptr};
  llvm::MCInstPrinter *IP{nullptr};
  llvm::TargetOptions *TargetOptions{nullptr};

public:
  [[nodiscard]] const llvm::Target *getTarget() const { return Target; }

  [[nodiscard]] llvm::GCNTargetMachine *getTargetMachine() const {
    return TargetMachine;
  }

  [[nodiscard]] const llvm::MCRegisterInfo *getMCRegisterInfo() const {
    return MRI;
  }

  [[nodiscard]] const llvm::MCAsmInfo *getMCAsmInfo() const { return MAI; }

  [[nodiscard]] const llvm::MCInstrInfo *getMCInstrInfo() const { return MII; }

  [[nodiscard]] const llvm::MCInstrAnalysis *getMCInstrAnalysis() const {
    return MIA;
  }

  [[nodiscard]] const llvm::MCSubtargetInfo *getMCSubTargetInfo() const {
    return STI;
  }

  [[nodiscard]] llvm::MCInstPrinter *getMCInstPrinter() const { return IP; }

  [[nodiscard]] const llvm::TargetOptions &getTargetOptions() const {
    return *TargetOptions;
  }
};

/**
 * \brief in charge of creating and managing LLVM constructs that are shared
 * among different components of Luthier (e.g. Disassembler, CodeGenerator)
 * Initializes the AMDGPU LLVM target upon construction
 */
class TargetManager: public Singleton<TargetManager> {
private:
  mutable std::unordered_map<hsa::ISA, TargetInfo> LLVMTargetInfo{};

public:
  TargetManager(const TargetManager &) = delete;
  TargetManager &operator=(const TargetManager &) = delete;

  TargetManager();
  ~TargetManager();

  llvm::Expected<const TargetInfo &> getTargetInfo(const hsa::ISA &Isa) const;
};

} // namespace luthier

#endif // TARGET_MANAGER_HPP
