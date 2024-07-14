//===-- target_manager.hpp - Luthier's LLVM Target Management  ------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's Target Manager, a singleton in charge of
/// initializing and finalizing the LLVM library, as well as creating
/// target description objects for each ISA.
//===----------------------------------------------------------------------===//
#ifndef TARGET_MANAGER_HPP
#define TARGET_MANAGER_HPP
#include <llvm/Target/TargetOptions.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/singleton.hpp"
#include "hsa/hsa_isa.hpp"

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
  const llvm::MCRegisterInfo *MRI{nullptr};
  const llvm::MCAsmInfo *MAI{nullptr};
  const llvm::MCInstrInfo *MII{nullptr};
  const llvm::MCInstrAnalysis *MIA{nullptr};
  const llvm::MCSubtargetInfo *STI{nullptr};
  llvm::MCInstPrinter *IP{nullptr};
  llvm::TargetOptions *TargetOptions{nullptr};

public:
  [[nodiscard]] const llvm::Target *getTarget() const { return Target; }

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

/// \brief in charge of creating and managing LLVM constructs that are shared
/// among different components of Luthier (e.g. CodeLifter, CodeGenerator)
/// Initializes the AMDGPU LLVM target upon construction, and shuts down LLVM
/// on destruction
class TargetManager : public Singleton<TargetManager> {
private:
  mutable std::unordered_map<hsa::ISA, TargetInfo> LLVMTargetInfo{};

public:
  /// Default constructor; Initializes the AMDGPU LLVM target, and parses the
  /// command line arguments
  TargetManager();

  /// Default destructor; Destroys all Target descriptors and shuts down LLVM
  ~TargetManager();

  llvm::Expected<const TargetInfo &> getTargetInfo(const hsa::ISA &Isa) const;

  /// Creates an \c llvm::GCNTargetMachine given the \p ISA and the
  /// <tt>TargetOptions</tt>
  /// \c llvm::GCNTargetMachine provides a description of the GCN target to
  /// an \c llvm::Module and \c llvm::MachineModuleInfo
  /// \param ISA \c hsa::ISA of the target
  /// \param TargetOptions target compilation options used with the target
  /// machine
  /// \return a unique pointer managing the newly-created
  /// <tt>llvm::GCNTargetMachine</tt>, or an \c llvm::Error if the process
  /// fails
  llvm::Expected<std::unique_ptr<llvm::GCNTargetMachine>>
  createTargetMachine(const hsa::ISA &ISA,
                      const llvm::TargetOptions &TargetOptions = {}) const;
};

} // namespace luthier

#endif // TARGET_MANAGER_HPP
