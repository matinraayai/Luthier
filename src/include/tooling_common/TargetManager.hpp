//===-- TargetManager.hpp - Luthier's LLVM Target Management  -------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes Luthier's Target Manager, a singleton in charge of
/// initializing and finalizing the LLVM library, as well as creating
/// target description objects for each ISA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_TARGET_MANAGER_HPP
#define LUTHIER_TOOLING_COMMON_TARGET_MANAGER_HPP
#include "luthier/common/Singleton.h"
#include "luthier/hsa/ISA.h"
#include "luthier/rocprofiler-sdk/ApiTableSnapshot.h"
#include <llvm/Target/TargetOptions.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

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
  mutable std::unordered_map<hsa_isa_t, TargetInfo> LLVMTargetInfo{};

  rocprofiler::HsaApiTableSnapshot<::CoreApiTable> CoreApiTableSnapshot;

public:
  /// Initializes the AMDGPU LLVM target
  explicit TargetManager(llvm::Error &Err);

  /// Default destructor; Destroys all Target descriptors and shuts down LLVM
  ~TargetManager() override;

  llvm::Expected<const TargetInfo &> getTargetInfo(hsa_isa_t Isa) const;

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
  createTargetMachine(hsa_isa_t ISA,
                      const llvm::TargetOptions &TargetOptions = {}) const;
};

} // namespace luthier

#endif // TARGET_MANAGER_HPP
