//===-- Controller.h - Luthier tool's Controller Logic ----------*- C++ -*-===//
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
/// This file defines a \c Controller singleton class in charge of keeping
/// track of all other singletons, as well as different callbacks invoked during
/// execution of an instrumented application.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_CONTEXT_H
#define LUTHIER_TOOLING_CONTEXT_H
#include "luthier/Common/Singleton.h"
#include "luthier/HSA/Metadata.h"
#include "luthier/HSA/PacketMointor.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"

namespace luthier {
class CodeGenerator;

class ToolExecutableLoader;

class CodeLifter;

class TargetManager;

namespace hsa {

class LoadedCodeObjectCache;
} // namespace hsa

/// \brief a \c Singleton in charge of managing all other singletons in Luthier,
/// as well as registration of Luthier with rocprofiler-sdk
class Context : public Singleton<Context> {
private:
  /// \c CodeGenerator \c Singleton instance
  CodeGenerator *CG{nullptr};

  /// \c ToolExecutableLoader \c Singleton instance
  ToolExecutableLoader *TEL{nullptr};

  /// \c CodeLifter \c Singleton instance
  CodeLifter *CL{nullptr};

  /// \c TargetManager \c Singleton instance
  TargetManager *TM{nullptr};

  rocprofiler::HipApiTableSnapshot<ROCPROFILER_HIP_COMPILER_TABLE>
      *HipCompilerTableSnapshot{nullptr};

  rocprofiler::HsaApiTableSnapshot<::CoreApiTable> *HsaCoreApiTableSnapshot{
      nullptr};

  rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *HsaAmdExtTableSnapshot{
      nullptr};

  rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
      *VenLoaderSnapshot{nullptr};

  /// \c hsa::ExecutableBackedObjectsCache \c Singleton instance
  hsa::LoadedCodeObjectCache *CodeObjectCache{nullptr};

  amdgpu::hsamd::MetadataParser *MDParser{nullptr};

  hsa::PacketMonitor *PacketMonitor{nullptr};

public:
  explicit Context(hsa::PacketMonitor::CallbackType PacketCallback,
                   llvm::Error &Err);

  ~Context() override;

  [[nodiscard]] hip::ApiTableContainer<ROCPROFILER_HIP_COMPILER_TABLE>
  getHipCompilerTable() const {
    return HipCompilerTableSnapshot->getTable();
  }

  [[nodiscard]] hsa::ApiTableContainer<::CoreApiTable> getHsaCoreTable() const {
    return HsaCoreApiTableSnapshot->getTable();
  }

  [[nodiscard]] hsa::ApiTableContainer<::AmdExtTable>
  getHsaAmdExtTable() const {
    return HsaAmdExtTableSnapshot->getTable();
  }

  [[nodiscard]] const hsa::ExtensionApiTableInfo<
      HSA_EXTENSION_AMD_LOADER>::TableType &
  getHsaLoaderTable() const {
    return VenLoaderSnapshot->getTable();
  }
};
} // namespace luthier

#endif