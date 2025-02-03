//===-- Controller.hpp - Luthier tool's Controller Logic ------------------===//
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
#ifndef LUTHIER_TOOLING_CONTROLLER_HPP
#define LUTHIER_TOOLING_CONTROLLER_HPP

#include "common/Singleton.hpp"
#include "luthier/types.h"

#include <functional>

namespace luthier {
class CodeGenerator;

class ToolExecutableLoader;

class CodeLifter;

class TargetManager;

namespace hip {
class HipCompilerApiInterceptor;

class HipRuntimeApiInterceptor;
} // namespace hip

namespace hsa {
class HsaRuntimeInterceptor;

class ExecutableBackedObjectsCache;
} // namespace hsa

/// \brief a \c Singleton in charge of managing all other singletons in Luthier,
/// as well as registration of Luthier with rocprofiler-sdk
class Controller : public Singleton<Controller> {
private:
  /// \c CodeGenerator \c Singleton instance
  CodeGenerator *CG{nullptr};

  /// \c ToolExecutableLoader \c Singleton instance
  ToolExecutableLoader *TEL{nullptr};

  /// \c CodeLifter \c Singleton instance
  CodeLifter *CL{nullptr};

  /// \c TargetManager \c Singleton instance
  TargetManager *TM{nullptr};

  /// \c hip::HipCompilerApiInterceptor \c Singleton instance
  hip::HipCompilerApiInterceptor *HipCompilerInterceptor{nullptr};

  /// \c hip::HipRuntimeApiInterceptor \c Singleton instance
  hip::HipRuntimeApiInterceptor *HipRuntimeInterceptor{nullptr};

  /// \c hsa::HsaRuntimeInterceptor \c Singleton instance
  hsa::HsaRuntimeInterceptor *HsaInterceptor{nullptr};

  /// \c hsa::ExecutableBackedObjectsCache \c Singleton instance
  hsa::ExecutableBackedObjectsCache *HsaPlatform{nullptr};

  /// A callback invoked before/after when rocprofiler has provided the
  /// HSA API table to the Luthier tool
  std::function<void(ApiEvtPhase)> AtHSAApiTableCaptureEvtCallback{
      [](ApiEvtPhase) {}};

  /// A callback invoked before/after when rocprofiler has provided the
  /// HIP Dispatch API table to the Luthier tool
  std::function<void(ApiEvtPhase)> AtHIPDispatchApiTableCaptureEvtCallback{
      [](ApiEvtPhase) {}};

  /// A callback invoked before/after when rocprofiler has provided the
  /// HIP Compiler API table to the Luthier tool
  std::function<void(ApiEvtPhase)> AtHIPCompilerApiTableCaptureEvtCallback{
      [](ApiEvtPhase) {}};

  /// A callback invoked inside the \c luthier::rocprofilerServiceInit
  /// function to allow for requesting additional rocprofiler-sdk services
  std::function<void()> RocprofilerServiceInitCallback{[]() {}};

public:
  Controller();

  ~Controller() override;

  void setAtHSAApiTableCaptureEvtCallback(
      const std::function<void(ApiEvtPhase)> &CB) {
    AtHSAApiTableCaptureEvtCallback = CB;
  }

  const std::function<void(ApiEvtPhase)> &getAtHSAApiTableCaptureEvtCallback() {
    return AtHSAApiTableCaptureEvtCallback;
  }

  void setAtHIPRuntimeApiTableCaptureEvtCallback(
      const std::function<void(ApiEvtPhase)> &CB) {
    AtHIPCompilerApiTableCaptureEvtCallback = CB;
  }

  const std::function<void(ApiEvtPhase)> &
  getAtHIPRuntimeApiTableCaptureEvtCallback() {
    return AtHIPCompilerApiTableCaptureEvtCallback;
  }

  void setAtHIPCompilerApiTableCaptureEvtCallback(
      const std::function<void(ApiEvtPhase)> &CB) {
    AtHIPCompilerApiTableCaptureEvtCallback = CB;
  }

  const std::function<void(ApiEvtPhase)> &
  getAtHIPCompilerApiTableCaptureEvtCallback() {
    return AtHIPCompilerApiTableCaptureEvtCallback;
  }

  void setRocprofilerServiceInitCallback(const std::function<void()> &CB) {
    RocprofilerServiceInitCallback = CB;
  }

  const std::function<void()> &getRocprofilerServiceInitCallback() {
    return RocprofilerServiceInitCallback;
  }
};
} // namespace luthier

#endif