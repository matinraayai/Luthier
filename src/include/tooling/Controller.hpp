//===-- Controller.hpp - Luthier tool's Controller Logic ------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file contains the main Luthier logic behind a Luthier tool. It
/// defines a \c Controller singleton class in charge of keeping track of
/// All other singletons of Luthier and different callbacks invoked during
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

class Controller : public Singleton<Controller> {
private:
  /// Controller manages its own lifetime
  static Controller *C;

  // All other singletons

  CodeGenerator *CG{nullptr};

  ToolExecutableLoader *COM{nullptr};

  CodeLifter *CL{nullptr};

  TargetManager *TM{nullptr};

  hip::HipCompilerApiInterceptor *HipCompilerInterceptor{nullptr};

  hip::HipRuntimeApiInterceptor *HipRuntimeInterceptor{nullptr};

  hsa::HsaRuntimeInterceptor *HsaInterceptor{nullptr};

  hsa::ExecutableBackedObjectsCache *HsaPlatform{nullptr};

  // Stored callbacks
  std::function<void(ApiEvtPhase)> AtHSAApiTableCaptureEvtCallback{
      [](ApiEvtPhase) {}};

  std::function<void(ApiEvtPhase)> AtApiTableReleaseEvtCallback{
      [](ApiEvtPhase) {}};

  __attribute__((constructor, used)) static void init();

  __attribute__((destructor, used)) static void finalize();

  Controller();

  ~Controller();

public:
  void setAtHSAApiTableCaptureEvtCallback(
      const std::function<void(ApiEvtPhase)> &CB) {
    AtHSAApiTableCaptureEvtCallback = CB;
  }

  const std::function<void(ApiEvtPhase)> &getAtHSAApiTableCaptureEvtCallback() {
    return AtHSAApiTableCaptureEvtCallback;
  }

  void
  setAtApiTableReleaseEvtCallback(const std::function<void(ApiEvtPhase)> &CB) {
    AtApiTableReleaseEvtCallback = CB;
  }

  const std::function<void(ApiEvtPhase)> &getAtApiTableReleaseEvtCallback() {
    return AtApiTableReleaseEvtCallback;
  }
};
} // namespace luthier

#endif