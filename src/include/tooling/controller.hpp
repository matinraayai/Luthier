//===-- controller.hpp - Luthier tool's Controller Logic ------------------===//
//
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

#include "luthier/types.h"
#include "common/singleton.hpp"

#include <functional>

namespace luthier {
class CodeGenerator;

class ToolExecutableManager;

class CodeLifter;

class TargetManager;

namespace hip {
class HipCompilerApiInterceptor;

class HipRuntimeApiInterceptor;
} // namespace hip

namespace hsa {
class HsaRuntimeInterceptor;

class Platform;
} // namespace hsa

class Controller : public Singleton<Controller> {
private:
  /// Controller manages its own lifetime
  static Controller *C;

  // All other singletons

  CodeGenerator *CG{nullptr};

  ToolExecutableManager *COM{nullptr};

  CodeLifter *CL{nullptr};

  TargetManager *TM{nullptr};

  hip::HipCompilerApiInterceptor *HipCompilerInterceptor{nullptr};

  hip::HipRuntimeApiInterceptor *HipRuntimeInterceptor{nullptr};

  hsa::HsaRuntimeInterceptor *HsaInterceptor{nullptr};

  hsa::Platform *HsaPlatform{nullptr};

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