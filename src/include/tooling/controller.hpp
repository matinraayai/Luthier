#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include "luthier/types.h"
#include "common/singleton.hpp"

#include <functional>

namespace luthier {
class CodeGenerator;

class ToolExecutableManager;

class CodeLifter;

class TargetManager;

namespace hip {
class CompilerInterceptor;

class RuntimeInterceptor;

class Platform;
} // namespace hip

namespace hsa {
class Interceptor;

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

  hip::CompilerInterceptor *HipCompilerInterceptor{nullptr};

  hip::RuntimeInterceptor *HipRuntimeInterceptor{nullptr};

  hip::Platform *HipPlatform{nullptr};

  hsa::Interceptor *HsaInterceptor{nullptr};

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