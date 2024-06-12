#ifndef CONTROLLER_HPP
#define CONTROLLER_HPP

#include "singleton.hpp"

namespace luthier {
class CodeGenerator;

class CodeObjectManager;

class CodeLifter;

class TargetManager;

namespace hip {
class Interceptor;

class Platform;
} // namespace hip

namespace hsa {
class Interceptor;

class Platform;
} // namespace hsa

class Controller : public Singleton<Controller> {
private:
  // Controller manages its own lifetime
  static Controller *C;

  CodeGenerator *CG{nullptr};

  CodeObjectManager *COM{nullptr};

  CodeLifter *CL{nullptr};

  TargetManager *TM{nullptr};

  hip::Interceptor *HipInterceptor{nullptr};

  hip::Platform *HipPlatform{nullptr};

  hsa::Interceptor *HsaInterceptor{nullptr};

  hsa::Platform *HsaPlatform{nullptr};

private:
  __attribute__((constructor)) static void init();

  __attribute__((destructor)) static void finalize();

  Controller();

  ~Controller();
};
} // namespace luthier

#endif