#ifndef GLOBAL_MANAGER_HPP
#define GLOBAL_MANAGER_HPP
#include <llvm/Support/Error.h>

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

class GlobalSingletonManager : public Singleton<GlobalSingletonManager> {
private:
  CodeGenerator *CG{nullptr};

  CodeObjectManager *COM{nullptr};

  CodeLifter *CL{nullptr};

  TargetManager *TM{nullptr};

  hip::Interceptor *HipInterceptor{nullptr};

  hip::Platform *HipPlatform{nullptr};

  hsa::Interceptor *HsaInterceptor{nullptr};

  hsa::Platform *HsaPlatform{nullptr};

public:
  GlobalSingletonManager();

  ~GlobalSingletonManager();

  llvm::Error onExecutableFreeze();

  llvm::Error onCodeObjectLoad();

  llvm::Error onExecutableDestroy();
};
} // namespace luthier

#endif