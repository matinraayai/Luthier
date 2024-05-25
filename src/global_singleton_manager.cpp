#include "global_singleton_manager.hpp"
#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "target_manager.hpp"
#include "hip_intercept.hpp"
#include "disassembler.hpp"

namespace luthier {

template <>
GlobalSingletonManager *Singleton<GlobalSingletonManager>::Instance{nullptr};

GlobalSingletonManager::GlobalSingletonManager()
    : Singleton<GlobalSingletonManager>() {
  CG = new CodeGenerator();
  COM = new CodeObjectManager();
  CL = new CodeLifter();
  TM = new TargetManager();
  HipInterceptor = new hip::Interceptor();
}

GlobalSingletonManager::~GlobalSingletonManager() {
  delete CG;
  delete COM;
  delete TM;
  delete HipInterceptor;
}
} // namespace luthier