#include "hsa/hsa_intercept.hpp"
#include <luthier/LoadedCodeObjectKernel.h>

namespace luthier::hsa {

llvm::Expected<const KernelDescriptor *>
LoadedCodeObjectKernel::getKernelDescriptor() const {
  auto ExecutableSymbol = getExecutableSymbol();
  LUTHIER_RETURN_ON_ERROR(ExecutableSymbol.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(ExecutableSymbol->has_value()));

  luthier::address_t KernelObject;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      hsa::HsaRuntimeInterceptor::instance()
          .getSavedApiTableContainer()
          .core.hsa_executable_symbol_get_info_fn(
              **ExecutableSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
              &KernelObject)));
  return reinterpret_cast<const KernelDescriptor *>(KernelObject);
}
} // namespace luthier::hsa