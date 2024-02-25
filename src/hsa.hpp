#ifndef HSA_HPP
#define HSA_HPP
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &agents);

llvm::Error getAllExecutables(llvm::SmallVectorImpl<Executable> &executables);

} // namespace luthier::hsa

#endif