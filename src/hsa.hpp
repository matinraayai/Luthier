#ifndef HSA_HPP
#define HSA_HPP
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

void getGpuAgents(llvm::SmallVectorImpl<GpuAgent>& agents);

void getAllExecutables(llvm::SmallVectorImpl<Executable>& executables);

}




#endif