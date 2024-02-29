#ifndef HSA_HPP
#define HSA_HPP
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &Agents);

llvm::Error getAllExecutables(llvm::SmallVectorImpl<Executable> &Executables);

template<typename T>
llvm::Expected<T*> queryHostAddress(T* DeviceAddress);

llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(llvm::ArrayRef<uint8_t> Code);

llvm::Expected<llvm::StringRef>
convertToHostEquivalent(llvm::StringRef Code);

} // namespace luthier::hsa

#endif