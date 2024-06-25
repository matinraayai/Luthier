//===-- cloning.hpp - IR and MIR Cloning Utilities ------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility functions used to clone LLVM IR and MIR
/// Modules and Functions, used frequently by Luthier components involved in the
/// code generation process.
//===----------------------------------------------------------------------===//

#ifndef CLONING_HPP
#define CLONING_HPP
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
namespace luthier {

/// Clones the content of the source Machine Module Info \p SrcMMI into
/// the \p DestMMI. Uses the \p VMap to map the global objects in the source
/// module into the destination, already cloned module\n
/// Underlying logic has been copy-pasted from llvm-reduce\n
/// \param [in] SrcMMI the source \c llvm::MachineModuleInfo
/// \param [in] VMap a mapping between a global object in the source module and
/// its copy in the destination module
/// \param [out] DestMMI the destination \c llvm::MachineModuleInfo
/// \return an \c llvm::Error if an issue was encountered during the process
llvm::Error cloneModuleAndMMI(const llvm::MachineModuleInfo &SrcMMI,
                              const llvm::ValueToValueMapTy &VMap,
                              llvm::MachineModuleInfo &DestMMI);

} // namespace luthier

#endif