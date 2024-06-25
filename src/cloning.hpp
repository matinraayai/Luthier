//===-- cloning.hpp - IR and MIR Cloning Utilities ------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility functions used to clone LLVM MIR constructs,
/// used frequently by Luthier components involved in the code generation
/// process. It is essentially a modified version of llvm-reduce.
//===----------------------------------------------------------------------===//

#ifndef CLONING_HPP
#define CLONING_HPP
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
namespace luthier {

/// Clones the content of the source machine function \p SrcMF into the
/// The module of \p DestMMI should be a cloned version of \p SrcMF 's module,
/// and the value map of the cloning operation must be passed in \p VMap
/// \param SrcMF
/// \param VMap
/// \param DestMMI
/// \return
llvm::Expected<std::unique_ptr<llvm::MachineFunction>>
cloneMF(const llvm::MachineFunction *SrcMF, const llvm::ValueToValueMapTy &VMap,
        llvm::MachineModuleInfo &DestMMI);

/// Clones the content of the source Machine Module Info \p SrcMMI into
/// the \p DestMMI. Uses the \p VMap to map the global objects in the source
/// module into the destination, already cloned module
/// \param [in] SrcMMI the source \c llvm::MachineModuleInfo
/// \param [in] VMap a mapping between a global object in the source module and
/// its copy in the destination module
/// \param [out] DestMMI the destination \c llvm::MachineModuleInfo
/// \return an \c llvm::Error if an issue was encountered during the process
llvm::Error cloneMMI(const llvm::MachineModuleInfo &SrcMMI,
                     const llvm::ValueToValueMapTy &VMap,
                     llvm::MachineModuleInfo &DestMMI);

} // namespace luthier

#endif