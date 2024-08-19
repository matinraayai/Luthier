//===-- Cloning.hpp - IR and MIR Cloning Utilities ------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility functions used to clone LLVM MIR constructs,
/// used frequently by Luthier components involved in the code generation
/// process. It is essentially a modified version of llvm-reduce.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_COMMON_CLONING_HPP
#define LUTHIER_COMMON_CLONING_HPP
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
namespace luthier {

/// Clones the content of the source machine function \p SrcMF into the \p
/// DestMMI, and returns a unique pointer to the cloned Machine Function The
/// module of \p DestMMI should be a cloned version of \p SrcMF 's module, and
/// the value map of the cloning operation must be passed in \p VMap
/// \param [in] SrcMF the \c llvm::MachineFunction to be cloned
/// \param [in] VMap A Value map, mapping the global objects in the module of
/// \p SrcMF into global objects of the module of \p DestMMI
/// \param [in] DestMMI the Machine Module Info \p SrcMF will be cloned into
/// \param [out] SrcToDstMIMap (optional) if not nullptr, returns a mapping
/// between the instructions in the \p SrcMF and the cloned Machine Function
/// \return on success, a unique pointer to the cloned Machine Function; On
/// failure, an \c llvm::Error describing the issue encountered during the
/// process
/// \sa llvm::CloneModule
llvm::Expected<std::unique_ptr<llvm::MachineFunction>> cloneMF(
    const llvm::MachineFunction *SrcMF, const llvm::ValueToValueMapTy &VMap,
    llvm::MachineModuleInfo &DestMMI,
    llvm::DenseMap<llvm::MachineInstr *, llvm::MachineInstr *> *SrcToDstMIMap =
        nullptr);

/// Clones the content of the source Machine Module Info \p SrcMMI into
/// the \p DestMMI. Uses the \p VMap to map the global objects in the source
/// module into the destination, already cloned module
/// \param [in] SrcMMI the source \c llvm::MachineModuleInfo
/// \param [in] VMap a mapping between a global object in the source module and
/// its copy in the destination module
/// \param [out] DestMMI the destination \c llvm::MachineModuleInfo
/// \param [out] SrcToDstMIMap (optional) if not \c nullptr, returns a
/// mapping between the machine instructions of the \p SrcMMI to the machine
/// instructions of the \c DestMMI
/// \return an \c llvm::Error if an issue was encountered during the process
llvm::Error cloneMMI(
    const llvm::MachineModuleInfo &SrcMMI, const llvm::Module &SrcModule,
    const llvm::ValueToValueMapTy &VMap, llvm::MachineModuleInfo &DestMMI,
    llvm::DenseMap<llvm::MachineInstr *, llvm::MachineInstr *> *SrcToDstMIMap);

} // namespace luthier

#endif