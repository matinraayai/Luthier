//===-- InitialExecutionPointAnalysis.cpp ---------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file InitialExecutionPointAnalysis.cpp
/// Implements the \c InitialExecutionPointAnalysis class together with the
/// accessors for the target module's \c luthier.initial_execution_point
/// metadata.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InitialExecutionPointAnalysis.h"
#include "luthier/Common/GenericLuthierError.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

namespace luthier {

llvm::AnalysisKey InitialExecutionPointAnalysis::Key;

void setInitialExecutionPoint(llvm::Module &M,
                              const llvm::amdhsa::kernel_descriptor_t &KD) {
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::NamedMDNode *NMD =
      M.getOrInsertNamedMetadata(InitialExecutionPointMDName);
  NMD->clearOperands();
  llvm::Metadata *Ops[] = {llvm::ConstantAsMetadata::get(
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx),
                             reinterpret_cast<uint64_t>(&KD)))};
  NMD->addOperand(llvm::MDNode::get(Ctx, Ops));
}

llvm::Expected<const llvm::amdhsa::kernel_descriptor_t *>
getInitialExecutionPoint(const llvm::Module &M) {
  const llvm::NamedMDNode *NMD =
      M.getNamedMetadata(InitialExecutionPointMDName);
  if (!NMD || NMD->getNumOperands() != 1)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Module '" + M.getName().str() + "' does not record exactly one '" +
        InitialExecutionPointMDName.str() + "' entry");

  const llvm::MDNode *Node = NMD->getOperand(0);
  if (!Node || Node->getNumOperands() != 1)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "'" + InitialExecutionPointMDName.str() +
        "' must hold a !{i64 kernel_descriptor_address} node");

  const auto *AddrMD =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(Node->getOperand(0));
  if (!AddrMD)
    return LUTHIER_MAKE_GENERIC_ERROR("'" +
                                      InitialExecutionPointMDName.str() +
                                      "' operand must be an integer constant");

  const auto *Addr = llvm::dyn_cast<llvm::ConstantInt>(AddrMD->getValue());
  if (!Addr)
    return LUTHIER_MAKE_GENERIC_ERROR("'" +
                                      InitialExecutionPointMDName.str() +
                                      "' operand must be an integer constant");

  return reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
      Addr->getZExtValue());
}

InitialExecutionPointAnalysis::Result
InitialExecutionPointAnalysis::run(llvm::Module &M,
                                   llvm::ModuleAnalysisManager &) {
  llvm::Expected<const llvm::amdhsa::kernel_descriptor_t *> KDOrErr =
      getInitialExecutionPoint(M);
  if (!KDOrErr) {
    M.getContext().emitError(llvm::toString(KDOrErr.takeError()));
    return Result{nullptr};
  }
  return Result{*KDOrErr};
}

} // namespace luthier
