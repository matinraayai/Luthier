//===-- InitialEntryPointAnalysis.cpp -------------------------------------===//
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
/// \file InitialEntryPointAnalysis.cpp
/// Implements the \c InitialEntryPointAnalysis class together with the
/// accessors for the target module's \c luthier.initial_entry_point metadata.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/Common/GenericLuthierError.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>

namespace luthier {

llvm::AnalysisKey InitialEntryPointAnalysis::Key;

void setInitialEntryPoint(llvm::Module &M, const EntryPoint &EP) {
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::NamedMDNode *NMD =
      M.getOrInsertNamedMetadata(InitialEntryPointMDName);
  NMD->clearOperands();
  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt64Ty(Ctx), EP.getRawAddress())),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(Ctx), EP.isKernel()))};
  NMD->addOperand(llvm::MDNode::get(Ctx, Ops));
}

llvm::Expected<EntryPoint> getInitialEntryPoint(const llvm::Module &M) {
  const llvm::NamedMDNode *NMD =
      M.getNamedMetadata(InitialEntryPointMDName);
  if (!NMD || NMD->getNumOperands() != 1)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Module '" + M.getName().str() + "' does not record exactly one '" +
        InitialEntryPointMDName.str() + "' entry");

  const llvm::MDNode *Node = NMD->getOperand(0);
  if (!Node || Node->getNumOperands() != 2)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "'" + InitialEntryPointMDName.str() +
        "' must hold a !{i64 address, i1 is_kernel_descriptor} node");

  const auto *AddrMD =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(Node->getOperand(0));
  const auto *IsKernelMD =
      llvm::dyn_cast<llvm::ConstantAsMetadata>(Node->getOperand(1));
  if (!AddrMD || !IsKernelMD)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "'" + InitialEntryPointMDName.str() +
        "' operands must both be integer constants");

  const auto *Addr = llvm::dyn_cast<llvm::ConstantInt>(AddrMD->getValue());
  const auto *IsKernel =
      llvm::dyn_cast<llvm::ConstantInt>(IsKernelMD->getValue());
  if (!Addr || !IsKernel)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "'" + InitialEntryPointMDName.str() +
        "' operands must both be integer constants");

  uint64_t RawAddress = Addr->getZExtValue();
  if (!IsKernel->isZero())
    return EntryPoint(*reinterpret_cast<const llvm::amdhsa::kernel_descriptor_t *>(
        RawAddress));
  return EntryPoint(RawAddress);
}

InitialEntryPointAnalysis::Result
InitialEntryPointAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
  llvm::Expected<EntryPoint> EPOrErr = getInitialEntryPoint(M);
  if (!EPOrErr) {
    M.getContext().emitError(llvm::toString(EPOrErr.takeError()));
    return Result{EntryPoint{}};
  }
  return Result{*EPOrErr};
}

} // namespace luthier
