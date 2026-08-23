//===-- SubstituteAMDGCNIntrinsicsPass.h -------------------------*- C++-*-===//
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
/// \file
/// Defines the \c SubstituteAMDGCNIntrinsicsPass which re-writes
/// a set of amdgcn intrinsics that require special lowering in the
/// instrumentation module's code generation. Every rewrite routes through
/// \c luthier::readSVA of the appropriate \c ScalarValueArgument slot.
/// Current rewrites:
///   \c llvm.amdgcn.workgroup.id.{x,y,z} -> \c luthier::readSVA of the
///       \c WORKGROUP_ID_{X,Y,Z} slots (returned as \c i32)
///   \c llvm.amdgcn.implicitarg.ptr -> \c luthier::readSVA of the
///       \c IMPLICIT_ARG_BUFFER slot (returned as \c i64 , then \c inttoptr
///       to <tt>ptr addrspace(4)</tt>)
///   \c llvm.amdgcn.workitem.id.{x,y,z} -> per-lane reconstruction from
///       \c luthier::readSVA of the \c WORKITEM_ID_{X,Y,Z} slots (lane-0's
///       workitem IDs), the lane's position within its wave
///       (\c llvm.amdgcn.mbcnt.lo /\c .hi ), and the workgroup dimensions
///       loaded from the implicit-arg buffer
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_IR_COMPILATION_SUBSTITUTE_AMDGCN_INTRINSICS_PASS_H
#define LUTHIER_TOOL_IR_COMPILATION_SUBSTITUTE_AMDGCN_INTRINSICS_PASS_H
#include <llvm/IR/PassManager.h>

namespace llvm {
class Module;
}

namespace luthier {

class SubstituteAMDGCNIntrinsicsPass
    : public llvm::PassInfoMixin<SubstituteAMDGCNIntrinsicsPass> {
public:
  SubstituteAMDGCNIntrinsicsPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }

  static llvm::StringRef name() {
    return "luthier-substitute-amdgcn-intrinsics";
  }
};

} // namespace luthier

#endif
