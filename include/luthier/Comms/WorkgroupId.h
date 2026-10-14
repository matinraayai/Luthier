//===-- WorkgroupId.h - Workgroup id inside an injected payload -*- C++ -*-===//
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
///
/// \file
/// Reads the workgroup id (blockIdx) of the executing wave inside an
/// injected payload, to fill \c MessageInfo::BlockIdx*. HIP's \c blockIdx in
/// a hook (and \c luthier::readSVA(WORKGROUP_ID_*)) returns wrong values in
/// a payload today, so the ids are read from the trap temporaries instead.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMS_WORKGROUP_ID_H
#define LUTHIER_COMMS_WORKGROUP_ID_H

#include <GCNSubtarget.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <luthier/Intrinsic/IntrinsicCalls.h>

namespace luthier::comms {

/// Appends the workgroup id x, y, z (three i32 values) to \p Args, built at
/// \p B's insertion point inside a payload of module \p M.
///
/// TTMPs ("trap temporaries") are SGPRs reserved for the trap handler, which
/// application code never writes; at wave launch the hardware/firmware leaves
/// the workgroup id in them: TTMP8 / 9 / 10 up to GFX11 (verified on gfx908),
/// and with architected SGPRs (GFX12) TTMP9 = x, TTMP7 = y | z << 16.
inline void appendWorkgroupIdArgs(const llvm::GCNSubtarget &ST,
                                  llvm::Module &M, llvm::IRBuilderBase &B,
                                  llvm::SmallVectorImpl<llvm::Value *> &Args) {
  auto ReadReg = [&](llvm::MCRegister R) -> llvm::Value * {
    return insertCallToIntrinsic(M, B, "luthier::readReg", *B.getInt32Ty(),
                                 uint32_t(R.id()));
  };
  if (ST.hasArchitectedSGPRs()) {
    llvm::Value *YZ = ReadReg(llvm::AMDGPU::TTMP7);
    Args.append({ReadReg(llvm::AMDGPU::TTMP9), B.CreateAnd(YZ, 0xffff),
                 B.CreateLShr(YZ, 16)});
  } else {
    Args.append({ReadReg(llvm::AMDGPU::TTMP8), ReadReg(llvm::AMDGPU::TTMP9),
                 ReadReg(llvm::AMDGPU::TTMP10)});
  }
}

} // namespace luthier::comms

#endif
