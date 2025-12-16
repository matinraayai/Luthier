//===-- comgr.hpp - AMD CoMGR High-level Wrapper --------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This files defines wrappers around AMD CoMGR functionality frequently
/// used by Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMGR_COMGR_H
#define LUTHIER_COMGR_COMGR_H
#include <llvm/Support/Error.h>

namespace luthier::comgr {

/// Links the relocatable object file \p Code to an executable,
/// which can then be loaded into the HSA runtime
/// \param [in] Code the relocatable file
/// \param [out] Out the linked executable
/// \return an \c llvm::Error indicating the success or failure of the operation
llvm::Error linkRelocatableToExecutable(llvm::ArrayRef<char> Code,
                                        llvm::SmallVectorImpl<char> &Out);

} // namespace luthier::comgr

#endif