//===-- HipError.h ----------------------------------------------*- C++ -*-===//
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
/// Defines an \c llvm::ErrorInfo for issues regarding the HIP runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HIP_HIP_ERROR_H
#define LUTHIER_HIP_HIP_ERROR_H
#include <hip/hip_runtime_api.h>
#include <luthier/common/ROCmLibraryError.h>

namespace luthier::hip {
class HipError final : RocmLibraryError {
  const std::optional<hipError_t> Error;

public:
  HipError(std::string ErrorMsg, const std::optional<hipError_t> Error,
           const std::source_location ErrorLocation =
               std::source_location::current(),
           StackTraceType StackTrace = StackTraceInitializer())
      : RocmLibraryError(std::move(ErrorMsg), ErrorLocation,
                         std::move(StackTrace)),
        Error(Error) {};

  HipError(const llvm::formatv_object_base &ErrorMsg,
           const std::optional<hipError_t> Error,
           const std::source_location ErrorLocation =
               std::source_location::current(),
           StackTraceType StackTrace = StackTraceInitializer())
      : RocmLibraryError(ErrorMsg.str(), ErrorLocation, std::move(StackTrace)),
        Error(Error) {};

  static char ID;

  void log(llvm::raw_ostream &OS) const override;
};

#define LUTHIER_HIP_CALL_ERROR_CHECK(Expr, ErrorMsg)                           \
  [&]() {                                                                      \
    if (const hipError_t Status = Expr; Status != hipSuccess) {                \
      return llvm::make_error<luthier::hip::HipError>(ErrorMsg, Status);       \
    }                                                                          \
    return llvm::Error::success();                                             \
  }()

} // namespace luthier::hip

#endif