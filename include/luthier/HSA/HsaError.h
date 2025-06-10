//===-- HsaError.h ----------------------------------------------*- C++ -*-===//
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
/// Defines a \c llvm::ErrorInfo for holding information regarding
/// issues encountered with using HSA APIs.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_ERROR_H
#define LUTHIER_HSA_HSA_ERROR_H
#include <hsa/hsa.h>
#include <luthier/Common/ROCmLibraryError.h>

namespace luthier::hsa {
class HsaError final : RocmLibraryError {
  const std::optional<hsa_status_t> Error;

public:
  HsaError(std::string ErrorMsg, const std::optional<hsa_status_t> Error,
           const std::source_location ErrorLocation =
               std::source_location::current(),
           StackTraceType StackTrace = StackTraceInitializer())
      : RocmLibraryError(std::move(ErrorMsg), ErrorLocation,
                         std::move(StackTrace)),
        Error(Error) {};

  HsaError(const llvm::formatv_object_base &ErrorMsg,
           const std::optional<hsa_status_t> Error,
           const std::source_location ErrorLocation =
               std::source_location::current(),
           StackTraceType StackTrace = StackTraceInitializer())
      : RocmLibraryError(ErrorMsg.str(), ErrorLocation, std::move(StackTrace)),
        Error(Error) {};

  static char ID;

  void log(llvm::raw_ostream &OS) const override;
};

#define LUTHIER_HSA_CALL_ERROR_CHECK(Expr, ErrorMsg)                           \
  [&]() {                                                                      \
    if (const hsa_status_t Status = Expr; Status != HSA_STATUS_SUCCESS) {      \
      return llvm::make_error<luthier::hsa::HsaError>(ErrorMsg, Status);       \
    }                                                                          \
    return llvm::Error::success();                                             \
  }()
} // namespace luthier::hsa

#endif