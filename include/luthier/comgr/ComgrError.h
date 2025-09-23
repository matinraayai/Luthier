//===-- ComgrError.h --------------------------------------------*- C++ -*-===//
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
/// Describes the \c ComgrError class which encapsulates errors
/// originating from the AMD COMGR library.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMGR_COMGR_ERROR_H
#define LUTHIER_COMGR_COMGR_ERROR_H
#include <amd_comgr/amd_comgr.h>
#include "luthier/common/ROCmLibraryError.h"

namespace luthier {

/// \brief Encapsulates errors originating from AMD Comgr library
class ComgrError final : RocmLibraryError {
  const std::optional<amd_comgr_status_t> Error;

  ComgrError(std::string ErrorMsg,
             const std::optional<amd_comgr_status_t> Error,
             const std::source_location ErrorLocation =
                 std::source_location::current(),
             std::stacktrace StackTrace = std::stacktrace::current())
      : RocmLibraryError(std::move(ErrorMsg), ErrorLocation,
                         std::move(StackTrace)),
        Error(Error) {};

  ComgrError(const llvm::formatv_object_base &ErrorMsg,
             const std::optional<amd_comgr_status_t> Error,
             const std::source_location ErrorLocation =
                 std::source_location::current(),
             std::stacktrace StackTrace = std::stacktrace::current())
      : RocmLibraryError(std::move(ErrorMsg.str()), ErrorLocation,
                         std::move(StackTrace)),
        Error(Error) {};

public:
  static char ID;

  void log(llvm::raw_ostream &OS) const override;
};

#define LUTHIER_COMGR_CALL_ERROR_CHECK(Expr, ErrorMsg)                         \
  [&]() {                                                                      \
    if (const amd_comgr_status_t Status = Expr;                                \
        Status != AMD_COMGR_STATUS_SUCCESS) {                                  \
      return llvm::make_error<ComgrError>(ErrorMsg, Status);                   \
    }                                                                          \
    return llvm::Error::success();                                             \
  }()

} // namespace luthier

#endif