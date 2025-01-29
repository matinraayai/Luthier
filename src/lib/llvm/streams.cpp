//===-- streams.cpp -------------------------------------------------------===//
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
/// \file This files implements versions of <tt>llvm::outs</tt>,
/// <tt>llvm::errs</tt>, and <tt>llvm::nulls</tt> that are safe to use with
/// Luthier tools.
//===----------------------------------------------------------------------===//
#include "luthier/llvm/streams.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include "luthier/llvm/EagerManagedStatic.h"
#include <llvm/Support/FileSystem.h>

namespace luthier {

llvm::raw_fd_ostream &outs() {
  // Set buffer settings to model stdout behavior.
  std::error_code EC;
#ifdef __MVS__
  EC = enablezOSAutoConversion(STDOUT_FILENO);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC, "Failed to initialize the standard output raw_fd_stream."));
#endif
  static EagerManagedStatic<llvm::raw_fd_ostream> S("-", EC,
                                                    llvm::sys::fs::OF_None);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC, "Failed to initialize the standard output raw_fd_stream."));
  return *S;
}

llvm::raw_fd_ostream &errs() {
  // Set standard error to be unbuffered.
#ifdef __MVS__
  std::error_code EC = enablezOSAutoConversion(STDERR_FILENO);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC, "Failed to initialize the standard error raw_fd_stream."));
#endif
  static EagerManagedStatic<llvm::raw_fd_ostream> S(STDERR_FILENO, false, true);
  return *S;
}

llvm::raw_ostream &nulls() {
  static EagerManagedStatic<llvm::raw_null_ostream> S;
  return *S;
}
} // namespace luthier