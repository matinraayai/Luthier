//===-- ErrorCheck.h - Luthier Error Checking Macros  -----------*- C++ -*-===//
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
/// This file contains useful macros to check for \c llvm::Error in Luthier.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_ERROR_CHECK_H
#define LUTHIER_ERROR_CHECK_H

/// \brief Reports a fatal error if the passed \p llvm::Error argument is not
/// equal to \c llvm::Error::success()
#define LUTHIER_REPORT_FATAL_ON_ERROR(Error)                                   \
  do {                                                                         \
    if (Error) {                                                               \
      llvm::report_fatal_error(std::move(Error), true);                        \
    }                                                                          \
  } while (0)

/// \brief returns from the function if the passed \p llvm::Error argument is
/// not equal to \c llvm::Error::success()
#define LUTHIER_RETURN_ON_ERROR(Error)                                         \
  do {                                                                         \
    if (Error) {                                                               \
      return (Error);                                                          \
    }                                                                          \
  } while (0)

#endif