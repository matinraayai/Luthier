//===-- RocmLibraryError.h --------------------------------------*- C++ -*-===//
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
/// This file describes the \c RocmLibraryError class,
/// used as a base class for creating specialized \c llvm::ErrorInfo for
/// ROCm libraries.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_ROCM_LIBRARY_ERROR_H
#define LUTHIER_COMMON_ROCM_LIBRARY_ERROR_H
#include <luthier/Common/LuthierError.h>

namespace luthier {

class RocmLibraryError : public LuthierError {
protected:
  explicit RocmLibraryError(std::string ErrorMsg,
                            const std::source_location ErrorLocation =
                                std::source_location::current(),
                            StackTraceType StackTrace = StackTraceInitializer())
      : LuthierError(std::move(ErrorMsg), ErrorLocation,
                     std::move(StackTrace)) {};

  explicit RocmLibraryError(const llvm::formatv_object_base &FormatObject,
                            const std::source_location ErrorLocation =
                                std::source_location::current(),
                            StackTraceType StackTrace = StackTraceInitializer())
      : LuthierError(std::move(FormatObject.str()), ErrorLocation,
                     std::move(StackTrace)){};

public:
  static char ID;
};

} // namespace luthier

#endif