//===-- RocprofilerError.cpp ----------------------------------------------===//
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
/// Implements the rocprofiler \c llvm::ErrorInfo class.
//===----------------------------------------------------------------------===//
#include "luthier/rocprofiler-sdk/RocprofilerError.h"
#include <llvm/Support/Signals.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace luthier::rocprofiler {

char RocprofilerError::ID;

void RocprofilerError::log(llvm::raw_ostream &OS) const {
  OS << "Rocprofiler SDK ";
  if (Error.has_value())
    OS << "error code" << *Error;
  else
    OS << "error";
  OS << " encountered in file " << ErrorLocation.file_name() << ", function "
     << ErrorLocation.function_name() << ", at " << ErrorLocation.line() << ": "
     << ErrorMsg << ".\n";
  OS << "Stack trace: \n";
#ifdef __cpp_lib_stacktrace
  OS << std::to_string(StackTrace);
#else
  OS << StackTrace;
#endif
  OS << "\n";
}

} // namespace luthier::rocprofiler