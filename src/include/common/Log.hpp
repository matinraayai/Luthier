//===-- Log.hpp - Luthier Logging Utilities -------------------------------===//
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
/// This file contains utility functions and macros used for logging in Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_COMMON_LOG_HPP
#define LUTHIER_COMMON_LOG_HPP

#include <llvm/Support/FormatVariadic.h>

#define LUTHIER_LOG_FUNCTION_CALL_START                                        \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("<< Function call to {0} >>\n",     \
                                           __PRETTY_FUNCTION__));
#define LUTHIER_LOG_FUNCTION_CALL_END                                          \
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("<< Return from function {0}>>\n",  \
                                           __PRETTY_FUNCTION__));

#endif // LUTHIER_SRC_LOG_HPP
