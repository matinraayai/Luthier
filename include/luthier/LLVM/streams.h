//===-- streams.h -----------------------------------------------*- C++ -*-===//
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
/// \file This files defines versions of <tt>llvm::outs</tt>,
/// <tt>llvm::errs</tt>, and <tt>llvm::nulls</tt> that are safe to use with
/// Luthier tools.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_STREAMS_H
#define LUTHIER_STREAMS_H
#include <llvm/Support/raw_ostream.h>

namespace luthier {

/// A version of \c llvm::outs that is safe to use within Luthier
/// \note Always use this function instead of \c llvm::outs inside a Luthier
/// tool to ensure the underlying \c llvm::raw_fd_ostream is not destroyed
/// before the tool's finalizer function is called
llvm::raw_fd_ostream &outs();

/// A version of \c llvm::errs that is safe to use within Luthier
/// \note Always use this function instead of \c llvm::errs inside a Luthier
/// tool to ensure the underlying \c llvm::raw_fd_ostream is not destroyed
/// before the tool's finalizer function is called
llvm::raw_fd_ostream &errs();

/// A version of \c llvm::nulls that is safe to use within Luthier
/// \note Always use this function instead of \c llvm::nulls inside a Luthier
/// tool to ensure the underlying \c llvm::raw_fd_ostream is not destroyed
/// before the tool's finalizer function is called
llvm::raw_ostream &nulls();

} // namespace luthier

#endif