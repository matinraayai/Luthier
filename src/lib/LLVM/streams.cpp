//===-- streams.cpp -------------------------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// Implements versions of <tt>llvm::outs</tt>, <tt>llvm::errs</tt>,
/// <tt>llvm::nulls</tt>, and <tt>llvm::dbgs</tt> that are safe to use with
/// Luthier tools.
///
/// Each stream is held in a \c luthier::NeverDestroyed slot, so it lives for
/// the whole process and is never torn down by \c llvm_shutdown. This avoids
/// the static-destruction-order fiasco under \c LD_PRELOAD (a late destructor
/// ordered after the tool finalizer can still legally use these streams).
//===----------------------------------------------------------------------===//
#include "luthier/LLVM/streams.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Common/LuthierError.h"
#include "luthier/Common/NeverDestroyed.h"
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/circular_raw_ostream.h>

namespace luthier {

llvm::raw_fd_ostream &outs() {
  // Set buffer settings to model stdout behavior.
  std::error_code EC;
#ifdef __MVS__
  EC = enablezOSAutoConversion(STDOUT_FILENO);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC, "Failed to initialize the standard output raw_fd_stream."));
#endif
  static NeverDestroyed<llvm::raw_fd_ostream> S("-", EC,
                                                llvm::sys::fs::OF_None);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !EC, "Failed to initialize the standard output raw_fd_stream."));
  // The stream is never destroyed, so its destructor will not flush the
  // buffered standard output. The caller is responsible for flushing the tail
  // at the end of tool teardown via finalizeStreams().
  return *S;
}

llvm::raw_fd_ostream &errs() {
  // Set standard error to be unbuffered.
#ifdef __MVS__
  std::error_code EC = enablezOSAutoConversion(STDERR_FILENO);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC, "Failed to initialize the standard error raw_fd_stream."));
#endif
  static NeverDestroyed<llvm::raw_fd_ostream> S(STDERR_FILENO, /*shouldClose=*/
                                                false, /*unbuffered=*/true);
  return *S;
}

llvm::raw_ostream &nulls() {
  static NeverDestroyed<llvm::raw_null_ostream> S;
  return *S;
}

#ifndef NDEBUG

namespace {

/// Resolves the circular-buffer size for \c dbgs from the standard \c -debug /
/// \c -debug-buffer-size options, exactly like \c llvm::dbgs. Returns \c 0
/// (immediate, unbuffered pass-through) unless buffering was both enabled by
/// the tool (\c llvm::EnableDebugBuffering) and requested at runtime
/// (\c -debug with a nonzero \c -debug-buffer-size). The \c -debug-buffer-size
/// value is read back from LLVM's already-registered option so the exact same
/// command line flag controls both \c llvm::dbgs and \c luthier::dbgs.
unsigned debugBufferSize() {
  if (!llvm::EnableDebugBuffering || !llvm::DebugFlag)
    return 0;
  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &Opts =
      llvm::cl::getRegisteredOptions();
  auto It = Opts.find("debug-buffer-size");
  if (It == Opts.end())
    return 0;
  return static_cast<llvm::cl::opt<unsigned> *>(It->second)->getValue();
}

/// Fatal-signal handler that dumps the buffered debug log, mirroring LLVM's
/// own \c debug_user_sig_handler. Reached only in the buffered configuration.
void debugSignalHandler(void *) {
  static_cast<llvm::circular_raw_ostream &>(luthier::dbgs())
      .flushBufferWithBanner();
}

} // namespace

llvm::raw_ostream &dbgs() {
  static const unsigned BufferSize = debugBufferSize();
  static NeverDestroyed<llvm::circular_raw_ostream> S(
      luthier::errs(), "*** Luthier Debug Log Output ***\n", BufferSize,
      llvm::circular_raw_ostream::REFERENCE_ONLY);
  // When buffering, install the fatal-signal handler so a crash still dumps the
  // buffered tail (mirrors llvm::dbgs). Normal-exit flushing of the tail is the
  // caller's responsibility via finalizeStreams(). Pass-through (BufferSize ==
  // 0) needs neither.
  static const int Init = [] {
    if (BufferSize != 0)
      llvm::sys::AddSignalHandler(&debugSignalHandler, nullptr);
    return 0;
  }();
  (void)Init;
  return *S;
}

#else

llvm::raw_ostream &dbgs() { return luthier::errs(); }

#endif

void initializeStreams() {
  (void)outs();
  (void)errs();
  (void)nulls();
#ifndef NDEBUG
  // Also constructs the circular debug stream and installs its signal handler.
  (void)dbgs();
#endif
}

void finalizeStreams() {
  // outs() is buffered; flush its tail. errs()/nulls() are unbuffered.
  outs().flush();
#ifndef NDEBUG
  // Dump the buffered debug log with its banner. flushBufferWithBanner() is a
  // no-op when dbgs() is an unbuffered pass-through (BufferSize == 0).
  auto &D = static_cast<llvm::circular_raw_ostream &>(dbgs());
  D.flush();
  D.flushBufferWithBanner();
#endif
}

} // namespace luthier
