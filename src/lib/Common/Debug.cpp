//===-- Debug.cpp ---------------------------------------------------------===//
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
/// Implements \c luthier::registerDebugCLOptions, which restores the standard
/// \c -debug* command line options when LLVM was built without debug support.
//===----------------------------------------------------------------------===//
#include "luthier/Common/Debug.h"
#include "luthier/Common/NeverDestroyed.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <string>
#include <vector>

// When this translation unit is compiled with NDEBUG, <llvm/Support/Debug.h>
// turns isCurrentDebugType / setCurrentDebugType(s) into no-op function-like
// macros. The real functions are nonetheless defined in libLLVMSupport
// regardless of how LLVM was built (see llvm/lib/Support/Debug.cpp), so undo
// the macros and forward-declare the real entry points to drive them — the
// same technique LLVM's own Debug.cpp uses. DebugFlag / EnableDebugBuffering
// are plain externs and are never macro-ized.
#ifdef isCurrentDebugType
#undef isCurrentDebugType
#endif
#ifdef setCurrentDebugType
#undef setCurrentDebugType
#endif
#ifdef setCurrentDebugTypes
#undef setCurrentDebugTypes
#endif

namespace llvm {
void setCurrentDebugTypes(const char **Types, unsigned Count);
/// Registers LLVM's own -debug* options when LLVM has debug support; an empty
/// no-op otherwise. Declared internally in LLVM (DebugOptions.h) but always
/// exported from libLLVMSupport.
void initDebugOptions();
} // namespace llvm

namespace luthier {

namespace {

/// Location object for the Luthier-owned \c -debug-only option, mirroring the
/// (anonymous) \c DebugOnlyOpt in LLVM's \c Debug.cpp: assigning the parsed
/// string splits it on commas and forwards to \c llvm::setCurrentDebugTypes,
/// turning \c DebugFlag on as a side effect.
struct DebugOnlyOpt {
  void operator=(const std::string &Val) const {
    if (Val.empty())
      return;
    llvm::DebugFlag = true;
    llvm::SmallVector<llvm::StringRef, 8> Split;
    llvm::StringRef(Val).split(Split, ',', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);
    std::vector<std::string> Owned;
    Owned.reserve(Split.size());
    for (llvm::StringRef S : Split)
      Owned.push_back(S.str());
    llvm::SmallVector<const char *, 8> CStrs;
    CStrs.reserve(Owned.size());
    for (const std::string &S : Owned)
      CStrs.push_back(S.c_str());
    llvm::setCurrentDebugTypes(CStrs.data(), CStrs.size());
  }
};

/// Each registrar constructs its \c cl::opt exactly once, into a never-
/// destroyed slot so the option outlives command line parsing and any later
/// access without a static-destruction-order hazard.
void registerDebug() {
  static NeverDestroyed<llvm::cl::opt<bool, /*ExternalStorage=*/true>> Opt(
      "debug", llvm::cl::desc("Enable debug output"), llvm::cl::Hidden,
      llvm::cl::location(llvm::DebugFlag));
  (void)Opt;
}

void registerDebugOnly() {
  static NeverDestroyed<DebugOnlyOpt> Loc;
  static NeverDestroyed<llvm::cl::opt<DebugOnlyOpt, /*ExternalStorage=*/true,
                                      llvm::cl::parser<std::string>>>
      Opt("debug-only",
          llvm::cl::desc("Enable a specific type of debug output (comma "
                         "separated list of types)"),
          llvm::cl::Hidden, llvm::cl::value_desc("debug string"),
          llvm::cl::location(*Loc), llvm::cl::ValueRequired);
  (void)Opt;
}

void registerDebugBufferSize() {
  static NeverDestroyed<llvm::cl::opt<unsigned>> Opt(
      "debug-buffer-size",
      llvm::cl::desc("Buffer the last N characters of debug output until "
                     "program termination. [default 0 -- immediate print-out]"),
      llvm::cl::Hidden, llvm::cl::init(0));
  (void)Opt;
}

} // namespace

void registerDebugCLOptions() {
  // Let LLVM register its own -debug* options if this LLVM has debug support.
  // Idempotent: the underlying options are ManagedStatics constructed once.
  llvm::initDebugOptions();
  // Opt into buffered debug output so luthier::dbgs() honors
  // -debug-buffer-size.
  llvm::EnableDebugBuffering = true;
  // Fill in whatever LLVM did not register (i.e. LLVM built without debug
  // support). The count() guard avoids a duplicate-option fatal otherwise.
  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &Opts =
      llvm::cl::getRegisteredOptions();
  if (!Opts.count("debug"))
    registerDebug();
  if (!Opts.count("debug-only"))
    registerDebugOnly();
  if (!Opts.count("debug-buffer-size"))
    registerDebugBufferSize();
}

} // namespace luthier
