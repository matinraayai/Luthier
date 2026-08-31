//===-- GotchaIoctlInterceptor.cpp - GOTCHA-based ioctl interception ------===//
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
/// Installs the process-wide interception with GOTCHA, so that several
/// independently built components can observe the same driver traffic in an order
/// they choose.
///
/// Two symbols' worth of it. \c ioctl is the one everything else here is about.
/// The \c open family is wrapped for a narrower purpose -- handing a later opener
/// of a DRM render node the descriptor already bound for that GPU, which is what
/// lets HSA coexist with an application that drives KFD itself. That redirection
/// is off unless a tool switches it on; see \c FdSharing.h.
///
/// \par Why this replaced symbol interposition
/// The previous mechanism defined \c ioctl in this library and reached the real one
/// with \c dlsym(RTLD_NEXT). That chains correctly, but the order is whatever the
/// loader's search order happens to be. GOTCHA adds what the existing arrangement
/// cannot express: \c gotcha_set_priority fixes the order independently of load
/// order. Measured, in \c issue-96/: two independently built GOTCHA tools both see
/// every call, and their order follows priority in both preload orders.
///
/// \par This file is kept apart from QueueWrapper.cpp on purpose
/// Compiled only into the preloadable shared library, never into \c LuthierKFD. The
/// wrapper's logic is linked into \c KFDUnitTests, and a constructor calling
/// \c gotcha_wrap in that translation unit would make the unit tests interpose on
/// their own process. \c Interposer.cpp existed for the same reason; only the
/// mechanism has changed.
///
/// \par A classic LD_PRELOAD interposer must not also wrap ioctl
/// GOTCHA resolves the displaced function through the normal symbol search, so a
/// library that \e defines \c ioctl and chains with \c dlsym(RTLD_NEXT) ends up
/// calling GOTCHA's wrapper, which calls it, until the stack is gone. That presents
/// as an ordinary segfault, so this file fails deliberately at a known depth with a
/// message naming the cause instead. Measured in \c issue-96/ (X4), and kept honest
/// by a test: exit 42 plus the message, rather than a crash to be puzzled over.
//===----------------------------------------------------------------------===//
#include "GotchaAbi.h"

#include "luthier/KFD/FdSharing.h"
#include "luthier/KFD/IoctlInterception.h"
#include "luthier/KFD/QueueWrapper.h"

#include <fcntl.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace {

constexpr const char *LogPrefix = "[luthier-kfd] ";

/// The name GOTCHA knows us by. \c gotcha_set_priority is keyed on this string, so
/// it also has to be unique among the tools in a process.
constexpr const char *ToolName = "luthier-kfd";

/// Where we sit relative to other GOTCHA tools. Explicit rather than left unset,
/// because an unset priority falls back to registration order -- the very
/// nondeterminism GOTCHA was adopted to remove. 0 is a deliberate middle: a peer
/// that wants to see traffic before us asks for a higher number, one that wants to
/// see it after asks for a lower.
constexpr int DefaultPriority = 0;

gotcha_wrappee_handle_t IoctlWrappee = nullptr;

/// The \c open family, wrapped so that a later open of a DRM render node can be
/// given the descriptor already bound for that GPU. See \c FdSharing.h for why
/// that is needed and why it is the bound descriptor rather than the first one.
///
/// All four variants, because they are not aliases at the symbol level and
/// missing one is invisible: interposing only \c open and \c openat missed
/// tinygrad entirely, since CPython's \c os.open resolves to \c open64, and the
/// symptom was the address-space collision looking exactly as though none of this
/// were installed.
gotcha_wrappee_handle_t OpenWrappee = nullptr;
gotcha_wrappee_handle_t Open64Wrappee = nullptr;
gotcha_wrappee_handle_t OpenatWrappee = nullptr;
gotcha_wrappee_handle_t Openat64Wrappee = nullptr;
/// Set once installation succeeds. Not exported: the useful question for a
/// caller is whether traffic was actually seen, which
/// luthierKfdInterceptedIoctlCount answers and this cannot.
bool WrapInstalled = false;
unsigned long long InterceptedTotal = 0;

/// Read the next link per call, never cached.
///
/// GOTCHA re-points \c IoctlWrappee when a tool with a lower priority inserts
/// itself inside us. Caching this in the constructor would keep calling whatever
/// was there first and silently skip every tool that loaded later, and since load
/// order decides constructor order, the symptom would be intermittent.
luthier::kfd::RealIoctlFn realIoctlProvider() {
  return reinterpret_cast<luthier::kfd::RealIoctlFn>(
      gotcha_get_wrappee(IoctlWrappee));
}

/// \c open and friends are variadic, and the mode argument is only present when
/// \c O_CREAT is set. Declaring the three-argument form is safe for the same
/// reason it is safe for \c ioctl -- on x86-64 SysV a variadic callee receives
/// its third argument in \c rdx however the caller declared it -- and the mode is
/// only ever read when the flags say it is there.
using OpenFn = int (*)(const char *, int, mode_t);
using OpenatFn = int (*)(int, const char *, int, mode_t);

/// Shared by all four wrappers: hand back the bound descriptor if there is one.
/// \return a descriptor, or -1 to mean "let the real call through", which is also
/// what makes the first opener the party whose descriptor gets bound.
int borrowedOrMinusOne(const char *Path) {
  if (!luthier::kfd::isFdSharingEnabled() ||
      !luthier::kfd::isRenderNodePath(Path))
    return -1;
  return luthier::kfd::borrowBoundRenderNodeFd(Path);
}

int wrapOpen(const char *Path, int Flags, mode_t Mode) {
  if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
    return Borrowed;
  return reinterpret_cast<OpenFn>(gotcha_get_wrappee(OpenWrappee))(Path, Flags,
                                                                   Mode);
}

int wrapOpen64(const char *Path, int Flags, mode_t Mode) {
  if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
    return Borrowed;
  return reinterpret_cast<OpenFn>(gotcha_get_wrappee(Open64Wrappee))(Path, Flags,
                                                                     Mode);
}

int wrapOpenat(int DirFd, const char *Path, int Flags, mode_t Mode) {
  if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
    return Borrowed;
  return reinterpret_cast<OpenatFn>(gotcha_get_wrappee(OpenatWrappee))(
      DirFd, Path, Flags, Mode);
}

int wrapOpenat64(int DirFd, const char *Path, int Flags, mode_t Mode) {
  if (const int Borrowed = borrowedOrMinusOne(Path); Borrowed >= 0)
    return Borrowed;
  return reinterpret_cast<OpenatFn>(gotcha_get_wrappee(Openat64Wrappee))(
      DirFd, Path, Flags, Mode);
}

/// Guards against the LD_PRELOAD conflict described in the file comment.
thread_local int Depth = 0;
constexpr int MaxDepth = 32;

int wrapIoctl(int Fd, unsigned long Request, void *Arg) {
  if (Depth > MaxDepth) {
    static const char Msg[] =
        "[luthier-kfd] ioctl recursion depth exceeded -- the wrappers are "
        "calling each other. Another preloaded library defines ioctl and "
        "resolves the next one with dlsym(RTLD_NEXT); that mechanism and GOTCHA "
        "cannot both wrap the same symbol. Remove it, or convert it to a GOTCHA "
        "tool.\n";
    // write() rather than fprintf: the stack is nearly gone and stdio may itself
    // allocate.
    const ssize_t Ignored = write(2, Msg, sizeof(Msg) - 1);
    (void)Ignored;
    _exit(42);
  }

  __atomic_add_fetch(&InterceptedTotal, 1, __ATOMIC_RELAXED);
  Depth++;
  const int Ret = luthier::kfd::handleIoctl(Fd, Request, Arg);
  Depth--;
  return Ret;
}

// Overridable only to build the mutation target that proves interception is
// actually measured -- see the -badsym target in this directory's CMakeLists.txt.
#ifndef LUTHIER_KFD_WRAP_SYMBOL
#define LUTHIER_KFD_WRAP_SYMBOL "ioctl"
#endif

gotcha_binding_t Bindings[] = {
    {LUTHIER_KFD_WRAP_SYMBOL, reinterpret_cast<void *>(wrapIoctl),
     &IoctlWrappee},
    {"open", reinterpret_cast<void *>(wrapOpen), &OpenWrappee},
    {"open64", reinterpret_cast<void *>(wrapOpen64), &Open64Wrappee},
    {"openat", reinterpret_cast<void *>(wrapOpenat), &OpenatWrappee},
    {"openat64", reinterpret_cast<void *>(wrapOpenat64), &Openat64Wrappee}};

constexpr int NumBindings =
    static_cast<int>(sizeof(Bindings) / sizeof(Bindings[0]));

/// A constructor, because LD_PRELOAD is the only way this library is loaded and
/// every preloaded constructor runs before main -- hence before the application's
/// first ioctl. An explicit init function would need a caller, and there is none.
__attribute__((constructor)) void install() {
  // Priority before wrapping: GOTCHA needs to know where to place us when it
  // inserts us into a chain that may already exist.
  if (gotcha_set_priority(ToolName, DefaultPriority) != GOTCHA_SUCCESS)
    fprintf(stderr, "%sgotcha_set_priority failed; ordering against other "
                    "GOTCHA tools is undefined\n",
            LogPrefix);

  const gotcha_error_t Err = gotcha_wrap(Bindings, NumBindings, ToolName);
  if (Err != GOTCHA_SUCCESS) {
    // Loud, but not fatal, and the distinction is deliberate. This library is
    // injected into applications that did not ask for it, so taking one down
    // because optional instrumentation could not attach is the wrong trade -- and
    // aborting here would kill the application for a reason that looks like a
    // crash in it.
    //
    // The failure that actually matters -- attaching and then silently observing
    // nothing -- is caught instead by luthierKfdInterceptedIoctlCount, which the
    // test harness checks under --require-wrapper. That guard is strictly
    // stronger: it also catches a wrap that succeeded and was then narrowed by a
    // peer, which no check at this point could see.
    fprintf(stderr,
            "%sWARNING: gotcha_wrap(\"%s\") failed with %d, so no ioctl will be "
            "intercepted and this library will observe nothing. The application "
            "is left to run unmodified.\n",
            LogPrefix, LUTHIER_KFD_WRAP_SYMBOL, static_cast<int>(Err));
    return;
  }

  luthier::kfd::setRealIoctlProvider(realIoctlProvider);
  WrapInstalled = true;
}

} // namespace

extern "C" {

/// \brief How many ioctls have passed through the wrapper.
///
/// The stronger guarantee: a non-zero count is evidence that traffic was actually
/// seen, which the fact that the wrap succeeded does not establish -- GOTCHA's
/// library filters are process-global mutable state, so a peer could narrow
/// interception after our wrap succeeded.
unsigned long long luthierKfdInterceptedIoctlCount() {
  return __atomic_load_n(&InterceptedTotal, __ATOMIC_RELAXED);
}

} // extern "C"
