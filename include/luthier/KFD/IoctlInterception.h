//===-- IoctlInterception.h - the seam under handleIoctl --------*- C++ -*-===//
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
/// How \c luthier::kfd::handleIoctl reaches the next \c ioctl in the chain.
///
/// \par Why this is injected rather than resolved in place
/// The wrapper's logic and the mechanism that intercepts \c ioctl are deliberately
/// in different translation units, so that linking the logic into a test does not
/// replace \c ioctl for the whole process -- \c KFDUnitTests depends on that. The
/// logic therefore cannot call \c dlsym itself, nor call into GOTCHA; it is handed
/// a way to reach the real call.
///
/// \par Why a provider, and not just a function pointer
/// The provider is consulted on \b every call, and that is not an oversight. Under
/// GOTCHA the next function in the chain is not fixed: when another tool inserts
/// itself \e inside us, GOTCHA re-points our wrappee slot at the newly inserted
/// link. A pointer resolved once at startup would keep calling whatever was there
/// first and silently skip every tool that registered later -- and since load order
/// decides which constructor runs first, that failure would be intermittent.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_IOCTL_INTERCEPTION_H
#define LUTHIER_KFD_IOCTL_INTERCEPTION_H

namespace luthier::kfd {

/// \brief The real \c ioctl, as the three-argument form the KFD path always uses.
///
/// \par Assumption A1, and why it holds
/// The C library's \c ioctl is variadic. Every KFD request passes exactly one
/// pointer, so declaring the three-argument form is a deliberate simplification --
/// but note where it now applies: the interception layer wraps \c ioctl for the
/// \e whole process, so a genuine two-argument call (a terminal \c ioctl, say)
/// also arrives here, and its third parameter is whatever happened to be in the
/// register. It is never dereferenced: \c handleIoctl establishes that the
/// descriptor is \c /dev/kfd before reading \c Arg at all.
///
/// So this is safe **by the x86-64 SysV calling convention** -- a variadic callee
/// receives its third argument in \c rdx regardless of how the caller declared it
/// -- and not by anything the code does. Worth knowing before porting this
/// anywhere with a different convention, or before adding a filter that reads
/// \c Arg earlier than the descriptor check.
using RealIoctlFn = int (*)(int, unsigned long, void *);

/// \brief Returns the next \c ioctl to call. Consulted per call; see the file
/// comment for why it must not be cached.
using RealIoctlProviderFn = RealIoctlFn (*)();

/// \brief Install the way \c handleIoctl reaches the real \c ioctl.
///
/// Called by whichever interception layer is linked in. Without it, \c handleIoctl
/// aborts with a message saying so rather than crashing: a wrapper that intercepts
/// but cannot forward would hang the application on work that never reaches the
/// driver, which is far harder to diagnose than an immediate abort.
void setRealIoctlProvider(RealIoctlProviderFn Provider);

} // namespace luthier::kfd

#endif // LUTHIER_KFD_IOCTL_INTERCEPTION_H
