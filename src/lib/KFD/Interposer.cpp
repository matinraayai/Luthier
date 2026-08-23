//===-- Interposer.cpp - LD_PRELOAD entry point for the KFD wrapper -------===//
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
/// The one symbol that has to be interposed for KFD-level interception to work.
///
/// Kept in its own file, and only in the preloadable shared library, so that
/// linking the wrapper's logic into a test or into Luthier itself does not
/// silently replace \c ioctl for the whole process. Everything of substance
/// lives in \c QueueWrapper.cpp.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/QueueWrapper.h"

extern "C" {

/// \brief Stands in for the C library's \c ioctl.
///
/// The real \c ioctl is variadic (\c int ioctl(int, unsigned long, ...)). We
/// declare the three-argument form because every KFD call passes exactly one
/// pointer. That is a deliberate simplification and would be wrong for a driver
/// that passes an argument by value.
int ioctl(int Fd, unsigned long Request, void *Arg) {
  return luthier::kfd::handleIoctl(Fd, Request, Arg);
}

} // extern "C"
