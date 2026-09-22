//===-- DynamicLibrary.cpp ------------------------------------------------===//
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
/// Implements the \c luthier::DynamicLibrary class.
//===----------------------------------------------------------------------===//
#include "luthier/Common/DynamicLibrary.h"
#include "luthier/Common/GenericLuthierError.h"

namespace luthier {

llvm::Error DynamicLibrary::getLastError() {
  const char *Error = ::dlerror();
  return Error != nullptr ? LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                                "Dynamic library failure: {0}", Error))
                          : llvm::Error::success();
}

llvm::Expected<DynamicLibrary> DynamicLibrary::open(const char *Path,
                                                    int Flags) {
  DynamicLibrary Out{::dlopen(Path, Flags), /*Owned=*/true};
  if (Out.Handle == nullptr) {
    return getLastError();
  }
  return Out;
}

llvm::Expected<DynamicLibrary>
DynamicLibrary::openInNamespace(Lmid_t Lmid, const char *Path, int Flags) {
  DynamicLibrary Out{::dlmopen(Lmid, Path, Flags), /*Owned=*/true};
  if (Out.Handle == nullptr) {
    return getLastError();
  }
  return Out;
}

DynamicLibrary &DynamicLibrary::operator=(DynamicLibrary &&Other) noexcept {
  if (this != &Other) {
    close();
    Handle = std::exchange(Other.Handle, nullptr);
    Owned = std::exchange(Other.Owned, false);
  }
  return *this;
}

DynamicLibrary::~DynamicLibrary() { close(); }

void DynamicLibrary::close() {
  if (Owned && Handle != nullptr)
    ::dlclose(Handle);
  Handle = nullptr;
  Owned = false;
}

void *DynamicLibrary::release() {
  Owned = false;
  return std::exchange(Handle, nullptr);
}

Lmid_t DynamicLibrary::getNamespace() const {
  Lmid_t Lmid = -1;
  if (Handle == nullptr || ::dlinfo(Handle, RTLD_DI_LMID, &Lmid) != 0)
    return -1;
  return Lmid;
}

} // namespace luthier
