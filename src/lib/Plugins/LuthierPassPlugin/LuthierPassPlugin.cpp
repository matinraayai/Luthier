//===-- LuthierPassPlugin.cpp ---------------------------------------------===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
/// \file LuthierPassPlugin.cpp
/// Implements Luthier's legacy pass manager plugin for use with Luthier's llc
/// fork.
//===----------------------------------------------------------------------===//
#include "luthier/Plugins/LuthierPassPlugin.h"
#include "luthier/Common/GenericLuthierError.h"
#include <llvm/IR/Module.h>

namespace luthier {

llvm::Expected<PassPlugin> PassPlugin::Load(const std::string &Filename) {
  std::string Error;
  auto Library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(Filename.c_str(), &Error);
  if (!Library.isValid())
    return LUTHIER_MAKE_GENERIC_ERROR(
        (llvm::Twine("Could not load library '") + Filename + "': " + Error)
            .str());

  PassPlugin P{Filename, Library};

  // llvmGetPassPluginInfo should be resolved to the definition from the plugin
  // we are currently loading.
  auto GetDetailsFn = reinterpret_cast<intptr_t>(
      Library.getAddressOfSymbol("luthierGetPassPluginInfo"));

  if (!GetDetailsFn)
    // If the symbol isn't found, this is probably a legacy plugin, which is an
    // error
    return LUTHIER_MAKE_GENERIC_ERROR(
        (llvm::Twine("Plugin entry point not found in '") + Filename +
         "'. Is this a legacy plugin?")
            .str());

  P.Info =
      reinterpret_cast<decltype(luthierGetPassPluginInfo) *>(GetDetailsFn)();

  if (P.Info.APIVersion != LUTHIER_PASS_PLUGIN_API_VERSION)
    return LUTHIER_MAKE_GENERIC_ERROR(
        (llvm::Twine("Wrong API version on plugin '") + Filename +
         "'. Got version " + llvm::Twine(P.Info.APIVersion) +
         ", supported version is " +
         llvm::Twine(LUTHIER_PASS_PLUGIN_API_VERSION) + ".")
            .str());

  return P;
}

std::unique_ptr<llvm::Module> PassPlugin::instrumentationModuleCreationCallback(
    llvm::LLVMContext &Context, const llvm::Triple &TT, llvm::StringRef CPUName,
    llvm::StringRef FS) const {
  if (Info.IModuleCreationCallback)
    return Info.IModuleCreationCallback(Context, TT, CPUName, FS,
                                        Info.ExtraArgs);
  return nullptr;
}
} // namespace luthier