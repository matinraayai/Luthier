//===-- LegacyPlugin.cpp --------------------------------------------------===//
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
/// \file
/// Implements Luthier's legacy pass manager plugin for use with Luthier's llc
/// fork.
//===----------------------------------------------------------------------===//
#include "luthier/Plugins/LegacyPlugin.h"
#include "luthier/Common/GenericLuthierError.h"

namespace luthier {

llvm::Expected<LegacyPassPlugin>
LegacyPassPlugin::Load(const std::string &Filename) {
  std::string Error;
  auto Library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(Filename.c_str(), &Error);
  if (!Library.isValid())
    return LUTHIER_MAKE_GENERIC_ERROR(
        (llvm::Twine("Could not load library '") + Filename + "': " + Error)
            .str());

  LegacyPassPlugin P{Filename, Library};

  // llvmGetPassPluginInfo should be resolved to the definition from the plugin
  // we are currently loading.
  auto getDetailsFn = reinterpret_cast<intptr_t>(
      Library.getAddressOfSymbol("luthierGetLegacyPassPluginInfo"));

  if (!getDetailsFn)
    // If the symbol isn't found, this is probably a legacy plugin, which is an
    // error
    return LUTHIER_MAKE_GENERIC_ERROR(
        (llvm::Twine("Plugin entry point not found in '") + Filename +
         "'. Is this a legacy plugin?")
            .str());

  P.Info = reinterpret_cast<decltype(luthierGetLegacyPassPluginInfo) *>(
      getDetailsFn)();

  if (P.Info.APIVersion != LUTHIER_LEGACY_PLUGIN_API_VERSION)
    return LUTHIER_MAKE_GENERIC_ERROR(
        (llvm::Twine("Wrong API version on plugin '") + Filename +
         "'. Got version " + llvm::Twine(P.Info.APIVersion) +
         ", supported version is " +
         llvm::Twine(LUTHIER_LEGACY_PLUGIN_API_VERSION) + ".")
            .str());

  return P;
}
} // namespace luthier