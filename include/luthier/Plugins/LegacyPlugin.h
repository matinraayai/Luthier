//===-- LegacyPlugin.h ------------------------------------------*- C++ -*-===//
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
/// fork. This is a work-around for allowing instrumentation passes written with
/// the legacy pass manager to be tested.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_PLUGINS_LEGACY_PLUGIN_H
#define LUTHIER_PLUGINS_LEGACY_PLUGIN_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include <string>
#include <utility>

namespace llvm {
class Module;
class PassBuilder;
class TargetMachine;
class TargetPassConfig;
class PassRegistry;
} // namespace llvm

namespace luthier {

/// \macro LUTHIER_LEGACY_PLUGIN_API_VERSION
/// Identifies the API version understood by this plugin.
///
/// When a plugin is loaded, the driver will check it's supported plugin version
/// against that of the plugin. A mismatch is an error. The supported version
/// will be incremented for ABI-breaking changes to the \c PassPluginLibraryInfo
/// struct, i.e. when callbacks are added, removed, or reordered.
#define LUTHIER_LEGACY_PLUGIN_API_VERSION 1

extern "C" {
/// Information about the plugin required to load its passes
///
/// This struct defines the core interface for pass plugins and is supposed to
/// be filled out by plugin implementors. Unused function pointers can be set to
/// nullptr. LLVM-side users of a plugin are expected to use the \c PassPlugin
/// class below to interface with it.
struct LegacyPassPluginLibraryInfo {
  /// The API version understood by this plugin, usually \c
  /// LUTHIER_LEGACY_PLUGIN_API_VERSION
  uint32_t APIVersion{};
  /// A meaningful name of the plugin.
  const char *PluginName{};
  /// The version of the plugin.
  const char *PluginVersion{};

  /// The callback for registering plugin passes with the legacy pass registry
  void (*RegisterPassesCallback)(llvm::PassRegistry &) = nullptr;

  /// The callback for augmenting the legacy pass manager via the target
  /// pass config
  void (*AugmentTargetPassConfigCallback)(llvm::TargetPassConfig &,
                                          llvm::TargetMachine &) = nullptr;
};
}

/// A loaded pass plugin.
///
/// An instance of this class wraps a loaded pass plugin and gives access to
/// its interface defined by the \c PassPluginLibraryInfo it exposes.
class LegacyPassPlugin {
public:
  /// Attempts to load a pass plugin from a given file.
  ///
  /// \returns Returns an error if either the library cannot be found or loaded,
  /// there is no public entry point, or the plugin implements the wrong API
  /// version.
  LLVM_ABI static llvm::Expected<LegacyPassPlugin>
  Load(const std::string &Filename);

  /// Get the filename of the loaded plugin.
  [[nodiscard]] llvm::StringRef getFilename() const { return Filename; }

  /// Get the plugin name
  [[nodiscard]] llvm::StringRef getPluginName() const {
    return Info.PluginName;
  }

  /// Get the plugin version
  [[nodiscard]] llvm::StringRef getPluginVersion() const {
    return Info.PluginVersion;
  }

  /// Get the plugin API version
  [[nodiscard]] uint32_t getAPIVersion() const { return Info.APIVersion; }

  /// Invoke the pass registration callback
  void registerPassesCallback(llvm::PassRegistry &PR) const {
    if (Info.RegisterPassesCallback)
      Info.RegisterPassesCallback(PR);
  }

  void registerAugmentTargetPassConfigCallback(llvm::TargetPassConfig &TPC,
                                               llvm::TargetMachine &TM) const {
    if (Info.AugmentTargetPassConfigCallback)
      Info.AugmentTargetPassConfigCallback(TPC, TM);
  }

private:
  LegacyPassPlugin(std::string Filename,
                   const llvm::sys::DynamicLibrary &Library)
      : Filename(std::move(Filename)), Library(Library), Info() {}

  std::string Filename;

  llvm::sys::DynamicLibrary Library;

  LegacyPassPluginLibraryInfo Info;
};
} // namespace luthier

// The function returns a struct with default initializers.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif
/// The public entry point for a legacy pass plugin.
///
/// This works similarly to an LLVM pass plugin. When a plugin is loaded by the
/// driver, it will call this entry point to obtain information about this
/// plugin and about how to register its passes. This function needs to be
/// implemented by the plugin.
///
extern "C" ::luthier::LegacyPassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
luthierGetLegacyPassPluginInfo();
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif /* LLVM_PLUGINS_PASSPLUGIN_H */
