//===-- LuthierPassPlugin.h -------------------------------------*- C++ -*-===//
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
/// \file
/// Implements Luthier's pass manager plugin. Similar to LLVM plugins, Luthier
/// pass plugins can be used to augment Luthier's instrumentation process.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_PASS_PLUGINS_PASS_PLUGIN_H
#define LUTHIER_PASS_PLUGINS_PASS_PLUGIN_H
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <string>
#include <utility>

namespace llvm {
class Module;

class PassBuilder;

class TargetMachine;

class TargetPassConfig;

class PassRegistry;

class LLVMContext;

class Triple;

class SubtargetFeatures;
} // namespace llvm

namespace luthier {

class InstrumentationPassBuilder;

extern "C" {

/// \brief Contains information provided by a Luthier pass plugin
/// \details This struct defines the core interface for pass plugins and is
/// supposed to be filled out by plugin implementors. Unused function pointers
/// and data pointers can be set to nullptr.
/// For now there are no plans for this struct to be "forward-compatible" (
/// similar to API tables in rocprofiler-sdk). In the future versions, these
/// entries might be re-ordered, which will be indicated by incrementing the
/// \c APIVersion field.
struct PassPluginLibraryInfo {
  /// The API version understood by this plugin, usually \c
  /// LUTHIER_PASS_PLUGIN_VERSION
  uint32_t APIVersion{};
  /// Name of the plugin
  const char *PluginName{};
  /// The version of the plugin
  const char *PluginVersion{};
  /// Extra arguments passed to all callbacks set by the plugin
  void *ExtraArgs{nullptr};
  /// The callback for augmenting the \c InstrumentationPassBuilder.
  void (*RegisterPrototypePassBuilderCallback)(InstrumentationPassBuilder &,
                                               void *){nullptr};
};

/// \macro LUTHIER_PASS_PLUGIN_API_VERSION
/// Tracks the API compatibility of the supported plugin version
///
/// When a plugin is loaded, the driver will check it's supported plugin version
/// against that of the plugin. A mismatch is an error. The supported version
/// will be incremented for ABI-breaking changes to the \c PassPluginLibraryInfo
/// struct, i.e. when callbacks are added, removed, or reordered.
#define LUTHIER_PASS_PLUGIN_API_VERSION 2
}

/// A loaded pass plugin.
///
/// An instance of this class wraps a loaded pass plugin and gives access to
/// its interface defined by the \c PassPluginLibraryInfo it exposes
class PassPlugin {
public:
  /// Attempts to load a pass plugin from a given file
  ///
  /// \returns Returns an error if either the library cannot be found or loaded,
  /// there is no public entry point, or the plugin implements the wrong API
  /// version
  LLVM_ABI static llvm::Expected<PassPlugin> Load(const std::string &Filename);

  /// Get the filename of the loaded plugin
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

  /// Invoke the callback for augmenting the \c InstrumentationPassBuilder
  /// used by the \c luthier-llc driver.
  void
  registerPrototypePassBuilderCallback(InstrumentationPassBuilder &PPB) const {
    if (Info.RegisterPrototypePassBuilderCallback)
      Info.RegisterPrototypePassBuilderCallback(PPB, Info.ExtraArgs);
  }

private:
  PassPlugin(std::string Filename, const llvm::sys::DynamicLibrary &Library)
      : Filename(std::move(Filename)), Library(Library), Info() {}

  std::string Filename;

  llvm::sys::DynamicLibrary Library;

  PassPluginLibraryInfo Info;
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
extern "C" ::luthier::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
luthierGetPassPluginInfo();
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
