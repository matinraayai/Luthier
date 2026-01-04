//===-- LuthierPassPlugin.h -------------------------------------*- C++ -*-===//
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
/// \file LuthierPassPlugin.h
/// Implements Luthier's pass manager plugin. Similar to LLVM plugins, Luthier
/// pass plugins can be used to augment the normal instrumentation process
/// defined in the \c InstrumentationPMDriver target module pass.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_PLUGINS_PASS_PLUGIN_H
#define LUTHIER_PLUGINS_PASS_PLUGIN_H
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

class LLVMContext;

class Triple;

class SubtargetFeatures;

class ModulePassManager;
} // namespace llvm

namespace luthier {

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
  /// The callback used to create the instrumentation module; If there are
  /// multiple plugins their created modules will be linked together
  std::unique_ptr<llvm::Module> (*IModuleCreationCallback)(llvm::LLVMContext &,
                                                           const llvm::Triple &,
                                                           llvm::StringRef,
                                                           llvm::StringRef &,
                                                           void *){nullptr};

  /// The callback for augmenting the instrumentation pass builder
  /// This can be used to add any additional analysis required by the plugin,
  /// and to also modify the "default" part of the IR optimization pipeline as
  /// documented by the LLVM pass plugin and the \c llvm::PassBuilder
  void (*RegisterInstrumentationPassBuilderCallback)(llvm::PassBuilder &,
                                                     void *){nullptr};
  /// The callback for adding passes to the instrumentation pass manager before
  /// the Luthier IR lowering pass
  /// Use this to register the "primary" instrumentation logic creation passes
  void (*PreLuthierIRIntrinsicLoweringPassesCallback)(llvm::ModulePassManager &,
                                                      void *){nullptr};

  /// The callback for adding passes to the instrumentation pass manager after
  /// the Luthier IR lowering pass
  void (*PostLuthierIRIntrinsicLoweringPassesCallback)(
      llvm::ModulePassManager &, void *){nullptr};

  /// The callback for registering the plugin's instrumentation legacy codegen
  /// passes with the legacy pass registry
  /// It is the plugin's responsibility to ensure each pass is registered
  /// only once
  void (*RegisterLegacyCodegenPassesCallback)(llvm::PassRegistry &,
                                              void *){nullptr};
  /// The callback for augmenting the legacy pass manager of the instrumentation
  /// codegen pipeline via the target pass config
  /// Use the passed \c llvm::PassRegistry to lookup information regarding each
  /// registered pass
  void (*AugmentTargetPassConfigCallback)(llvm::PassRegistry &,
                                          llvm::TargetPassConfig &,
                                          llvm::TargetMachine &,
                                          void *){nullptr};
};

/// \macro LUTHIER_PASS_PLUGIN_API_VERSION
/// Tracks the API compatibility of the supported plugin version
///
/// When a plugin is loaded, the driver will check it's supported plugin version
/// against that of the plugin. A mismatch is an error. The supported version
/// will be incremented for ABI-breaking changes to the \c PassPluginLibraryInfo
/// struct, i.e. when callbacks are added, removed, or reordered.
#define LUTHIER_PASS_PLUGIN_API_VERSION 1
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

  /// Invoke the instrumentation module creation callback
  std::unique_ptr<llvm::Module> instrumentationModuleCreationCallback(
      llvm::LLVMContext &Context, const llvm::Triple &TT,
      llvm::StringRef CPUName, llvm::StringRef FS) const;

  /// Invoke the pass builder callback
  void registerInstrumentationPassBuilderCallback(llvm::PassBuilder &PB) const {
    if (Info.RegisterInstrumentationPassBuilderCallback)
      return Info.RegisterInstrumentationPassBuilderCallback(PB,
                                                             Info.ExtraArgs);
  }

  /// Invoke the callback before adding the Luthier intrinsic IR lowering pass
  /// to \p MPM
  void invokePreLuthierIRIntrinsicLoweringPassesCallback(
      llvm::ModulePassManager &MPM) const {
    if (Info.PreLuthierIRIntrinsicLoweringPassesCallback) {
      return Info.PreLuthierIRIntrinsicLoweringPassesCallback(MPM,
                                                              Info.ExtraArgs);
    }
  }

  /// Invoke the callback for adding passes to the instrumentation pass manager
  /// after the Luthier IR lowering pass
  void invokePostLuthierIRIntrinsicLoweringPassesCallback(
      llvm::ModulePassManager &MPM) const {
    if (Info.PostLuthierIRIntrinsicLoweringPassesCallback) {
      return Info.PostLuthierIRIntrinsicLoweringPassesCallback(MPM,
                                                               Info.ExtraArgs);
    }
  }

  /// Invoke the callback for registration codegen passes to the driver's
  /// pass registry
  void registerLegacyCodegenPassesCallback(llvm::PassRegistry &PR) const {
    if (Info.RegisterLegacyCodegenPassesCallback)
      Info.RegisterLegacyCodegenPassesCallback(PR, Info.ExtraArgs);
  }

  /// Invoke the callback for modifying the instrumentation legacy codegen pass
  /// pipeline
  void invokeAugmentTargetPassConfigCallback(llvm::PassRegistry &PR,
                                             llvm::TargetPassConfig &TPC,
                                             llvm::TargetMachine &TM) const {
    if (Info.AugmentTargetPassConfigCallback)
      Info.AugmentTargetPassConfigCallback(PR, TPC, TM, Info.ExtraArgs);
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
luthierGetLegacyPassPluginInfo();
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
