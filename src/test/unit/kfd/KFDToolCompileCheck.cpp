//===-- KFDToolCompileCheck.cpp --------------------------------------------===//
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
/// Compiles a concrete \c KFDTool subclass, and nothing else.
///
/// \par Why this is a compile check rather than a test
/// \c KFDTool is a CRTP base composed of five traits, and almost everything that
/// can go wrong with such a class goes wrong at instantiation: a trait whose
/// constructor signature drifts, a \c Derived requirement that stops being
/// satisfied, a member that never gets instantiated because nothing calls it.
/// None of that is visible from the header alone -- a class template that no one
/// instantiates is only checked for syntax -- so without a subclass somewhere in
/// the build, \c KFDTool.h would be unverified.
///
/// It cannot be a linked test. \c ToolDeviceCodeOffloadParserTrait's static
/// fields are defined by \c LUTHIER_DEFINE_TOOL_OFFLOAD_PARSER_HANDLES, whose
/// \c __attribute__((managed)) requires the translation unit to be compiled as
/// HIP device code. So this target is compiled and never linked, which is
/// exactly the amount of checking the question needs.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/KFDTool.h"

namespace {

/// The smallest thing that is still a tool: it observes dispatches and injects
/// nothing.
class ReadOnlyKfdTool : public luthier::KFDTool<ReadOnlyKfdTool> {
public:
  using luthier::KFDTool<ReadOnlyKfdTool>::KFDTool;

  void onDispatchPacket(const luthier::kfd::QueueInfo &, uint64_t,
                        luthier::hsa::AqlPacket &) {}
};

} // namespace

/// Instantiate the members the instrumentation pipeline reaches a tool through.
/// Naming them explicitly matters: a member function template of a class
/// template is only instantiated when used, so a member that compiles nowhere
/// else compiles nowhere at all.
void luthierKfdToolCompileCheck(ReadOnlyKfdTool &T,
                                const llvm::amdhsa::kernel_descriptor_t &KD) {
  (void)T.createMemoryAllocationAccessor();
  llvm::consumeError(T.buildTargetMachineForKD(&KD).takeError());
  (void)T.getIntrinsicProcessorRegistry();

  // The two members that make instrumentation possible rather than just
  // analysis: bringing HSA up, and naming the device for the loader.
  llvm::consumeError(T.ensureHsaInitialized());
  llvm::consumeError(T.agentForCurrentDispatch().takeError());

  // The entry point a tool below the runtime actually uses. Instantiating it
  // here is what checks that code discovery composes with a driver-only
  // accessor and a queue-derived target machine -- the three pieces this whole
  // module exists to join.
  llvm::consumeError(T.runCodeDiscoveryForDispatch(
      KD, [](luthier::Prototype &, luthier::PrototypeAnalysisManager &,
             llvm::ModuleAnalysisManager &) {
        return llvm::Error::success();
      }));
}
