//===-- IntrinsicProcessorRegistry.cpp ------------------------------------===//
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
/// Implements Luthier's Intrinsic Processor registry.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/IntrinsicProcessorRegistry.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Intrinsic/ImplicitArgPtr.h"
#include "luthier/Intrinsic/ReadReg.h"
#include "luthier/Intrinsic/ReadSVA.h"
#include "luthier/Intrinsic/SAtomicAdd.h"
#include "luthier/Intrinsic/WriteExec.h"
#include "luthier/Intrinsic/WriteReg.h"

namespace luthier {

namespace {
/// Spelled out so the IR-only registration macro stays readable.
using ArgsRef = llvm::ArrayRef<
    std::pair<llvm::InlineAsm::Flag, const llvm::MachineOperand *>>;
using MIBuilderFn = std::function<llvm::MachineInstrBuilder(int)>;
using VirtRegFn = std::function<llvm::Register(const llvm::TargetRegisterClass *)>;
} // namespace

IntrinsicProcessorRegistry::IntrinsicProcessorRegistry() {
  /// Register built-in Luthier intrinsics
#define REGISTER_INTRINSIC(NAME, IR_PROCESSOR, MIR_PROCESSOR)                  \
  Processors.try_emplace(NAME, IntrinsicProcessor{IR_PROCESSOR, MIR_PROCESSOR});
  /// The MIR slot of an IR-only intrinsic is never read -- \c
  /// IntrinsicMIRLoweringPass dispatches those by name. Rather than leave an
  /// empty \c std::function, which would throw if the assumption ever broke,
  /// install one that says so.
#define REGISTER_INTRINSIC_IR_ONLY(NAME, IR_PROCESSOR)                         \
  Processors.try_emplace(                                                      \
      NAME,                                                                    \
      IntrinsicProcessor{                                                      \
          IR_PROCESSOR,                                                        \
          [](const llvm::MachineFunction &, ArgsRef,                           \
             const MIBuilderFn &, const VirtRegFn &) -> llvm::Error {          \
            return LUTHIER_MAKE_GENERIC_ERROR(                                 \
                "The intrinsic " NAME " is lowered by a named special case in " \
                "IntrinsicMIRLoweringPass, so its registry MIR processor is "  \
                "not expected to be called.");                                 \
          }});
#include "luthier/Intrinsic/IntrinsicRegistry.def"
}

} // namespace luthier