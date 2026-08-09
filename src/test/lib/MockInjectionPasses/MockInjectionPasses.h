//===-- MockInjectionPasses.h - test-only payload-injection passes -*-C++-*===//
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
/// Test-only mock instrumentation passes deriving from
/// \c InjectedPayloadCreationPass. Each one is a \c luthier::Prototype -level
/// pass: it walks the target <tt>MachineFunction</tt>s of the prototype, picks
/// a deterministic set of <tt>MachineInstr</tt>s out of them, and creates an
/// injected-payload function in the prototype's instrumentation module that
/// calls a configured hook for each one.
///
/// Hook lookup: each pass reads the mangled name of the hook function from a
/// global \c cl::opt (default: \c "_Z11bumpCounterv") and looks that name up
/// in the prototype's instrumentation module. A pass whose hook is absent from
/// the IModule is a no-op.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TEST_MOCK_INJECTION_PASSES_H
#define LUTHIER_TEST_MOCK_INJECTION_PASSES_H
#include "luthier/ToolCodeGen/InjectedPayloadCreationPass.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

namespace luthier::test {

llvm::StringRef getMockHookNameOpt();
llvm::StringRef getMockOpcodeMnemonicOpt();

//===----------------------------------------------------------------------===//
// At-function-entry: inject before the first MI of the entry MBB
//===----------------------------------------------------------------------===//

class MockInjectAtFunctionEntryPass
    : public luthier::InjectedPayloadCreationPass<
          MockInjectAtFunctionEntryPass> {
public:
  static llvm::StringRef name() {
    return "luthier-mock-inject-at-function-entry";
  }

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// At-MBB-entry: inject before the first MI of every MBB
//===----------------------------------------------------------------------===//

class MockInjectAtMBBEntryPass
    : public luthier::InjectedPayloadCreationPass<MockInjectAtMBBEntryPass> {
public:
  static llvm::StringRef name() { return "luthier-mock-inject-at-mbb-entry"; }

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// At-MBB-terminator: inject before each MBB's terminator
//===----------------------------------------------------------------------===//

class MockInjectAtMBBTerminatorPass
    : public luthier::InjectedPayloadCreationPass<
          MockInjectAtMBBTerminatorPass> {
public:
  static llvm::StringRef name() {
    return "luthier-mock-inject-at-mbb-terminator";
  }

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// At-all-VALU: inject before every vector ALU MI
//===----------------------------------------------------------------------===//

class MockInjectAtAllVALUPass
    : public luthier::InjectedPayloadCreationPass<MockInjectAtAllVALUPass> {
public:
  static llvm::StringRef name() { return "luthier-mock-inject-at-all-valu"; }

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// At-all-scalar: inject before every scalar ALU MI
//===----------------------------------------------------------------------===//

class MockInjectAtAllScalarPass
    : public luthier::InjectedPayloadCreationPass<MockInjectAtAllScalarPass> {
public:
  static llvm::StringRef name() { return "luthier-mock-inject-at-all-scalar"; }

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// At-opcode: inject before every MI whose mnemonic matches a cl::opt-supplied
// string (case-sensitive substring match).
//===----------------------------------------------------------------------===//

class MockInjectAtOpcodePass
    : public luthier::InjectedPayloadCreationPass<MockInjectAtOpcodePass> {
public:
  static llvm::StringRef name() { return "luthier-mock-inject-at-opcode"; }

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);
};

//===----------------------------------------------------------------------===//
// At-all-VGPR-defs-with-regarg: for every MI that defines a VGPR, inject a
// payload that forwards the first defined VGPR as a uint32_t RegArg, to
// exercise the luthier::readReg intrinsic call lowering.
//===----------------------------------------------------------------------===//

class MockInjectAtAllVGPRDefsWithRegArgPass
    : public luthier::InjectedPayloadCreationPass<
          MockInjectAtAllVGPRDefsWithRegArgPass> {
public:
  static llvm::StringRef name() {
    return "luthier-mock-inject-at-all-vgpr-defs-with-regarg";
  }

  llvm::PreservedAnalyses run(Prototype &P, PrototypeAnalysisManager &PAM);
};

} // namespace luthier::test

#endif
