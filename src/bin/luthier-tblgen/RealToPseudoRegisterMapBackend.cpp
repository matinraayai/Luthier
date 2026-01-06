//===- RealToPseudoRegisterMapBackend.cpp - Real To Pseudo Register Map  --===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Contains implementation for the real to pseudo register tablegen backend
/// for the Luthier tablegen.
//===----------------------------------------------------------------------===//
#include "RealToPseudoRegisterMapBackend.hpp"
#include <Common/CodeGenRegisters.h>
#include <Common/CodeGenTarget.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Record.h>

namespace luthier {

unsigned RealToPseudoRegisterMapEmitter::emitTable(llvm::raw_ostream &OS) {
  auto &NumberedRegisters = Target.getRegBank().getRegisters();
  llvm::StringRef Namespace = Target.getRegNamespace();
  OS << "static constexpr unsigned short RealToPseudoRegisterMapTable[] {\n0,\n";
  for (const auto &NumberedReg : NumberedRegisters) {
    const llvm::Record *SIReg = NumberedReg.TheDef;
    llvm::StringRef RegName = SIReg->getName();
    std::string PseudoRegName(RegName);

    for (auto RealRegSuffix :
         {"_gfx9plus", "_vi", "_ci", "_gfx11plus", "_gfxpre11"}) {
      auto PrefixPos = PseudoRegName.find(RealRegSuffix);
      if (PrefixPos != std::string::npos) {
        PseudoRegName.erase(PrefixPos, strlen(RealRegSuffix));
      }
    }
    OS << "llvm::" << Namespace << "::" << PseudoRegName << ", \n";
  }
  OS << "}; // End of Table\n\n";
  return NumberedRegisters.size();
}

void RealToPseudoRegisterMapEmitter::emitIndexing(llvm::raw_ostream &OS,
                                                  unsigned TableSize) {
  OS << llvm::formatv("  if (RegNum > {0})\n", TableSize);
  OS << "    return -1;\n";
  OS << "  else\n";
  OS << "    return RealToPseudoRegisterMapTable[RegNum];\n";
}

void RealToPseudoRegisterMapEmitter::emitMapFuncBody(llvm::raw_ostream &OS,
                                                     unsigned TableSize) {
  // Emit binary search algorithm to locate instructions in the
  // relation table. If found, return opcode value from the appropriate column
  // of the table.
  emitIndexing(OS, TableSize);

  OS << "}\n\n";
}

void RealToPseudoRegisterMapEmitter::emitTablesWithFunc(llvm::raw_ostream &OS) {
  OS << "LLVM_READONLY\n";
  OS << "unsigned short RealToPseudoRegisterMapTable(unsigned short RegNum) "
        "{\n";

  // Emit map table.
  unsigned TableSize = emitTable(OS);

  // Emit rest of the function body.
  emitMapFuncBody(OS, TableSize);
}

void emitRealToPseudoRegisterTable(const llvm::RecordKeeper &Records,
                                   llvm::raw_ostream &OS) {
  llvm::CodeGenTarget Target(Records);

  RealToPseudoRegisterMapEmitter IMap(Target);

  // Emit map tables and the functions to query them.
  IMap.emitTablesWithFunc(OS);
}

} // namespace luthier
