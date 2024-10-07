//===- RealToPseudoOpcodeMapBackend.cpp - Real To Pseudo Opcode Map  ------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// Contains implementation for the real to pseudo opcode tablegen backend
/// for the Luthier tablegen. It is inspired by the InstrMap class in tablegen,
/// except modified to handle cases where real opcodes don't have a pseudo
/// equivalent, or when the same real instructions has multiple pseudo opcodes.
//===----------------------------------------------------------------------===//

#include "RealToPseudoOpcodeMapBackend.hpp"
#include <Common/CodeGenInstruction.h>
#include <Common/CodeGenTarget.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Record.h>

namespace luthier {

unsigned
RealToPseudoOpcodeMapEmitter::emitTable(llvm::raw_ostream &OS) {
  llvm::ArrayRef<const llvm::CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  llvm::StringRef Namespace = Target.getInstNamespace();
  OS << "static constexpr uint16_t RealToPseudoOpcodeMapTable[] {\n";
  for (const auto &NumberedInst : NumberedInstructions) {
    const llvm::Record *SIInst = NumberedInst->TheDef;
    llvm::StringRef PseudoInstName;
    bool IsReal =
        SIInst->getValue("isPseudo")->getValue()->getAsUnquotedString() == "0";
    llvm::StringRef InstName = SIInst->getName();
    if (IsReal) {
      auto *PseudoInstrRecord = SIInst->getValue("PseudoInstr");
      // If there is a pseudo instr string record for this real inst, query
      // it from the map
      if (PseudoInstrRecord) {
        auto PseudoSIInstIt = PseudoInsts.find(
            PseudoInstrRecord->getValue()->getAsUnquotedString());
        // If the pseudo instr record was not in the map, return the real instr
        // name; Otherwise, assign the map value
        if (PseudoSIInstIt == PseudoInsts.end())
          PseudoInstName = InstName;
        else
          PseudoInstName = PseudoSIInstIt->second->getName();
      } // This inst doesn't have a pseudo inst record, assign it its real instr
      // name
      else {
        PseudoInstName = InstName;
      }
    } else {
      PseudoInstName = InstName;
    }
    OS << "llvm::" << Namespace << "::" << PseudoInstName << ", \n";
  }
  OS << "}; // End of Table\n\n";
  return NumberedInstructions.size();
}

void RealToPseudoOpcodeMapEmitter::emitIndexing(llvm::raw_ostream &OS,
                                                 unsigned TableSize) {
  OS << llvm::formatv("  if (Opcode > {0})\n", TableSize);
  OS << "    return -1;\n";
  OS << "  else\n";
  OS << "    return RealToPseudoOpcodeMapTable[Opcode];\n";
}

void RealToPseudoOpcodeMapEmitter::emitMapFuncBody(llvm::raw_ostream &OS,
                                                   unsigned TableSize) {
  emitIndexing(OS, TableSize);
  OS << "}\n\n";
}

void RealToPseudoOpcodeMapEmitter::emitTablesWithFunc(llvm::raw_ostream &OS) {
  OS << "LLVM_READONLY\n";
  OS << "uint16_t getPseudoOpcodeFromReal(uint16_t Opcode) {\n";

  // Emit map table.
  unsigned TableSize = emitTable(OS);

  // Emit rest of the function body.
  emitMapFuncBody(OS, TableSize);
}

void emitRealToPseudoOpcodeTable(llvm::RecordKeeper &Records, llvm::raw_ostream &OS) {
  llvm::CodeGenTarget Target(Records);
  OS << "#ifndef GET_REAL_TO_PSEUDO_OPCODE_MAP\n";
  OS << "#define GET_REAL_TO_PSEUDO_OPCODE_MAP\n";
  OS << "namespace luthier {\n\n";

  RealToPseudoOpcodeMapEmitter IMap(Target, Records);

  // Emit map tables and the functions to query them.
  IMap.emitTablesWithFunc(OS);
  OS << "} // end namespace luthier\n";
  OS << "#endif // GET_REAL_TO_PSEUDO_OPCODE_MAP\n\n";
}

} // namespace luthier
