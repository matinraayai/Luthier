//===- RealToPseudoOpcodeMapBackend.cpp - Real To Pseudo Opcode Map  ------===//
//
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
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

namespace luthier {

unsigned
RealToPseudoOpcodeMapEmitter::emitBinSearchTable(llvm::raw_ostream &OS) {
  llvm::ArrayRef<const llvm::CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  unsigned TotalNumInstr = NumberedInstructions.size();
  unsigned TableSize = 0;

  llvm::StringRef Namespace = Target.getInstNamespace();
  llvm::outs() << "Number of pseudo insts: " << PseudoInsts.size() << "\n";
  OS << "static const uint16_t RealToPseudoOpcodeMapTable[][2] = {\n";
  for (const auto &NumberedInst : NumberedInstructions) {
    llvm::Record *SIInst = NumberedInst->TheDef;
    bool IsReal =
        SIInst->getValue("isPseudo")->getValue()->getAsUnquotedString() == "0";
    if (IsReal) {
      llvm::StringRef RealInstName = SIInst->getName();
      llvm::StringRef PseudoInstName;
      auto *PseudoInstrRecord = SIInst->getValue("PseudoInstr");
      // If there is a pseudo instr string record for this real inst, query
      // it from the map
      if (PseudoInstrRecord) {
        auto PseudoSIInstIt = PseudoInsts.find(
            PseudoInstrRecord->getValue()->getAsUnquotedString());
        // If the pseudo instr record was not in the map, return the real instr
        // name; Otherwise, assign the map value
        if (PseudoSIInstIt == PseudoInsts.end())
          PseudoInstName = RealInstName;
        else
          PseudoInstName = PseudoSIInstIt->second->getName();
        TableSize++;
      } // This inst doesn't have a pseudo inst record, assign it its real instr
      // name
      else {
        PseudoInstName = RealInstName;
      }
      OS << "{ llvm::" << Namespace << "::" << RealInstName
         << ", llvm::" << Namespace << "::" << PseudoInstName << "}, \n";
    }
  }
  OS << "}; // End of Table\n\n";
  return TableSize;
}

void RealToPseudoOpcodeMapEmitter::emitBinSearch(llvm::raw_ostream &OS,
                                                 unsigned TableSize) {
  OS << "  unsigned mid;\n";
  OS << "  unsigned start = 0;\n";
  OS << "  unsigned end = " << TableSize << ";\n";
  OS << "  while (start < end) {\n";
  OS << "    mid = start + (end - start) / 2;\n";
  OS << "    if (Opcode == RealToPseudoOpcodeMapTable[mid][0]) {\n";
  OS << "      break;\n";
  OS << "    }\n";
  OS << "    if (Opcode < RealToPseudoOpcodeMapTable[mid][0])\n";
  OS << "      end = mid;\n";
  OS << "    else\n";
  OS << "      start = mid + 1;\n";
  OS << "  }\n";
  OS << "  if (start == end)\n";
  OS << "    return -1; // Instruction doesn't exist in this table.\n\n";
}

void RealToPseudoOpcodeMapEmitter::emitMapFuncBody(llvm::raw_ostream &OS,
                                                   unsigned TableSize) {
  // Emit binary search algorithm to locate instructions in the
  // relation table. If found, return opcode value from the appropriate column
  // of the table.
  emitBinSearch(OS, TableSize);

  OS << "  return RealToPseudoOpcodeMapTable[mid][1];\n";

  OS << "}\n\n";
}

void RealToPseudoOpcodeMapEmitter::emitTablesWithFunc(llvm::raw_ostream &OS) {
  OS << "LLVM_READONLY\n";
  OS << "uint16_t getPseudoOpcodeFromReal (uint16_t Opcode) {\n";

  // Emit map table.
  unsigned TableSize = emitBinSearchTable(OS);

  // Emit rest of the function body.
  emitMapFuncBody(OS, TableSize);
}

void EmitMapTable(llvm::RecordKeeper &Records, llvm::raw_ostream &OS) {
  llvm::CodeGenTarget Target(Records);

  OS << "#ifdef GET_INSTRMAP_INFO\n";
  OS << "#undef GET_INSTRMAP_INFO\n";
  OS << "namespace luthier {\n\n";

  RealToPseudoOpcodeMapEmitter IMap(Target, Records);

  // Emit map tables and the functions to query them.
  IMap.emitTablesWithFunc(OS);
  OS << "} // end namespace luthier\n";
  OS << "#endif // GET_INSTRMAP_INFO\n\n";
}

} // namespace luthier
