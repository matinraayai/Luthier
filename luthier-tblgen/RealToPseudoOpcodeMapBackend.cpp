//===- RealToPseudoOpcodeMapBackend.cpp - Real To Pseudo Opcode Map  ------===//
//
//===----------------------------------------------------------------------===//

#include "RealToPseudoOpcodeMapBackend.hpp"
#include <Common/CodeGenInstruction.h>
#include <Common/CodeGenTarget.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

namespace luthier {

//===----------------------------------------------------------------------===//
// Emit one table per relation. Only instructions with a valid relation of a
// given type are included in the table sorted by their enum values (opcodes).
// Binary search is used for locating instructions in the table.
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Emit binary search algorithm as part of the functions used to query
// relation tables.
//===----------------------------------------------------------------------===//

void RealToPseudoOpcodeMapEmitter::emitBinSearch(llvm::raw_ostream &OS, unsigned TableSize) {
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

//===----------------------------------------------------------------------===//
// Emit functions to query relation tables.
//===----------------------------------------------------------------------===//

void RealToPseudoOpcodeMapEmitter::emitMapFuncBody(llvm::raw_ostream &OS,
                                      unsigned TableSize) {
  // Emit binary search algorithm to locate instructions in the
  // relation table. If found, return opcode value from the appropriate column
  // of the table.
  emitBinSearch(OS, TableSize);

  OS << "  return RealToPseudoOpcodeMapTable[mid][1];\n";

  OS << "}\n\n";
}

//===----------------------------------------------------------------------===//
// Emit relation tables and the functions to query them.
//===----------------------------------------------------------------------===//

void RealToPseudoOpcodeMapEmitter::emitTablesWithFunc(llvm::raw_ostream &OS) {

  // Emit function name and the input parameters : mostly opcode value of the
  // current instruction. However, if a table has multiple columns (more than 2
  // since first column is used for the key instructions), then we also need
  // to pass another input to indicate the column to be selected.

  OS << "LLVM_READONLY\n";
  OS << "uint16_t getPseudoOpcodeFromReal (uint16_t Opcode) {\n";

  // Emit map table.
  unsigned TableSize = emitBinSearchTable(OS);

  // Emit rest of the function body.
  emitMapFuncBody(OS, TableSize);
}

//===----------------------------------------------------------------------===//
// Parse 'InstrMapping' records and use the information to form relationship
// between instructions. These relations are emitted as a tables along with the
// functions to query them.
//===----------------------------------------------------------------------===//
void EmitMapTable(llvm::RecordKeeper &Records, llvm::raw_ostream &OS) {
  llvm::CodeGenTarget Target(Records);

  OS << "#ifdef GET_INSTRMAP_INFO\n";
  OS << "#undef GET_INSTRMAP_INFO\n";
  OS << "namespace luthier {\n\n";

  // Iterate over all instruction mapping records and construct relationship
  // maps based on the information specified there.
  //
  RealToPseudoOpcodeMapEmitter IMap(Target, Records);

  // Emit map tables and the functions to query them.
  IMap.emitTablesWithFunc(OS);
  OS << "} // end namespace luthier\n";
  OS << "#endif // GET_INSTRMAP_INFO\n\n";
}

} // namespace luthier
