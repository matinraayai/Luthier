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
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Record.h>

namespace luthier {

unsigned
RealToPseudoOpcodeMapEmitter::emitBinSearchTable(llvm::raw_ostream &OS) {
  llvm::ArrayRef<const llvm::CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  llvm::StringRef Namespace = Target.getInstNamespace();
  llvm::outs() << "Number of pseudo insts: " << PseudoInsts.size() << "\n";
  OS << "static constexpr uint16_t RealToPseudoOpcodeMapTable[] {\n";
  for (const auto &NumberedInst : NumberedInstructions) {
    llvm::Record *SIInst = NumberedInst->TheDef;
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

void RealToPseudoOpcodeMapEmitter::emitBinSearch(llvm::raw_ostream &OS,
                                                 unsigned TableSize) {
  OS << llvm::formatv("  if (Opcode > {0})\n", TableSize);
  OS << "    return -1;\n";
  OS << "  else\n";
  OS << "    return RealToPseudoOpcodeMapTable[Opcode];\n";
}

void RealToPseudoOpcodeMapEmitter::emitMapFuncBody(llvm::raw_ostream &OS,
                                                   unsigned TableSize) {
  // Emit binary search algorithm to locate instructions in the
  // relation table. If found, return opcode value from the appropriate column
  // of the table.
  emitBinSearch(OS, TableSize);

  OS << "}\n\n";
}

void RealToPseudoOpcodeMapEmitter::emitTablesWithFunc(llvm::raw_ostream &OS) {
  OS << "LLVM_READONLY\n";
  OS << "uint16_t getPseudoOpcodeFromReal(uint16_t Opcode) {\n";

  // Emit map table.
  unsigned TableSize = emitBinSearchTable(OS);

  // Emit rest of the function body.
  emitMapFuncBody(OS, TableSize);
}

void EmitMapTable(llvm::RecordKeeper &Records, llvm::raw_ostream &OS) {
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
