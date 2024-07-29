//===-- RealToPseudoOpcodeMapBackend.hpp - Real to Pseudo Opcode Map ------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Contains definitions for the real to pseudo opcode tablegen backend
/// for the Luthier tablegen. It is inspired by the InstrMap class in tablegen,
/// except modified to handle cases where real opcodes don't have a pseudo
/// equivalent, or when the same real instructions has multiple pseudo opcodes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TBLGEN_REAL_TO_PSEUDO_OPCODE_MAP_BACKEND_HPP
#define LUTHIER_TBLGEN_REAL_TO_PSEUDO_OPCODE_MAP_BACKEND_HPP
#include "RealToPseudoOpcodeMapBackend.hpp"
#include <Common/CodeGenInstruction.h>
#include <Common/CodeGenTarget.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>

//===----------------------------------------------------------------------===//
// class MapTableEmitter : It builds the instruction relation maps using
// the information provided in InstrMapping records. It outputs these
// relationship maps as tables into XXXGenInstrInfo.inc file along with the
// functions to query them.

namespace luthier {
class MapTableEmitter {
private:
  //  std::string TargetName;
  const llvm::CodeGenTarget &Target;

  /// A mapping between the pseudo inst string opcode and the SI Pseudo
  /// instruction
  llvm::StringMap<llvm::Record *> PseudoInsts;

  /// Number of
  unsigned int NumRealSIInsts{0};

public:
  MapTableEmitter(llvm::CodeGenTarget &Target, llvm::RecordKeeper &Records)
      : Target(Target) {
    auto SIInsts = Records.getAllDerivedDefinitions("SIMCInstr");
    for (auto SIInst : SIInsts) {
      // Find all pseudo SI instructions
      bool IsPseudo =
          SIInst->getValue("isPseudo")->getValue()->getAsUnquotedString() ==
          "1";
      if (IsPseudo) {
        PseudoInsts.insert(
            {SIInst->getValueAsString("PseudoInstr"), SIInst});
      } else {
        NumRealSIInsts++;
      }
    }
  }

  void emitBinSearch(llvm::raw_ostream &OS, unsigned TableSize);
  void emitTablesWithFunc(llvm::raw_ostream &OS);
  unsigned emitBinSearchTable(llvm::raw_ostream &OS);

  // Lookup functions to query binary search tables.
  void emitMapFuncBody(llvm::raw_ostream &OS, unsigned TableSize);
};

void EmitMapTable(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
} // namespace luthier

#endif