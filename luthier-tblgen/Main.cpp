//===-- Main.cpp - main function of luthier-tblgen ------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility functions used to clone LLVM MIR constructs,
/// used frequently by Luthier components involved in the code generation
/// process. It is essentially a modified version of llvm-reduce.
//===----------------------------------------------------------------------===//
#include <llvm/Support/CommandLine.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/Record.h>

#include "RealToPseudoOpcodeMapBackend.hpp"

namespace {

llvm::cl::opt<bool> GenerateSiRealToPseudoOpcode(
    "gen-si-real-to-pseudo-opcode-map",
    llvm::cl::desc(
        "Generate a Real to Pseudo Opcode map for the AMDGPU backend"));

} // anonymous namespace

namespace luthier {

bool TableGenMain(llvm::raw_ostream &OS, llvm::RecordKeeper &Records) {
  if (GenerateSiRealToPseudoOpcode)
    EmitMapTable(Records, OS);
  return false;
}

} // namespace luthier

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &luthier::TableGenMain);
}
