//===-- Main.cpp - main function of luthier-tblgen ------------------------===//
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
/// This file contains the main function for the Luthier tablegen utility
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
