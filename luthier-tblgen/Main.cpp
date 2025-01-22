//===-- Main.cpp - main function of luthier-tblgen ------------------------===//
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
/// This file contains the main function for the Luthier tablegen utility.
//===----------------------------------------------------------------------===//
#include "RealToPseudoOpcodeMapBackend.hpp"
#include "RealToPseudoRegisterMapBackend.hpp"
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/TableGen/Main.h>
#include <llvm/TableGen/TableGenBackend.h>

int main(int argc, char *argv[]) {
  llvm::InitLLVM Y(argc, argv);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::TableGen::Emitter::Opt RealToPseudoOpcodeOption(
      "gen-si-real-to-pseudo-opcode-map", luthier::emitRealToPseudoOpcodeTable,
      "Generate a Real to Pseudo Opcode map for the AMDGPU backend");

  llvm::TableGen::Emitter::Opt RealToPseudoRegisterOption(
      "gen-si-real-to-pseudo-reg-map", luthier::emitRealToPseudoRegisterTable,
      "Generate a Real to Pseudo Register enum map for the AMDGPU backend");
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return llvm::TableGenMain(argv[0]);
}
