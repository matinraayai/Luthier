//===-- comgr-link.cpp ----------------------------------------------------===//
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
/// This file implements comgr-link, an executable used to test the relocatable
/// linking functionality of Comgr used in Luthier.
//===----------------------------------------------------------------------===//
#include "luthier/Comgr/Comgr.h"
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/WithColor.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/LuthierError.h>

static llvm::cl::OptionCategory ComgrLinkOptions("Comgr Link Options");

static llvm::cl::opt<std::string>
    InputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::cat(ComgrLinkOptions));

static llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(ComgrLinkOptions));

int main(int Argc, char *Argv[]) {
  llvm::InitLLVM X(Argc, Argv);

  llvm::cl::HideUnrelatedOptions(
      {&ComgrLinkOptions, &llvm::getColorCategory()});
  llvm::cl::ParseCommandLineOptions(Argc, Argv,
                                    "Luthier link with comgr tool\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferPtr =
      llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);

  std::error_code EC = BufferPtr.getError();
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC.operator bool(), "Failed to open input file, error: {0}.",
      EC.message()));

  llvm::StringRef Buffer = BufferPtr->get()->getBuffer();

  llvm::SmallVector<char> Executable;
  LUTHIER_REPORT_FATAL_ON_ERROR(luthier::comgr::linkRelocatableToExecutable(
      llvm::arrayRefFromStringRef<char>(Buffer), Executable));

  auto OutFile = std::make_unique<llvm::ToolOutputFile>(OutputFilename, EC,
                                                        llvm::sys::fs::OF_None);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC.operator bool(), "Failed to open output file, error: {0}.",
      EC.message()));
  OutFile->os() << Executable;

  OutFile->keep();

  return 0;
}