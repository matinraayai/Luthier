//===-- object-test.cpp ---------------------------------------------------===//
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
/// This file implements object-test, an executable used for testing Luthier's
/// object library functionality using LLVM LIT.
//===----------------------------------------------------------------------===//
#include "SymbolLookupTest.hpp"
#include "TripleTest.hpp"
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/WithColor.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/LuthierError.h>
#include <luthier/object/ObjectUtils.h>

static llvm::cl::OptionCategory ObjectTestOptions("Object Test Options");

static llvm::cl::opt<std::string>
    InputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::cat(ObjectTestOptions));

static llvm::cl::opt<bool> TargetTripleTest(
    "triple-test",
    llvm::cl::desc("Test obtaining the target triple of the object file"),
    llvm::cl::init(false), llvm::cl::cat(ObjectTestOptions));

static llvm::cl::opt<bool>
    SymbolNameLookupTest("symbol-lookup-test",
                         llvm::cl::desc("Run tests for looking up symbols with "
                                        "their names (Applies to ELFs only)."),
                         llvm::cl::init(false),
                         llvm::cl::cat(ObjectTestOptions));

static llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(ObjectTestOptions));

int main(int Argc, char *Argv[]) {
  llvm::InitLLVM X(Argc, Argv);

  llvm::cl::HideUnrelatedOptions(
      {&ObjectTestOptions, &llvm::getColorCategory()});
  llvm::cl::ParseCommandLineOptions(Argc, Argv,
                                    "Luthier object library testing tool\n");

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferPtr =
      llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);

  std::error_code EC = BufferPtr.getError();
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC.operator bool(), "Failed to open input file, error: {0}.",
      EC.message()));

  llvm::StringRef Buffer = BufferPtr->get()->getBuffer();

  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> ObjectFileOrErr =
      luthier::object::createObjectFile(Buffer);
  LUTHIER_REPORT_FATAL_ON_ERROR(ObjectFileOrErr.takeError());

  const auto &ObjFile = **ObjectFileOrErr;

  auto OutFile = std::make_unique<llvm::ToolOutputFile>(OutputFilename, EC,
                                                        llvm::sys::fs::OF_None);
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !EC.operator bool(), "Failed to open output file, error: {0}.",
      EC.message()));

  // Print the target triple if enabled
  if (TargetTripleTest) {
    LUTHIER_REPORT_FATAL_ON_ERROR(
        performTargetTripleTest(ObjFile, OutFile->os()));
  }

  if (SymbolNameLookupTest) {
    symbolLookupTest(ObjFile, OutFile->os());
  }

  OutFile->keep();

  return 0;
}