//===-- Linker.cpp - LLD Linker Wrapper -----------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file Linker.cpp
/// Implements linking of relocatable AMDGPU ELF objects into executable shared
/// objects using LLD.
//===----------------------------------------------------------------------===//
#include "luthier/Linker/Linker.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include <lld/Common/CommonLinkerContext.h>
#include <lld/Common/Driver.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/Support/raw_ostream.h>

LLD_HAS_DRIVER(elf)

namespace luthier::linker {

llvm::Error linkRelocatableToExecutable(llvm::ArrayRef<char> Code,
                                        llvm::SmallVectorImpl<char> &Out) {
  llvm::TimeTraceScope Scope("LLD Executable Linking");

  // Create a temporary directory for input/output files
  llvm::SmallString<128> TmpDir;
  if (auto EC = llvm::sys::fs::createUniqueDirectory("luthier-link", TmpDir))
    return LUTHIER_MAKE_ERROR(
        GenericLuthierError,
        llvm::formatv("Failed to create temporary directory: {0}",
                      EC.message()));

  // RAII cleanup of the temp directory
  auto CleanupTmpDir = llvm::scope_exit(
      [&TmpDir]() { (void)llvm::sys::fs::remove_directories(TmpDir); });

  // Write the relocatable input to a temp file
  llvm::SmallString<128> InputPath(TmpDir);
  llvm::sys::path::append(InputPath, "input.o");
  {
    std::error_code EC;
    llvm::raw_fd_ostream InputFile(InputPath, EC);
    if (EC)
      return LUTHIER_MAKE_ERROR(
          GenericLuthierError,
          llvm::formatv("Failed to write relocatable to temp file: {0}",
                        EC.message()));
    InputFile.write(Code.data(), Code.size());
  }

  // Set up the output path
  llvm::SmallString<128> OutputPath(TmpDir);
  llvm::sys::path::append(OutputPath, "output.so");

  // Construct LLD arguments
  const char *LLDArgs[] = {"ld.lld",
                           "-shared",
                           "--threads=1",
                           "--unresolved-symbols=ignore-all",
                           "-o",
                           OutputPath.c_str(),
                           InputPath.c_str()};

  // Capture LLD's stdout/stderr for diagnostics
  std::string StdoutStr, StderrStr;
  llvm::raw_string_ostream StdoutOS(StdoutStr), StderrOS(StderrStr);

  // Invoke LLD
  lld::Result LLDResult =
      lld::lldMain(LLDArgs, StdoutOS, StderrOS, {{lld::Gnu, &lld::elf::link}});
  lld::CommonLinkerContext::destroy();

  if (LLDResult.retCode != 0)
    return LUTHIER_MAKE_ERROR(
        GenericLuthierError,
        llvm::formatv("LLD linking failed (exit code {0}): {1}",
                      LLDResult.retCode, StderrStr));

  // TODO: Add verbose logging of LLD's output

  // Read the linked executable back into memory
  auto OutputBufOrErr = llvm::MemoryBuffer::getFile(OutputPath);
  if (!OutputBufOrErr)
    return LUTHIER_MAKE_ERROR(
        GenericLuthierError,
        llvm::formatv("Failed to read linked executable: {0}",
                      OutputBufOrErr.getError().message()));

  llvm::StringRef OutputData = (*OutputBufOrErr)->getBuffer();
  Out.assign(OutputData.begin(), OutputData.end());
  return llvm::Error::success();
}

} // namespace luthier::linker
