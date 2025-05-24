//===-- InstrumentationModule.cpp - Luthier Instrumentation Module --------===//
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
/// This file implements Luthier's Instrumentation Module.
//===----------------------------------------------------------------------===//
#include "luthier/tooling/InstrumentationModule.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include "luthier/consts.h"

namespace {

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
getBCBufferInObjectFile(const llvm::object::ObjectFile &ObjFile) {
  // Find the Luthier bitcode section of the ELF
  for (const llvm::object::SectionRef &Section : ObjFile.sections()) {
    auto SectionName = Section.getName();
    LUTHIER_RETURN_ON_ERROR(SectionName.takeError());
    if (*SectionName == luthier::IModuleBCSectionName) {
      auto SectionContents = Section.getContents();
      LUTHIER_RETURN_ON_ERROR(SectionContents.takeError());
      return llvm::MemoryBuffer::getMemBuffer(*SectionContents, "", false);
    }
  }
  return nullptr;
}

} // namespace

namespace luthier {

llvm::Expected<std::unique_ptr<InstrumentationModule>>
InstrumentationModule::create(std::vector<uint8_t> CodeObject) {

  std::unique_ptr<llvm::object::ObjectFile> ObjFile;
  LUTHIER_RETURN_ON_ERROR(
      luthier::object::createObjectFile(llvm::toStringRef(CodeObject))
          .moveInto(ObjFile));

  // Confirm that the ELF contains a Luthier bitcode section
  std::unique_ptr<llvm::MemoryBuffer> BCBuffer;
  LUTHIER_RETURN_ON_ERROR(getBCBufferInObjectFile(*ObjFile).moveInto(BCBuffer));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      BCBuffer != nullptr, "Failed to find the Luthier bitcode section inside "
                           "the instrumentation module"));

  return std::make_unique<InstrumentationModule>(
      std::move(CodeObject), std::move(ObjFile), std::move(BCBuffer));
}

} // namespace luthier