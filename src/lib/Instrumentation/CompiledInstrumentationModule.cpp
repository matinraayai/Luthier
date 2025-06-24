//===-- CompiledInstrumentationModule.cpp ---------------------------------===//
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
/// Implements the \c CompiledInstrumentationModule class.
//===----------------------------------------------------------------------===//
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/Instrumentation/CompiledInstrumentationModule.h>
#include <luthier/Instrumentation/consts.h>

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

llvm::Expected<bool>
isInstrumentationModule(const llvm::object::ObjectFile &ObjFile) {
  // Check for the presence of the Luthier reserved managed variable
  std::optional<llvm::object::ELFSymbolRef>
      LuthierReservedManagedVariableIfPresent{};
  LUTHIER_RETURN_ON_ERROR(
      luthier::object::lookupSymbolByName(
          llvm::cast<llvm::object::ELF64LEObjectFile>(ObjFile),
          luthier::IModuleReservedManagedVar)
          .moveInto(LuthierReservedManagedVariableIfPresent));
  return LuthierReservedManagedVariableIfPresent.has_value() &&
         LuthierReservedManagedVariableIfPresent->getBinding() ==
             llvm::ELF::STB_GLOBAL;
}

} // namespace

namespace luthier {

llvm::Expected<bool> CompiledInstrumentationModule::isInstrumentationModule(
    llvm::ArrayRef<uint8_t> CodeObject) {
  std::unique_ptr<llvm::object::ObjectFile> ObjFile;
  LUTHIER_RETURN_ON_ERROR(
      luthier::object::createObjectFile(llvm::toStringRef(CodeObject))
          .moveInto(ObjFile));
  return ::isInstrumentationModule(*ObjFile);
}

llvm::Expected<std::unique_ptr<CompiledInstrumentationModule>>
CompiledInstrumentationModule::get(std::vector<uint8_t> CodeObject) {

  std::unique_ptr<llvm::object::ObjectFile> ObjFile;
  LUTHIER_RETURN_ON_ERROR(
      luthier::object::createObjectFile(llvm::toStringRef(CodeObject))
          .moveInto(ObjFile));

  // Check for the presence of the Luthier reserved managed variable
  // Check for the Luthier reserved managed variable inside the object
  bool IsReservedManagedVarPresent{false};
  LUTHIER_RETURN_ON_ERROR(::isInstrumentationModule(*ObjFile).moveInto(
      IsReservedManagedVarPresent));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      IsReservedManagedVarPresent,
      "Failed to locate the Luthier managed variable "
      "inside the instrumentation module."));

  // Get the CUID of the instrumentation module
  size_t CUID = 0;

  for (const auto &Symbol : ObjFile->symbols()) {
    llvm::Expected<llvm::StringRef> SymNameOrErr = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymNameOrErr.takeError());

    if (SymNameOrErr->starts_with(HipCUIDPrefix)) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          llvm::to_integer(SymNameOrErr->substr(strlen(HipCUIDPrefix)), CUID),
          "Failed to parse the CUID of the instrumentation module"));
      break;
    }
  }

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      CUID != 0, "Failed to find the CUID of the instrumentation module."));

  // Confirm that the ELF contains a Luthier bitcode section
  std::unique_ptr<llvm::MemoryBuffer> BCBuffer;
  LUTHIER_RETURN_ON_ERROR(getBCBufferInObjectFile(*ObjFile).moveInto(BCBuffer));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      BCBuffer != nullptr, "Failed to find the Luthier bitcode section inside "
                           "the instrumentation module"));

  // Confirm that the target triple of the bitcode and the object file are the
  // same
  llvm::Expected<std::string> BitCodeTTOrErr =
      llvm::getBitcodeTargetTriple(*BCBuffer);
  LUTHIER_RETURN_ON_ERROR(BitCodeTTOrErr.takeError());

  llvm::Triple ObjFileTriple = ObjFile->makeTriple();

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      ObjFileTriple == llvm::Triple(*BitCodeTTOrErr),
      "The Target Triple of the object file of the instrumentation module "
      "({0}) does not match its instrumentation bitcode {1}",
      ObjFileTriple, *BitCodeTTOrErr));

  return std::make_unique<CompiledInstrumentationModule>(
      std::move(CodeObject), std::move(ObjFile), std::move(BCBuffer), CUID);
}

} // namespace luthier