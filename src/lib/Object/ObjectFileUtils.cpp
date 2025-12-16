//===-- ObjectFileUtils.cpp - Luthier Object File Utilities ---------------===//
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
/// This file implements a set of object utilities used frequently by Luthier.
//===----------------------------------------------------------------------===//
#include "luthier/Object/ObjectFileUtils.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include <llvm/TargetParser/SubtargetFeature.h>

namespace luthier::object {

llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>>
createObjectFile(llvm::StringRef ObjFile, bool InitContent) {
  std::unique_ptr<llvm::MemoryBuffer> Buffer =
      llvm::MemoryBuffer::getMemBuffer(ObjFile, "", false);
  return llvm::object::ObjectFile::createObjectFile(
      *Buffer, llvm::file_magic::unknown, InitContent);
}

llvm::Expected<std::string>
getObjectFileTarget(const llvm::object::ObjectFile &ObjFile) {
  llvm::Triple TT = ObjFile.makeTriple();
  std::optional<llvm::StringRef> CpuNameIfAvailable = ObjFile.tryGetCPUName();
  llvm::Expected<llvm::SubtargetFeatures> SubTargetFeaturesOrErr =
      ObjFile.getFeatures();
  LUTHIER_RETURN_ON_ERROR(SubTargetFeaturesOrErr.takeError());

  std::string Out =
      TT.str() + "--" +
      (CpuNameIfAvailable.has_value() ? std::string(*CpuNameIfAvailable)
                                      : "unknown");
  std::string FeatureString = SubTargetFeaturesOrErr->getString();
  if (!FeatureString.empty())
    Out += ":" + SubTargetFeaturesOrErr->getString();
  return Out;
}

llvm::Expected<
    std::tuple<llvm::Triple, llvm::StringRef, llvm::SubtargetFeatures>>
getObjectFileTargetTuple(const llvm::object::ObjectFile &Obj) {
  llvm::Triple TT = Obj.makeTriple();
  std::optional<llvm::StringRef> CPU = Obj.tryGetCPUName();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      CPU.has_value(), "Failed to get the CPU name of the object file."));
  llvm::SubtargetFeatures Features;
  LUTHIER_RETURN_ON_ERROR(Obj.getFeatures().moveInto(Features));
  return std::make_tuple(TT, *CPU, Features);
}

llvm::Expected<llvm::StringMap<uint64_t>>
getSymbolLoadOffsetsMap(const llvm::object::ObjectFile &ObjFile) {
  llvm::StringMap<uint64_t> Out;
  for (const llvm::object::SymbolRef &Symbol : ObjFile.symbols()) {
    llvm::Expected<llvm::StringRef> SymNamOrErr = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymNamOrErr.takeError());

    llvm::Expected<uint64_t> SymAddrOrErr = Symbol.getAddress();
    LUTHIER_RETURN_ON_ERROR(SymAddrOrErr.takeError());

    Out.insert({*SymNamOrErr, *SymAddrOrErr});
  }

  return Out;
}

} // namespace luthier::object