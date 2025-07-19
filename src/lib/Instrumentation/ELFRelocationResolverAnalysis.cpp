//===-- ELFRelocationResolverAnalysis.cpp ---------------------------------===//
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
/// Implements the \c RelocationResolverAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Instrumentation/GlobalObjectOffsetsAnalysis.h"
#include <llvm/ADT/IntervalMap.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/ELFObjectFile.h>
#include "luthier/Instrumentation/ELFRelocationResolverAnalysis.h"
#include "luthier/Instrumentation/ObjectFileAnalysisPass.h"
#include "luthier/LLVM/LLVMError.h"

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-relocation-resolver"

namespace luthier {

llvm::AnalysisKey ELFRelocationResolverAnalysis::Key;

ELFRelocationResolverAnalysis::Result
ELFRelocationResolverAnalysis::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &MAM) {
  Result Out;
  const llvm::object::ObjectFile &ObjFile =
      MAM.getResult<ObjectFileAnalysisPass>(M).getObject();

  llvm::LLVMContext &Ctx = M.getContext();

  LUTHIER_EMIT_ERROR_IN_CONTEXT(
      Ctx, LUTHIER_GENERIC_ERROR_CHECK(
               llvm::isa<llvm::object::ELFObjectFileBase>(ObjFile),
               "The object file is not an ELF file"));

  auto &GlobalObjectOffsets = MAM.getResult<GlobalObjectOffsetsAnalysis>(M);

  /// Begin extracting the relocation information from the ELF object file
  for (const llvm::object::ELFSectionRef Section : ObjFile.sections()) {
    for (const llvm::object::RelocationRef Reloc : Section.relocations()) {
      llvm::object::elf_symbol_iterator RelocSymbolIterator = Reloc.getSymbol();
      if (RelocSymbolIterator != ObjFile.symbol_end()) {
        llvm::object::ELFSymbolRef RelocSymbol = *RelocSymbolIterator;

        llvm::Expected<uint64_t> RelocSymAddrOrErr =
            Reloc.getSymbol()->getAddress();
        LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, RelocSymAddrOrErr.takeError());

        // Look for the associated global object for this address
        const llvm::GlobalObject *GO =
            GlobalObjectOffsets.getGlobalObject(*RelocSymAddrOrErr);

        LUTHIER_EMIT_ERROR_IN_CONTEXT(
            Ctx,
            LUTHIER_GENERIC_ERROR_CHECK(
                GO != nullptr, llvm::formatv("Failed to find the associated "
                                             "global object at offset {0:x}",
                                             *RelocSymAddrOrErr)));
        Out.Relocations.insert({Reloc.getOffset(), {*GO, Reloc}});

        LLVM_DEBUG(llvm::Expected<llvm::StringRef> SymNameOrErr =
                       RelocSymbol.getName();
                   LUTHIER_EMIT_ERROR_IN_CONTEXT(Ctx, SymNameOrErr.takeError());
                   llvm::dbgs() << llvm::formatv(
                       "Found relocation for symbol {0} at offset {1:x}.\n",
                       *SymNameOrErr, Reloc.getOffset()););
      }
    }
  }

  return Out;
}

} // namespace luthier