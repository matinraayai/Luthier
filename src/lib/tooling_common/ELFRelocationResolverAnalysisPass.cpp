//===-- RelocationResolverPass.hpp ----------------------------------------===//
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
/// This file implements the <tt>RelocationResolverAnalysisPass</tt> class.
//===----------------------------------------------------------------------===//
#include "tooling_common/ELFRelocationResolverAnalysisPass.hpp"
#include "tooling_common/ELFObjectFileAnalysisPass.hpp"
#include <llvm/ADT/IntervalMap.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/ELFObjectFile.h>
#include <luthier/llvm/LLVMError.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-relocation-resolver"

namespace luthier {

llvm::AnalysisKey ELFRelocationResolverAnalysisPass::Key;

ELFRelocationResolverAnalysisPass::Result
ELFRelocationResolverAnalysisPass::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &MAM) {
  Result Out;
  const llvm::object::ELFObjectFileBase &ObjFile =
      MAM.getResult<ELFObjectFileAnalysisPass>(M).getObject();

  /// Create an interval map between [symbol start address, symbol end address]
  /// -> symbol for symbols of type STT_FUNC, STT_OBJECT, and STT_IFUNC
  using SymbolIntervalMapT =
      llvm::IntervalMap<uint64_t, llvm::object::ELFSymbolRef>;
  SymbolIntervalMapT::Allocator A;
  SymbolIntervalMapT SymbolIntervalMap(A);

  for (const llvm::object::ELFSymbolRef &Sym : ObjFile.symbols()) {
    llvm::Expected<uint64_t> SymAddrOrErr = Sym.getAddress();
    if (llvm::Error Err = LLVM_ERROR_CHECK(SymAddrOrErr.takeError())) {
      M.getContext().emitError(toString(std::move(Err)));
    }
    const size_t SymSize = Sym.getSize();
    const uint8_t SymType = Sym.getELFType();
    if (SymType == llvm::ELF::STT_FUNC || SymType == llvm::ELF::STT_OBJECT ||
        SymType == llvm::ELF::STT_GNU_IFUNC)
      SymbolIntervalMap.insert(*SymAddrOrErr, *SymAddrOrErr + SymSize, Sym);
  }

  /// Begin extracting the relocation information from the ELF object file
  for (const llvm::object::ELFSectionRef Section : ObjFile.sections()) {
    for (const llvm::object::RelocationRef Reloc : Section.relocations()) {
      llvm::object::elf_symbol_iterator RelocSymbolIterator = Reloc.getSymbol();
      if (RelocSymbolIterator != ObjFile.symbol_end()) {
        llvm::object::ELFSymbolRef RelocSymbol = *RelocSymbolIterator;

        llvm::Expected<uint64_t> RelocSymAddrOrErr =
            Reloc.getSymbol()->getAddress();
        if (llvm::Error Err = RelocSymAddrOrErr.takeError()) {
          M.getContext().emitError(toString(std::move(Err)));
        }
        // If this symbol is not a function, object, or an ifunc, look into
        // the symbol interval map to find a symbol with those types in the same
        // address.
        // We do this because ELF relocation entries use STB_GLOBAL symbols, and
        // if the target symbol has a non-global binding, then the relocation
        // info will use the target symbol's section instead.
        // For example, we have an AMD GPU ELF, in which the kernel foo makes
        // a call to device function bar. Since in AMD GPU ELFs, device
        // functions have a local binding, the '.rela.text' will say the
        // relocation symbol associated with the address calculation
        // instructions (e.g.
        // S_GETPC_B64 s[4:5],
        // S_ADD_B64 s[4:5] bar@rel32@lo+4,
        // S_ADDC_B64 s[4:5] bar@rel32@hi+12) is '.text' instead of
        // 'bar':
        // Offset Type              Sym. Value  Sym. Name + Addend
        // 1950   R_AMDGPU_REL32_LO 01800       .text + 4
        // 1958   R_AMDGPU_REL32_HI 01800       .text + c
        // Now, if we look at the symbols, we can see that
        // symbol value 1800 actually corresponds to the function bar:
        // Num: Value Size Type    Bind   Vis      Ndx Name
        // 0:   0000  0    NOTYPE  LOCAL  DEFAULT  UND
        // 1:   1800  12   FUNC    LOCAL  DEFAULT    7 bar
        // 2:   1800  0    SECTION LOCAL  DEFAULT    7 .text
        // 3:   0780  0    SECTION LOCAL  DEFAULT    6 .rodata
        // 4:   0200  0    SECTION LOCAL  DEFAULT    1 .note
        // 5:   0000  0    SECTION LOCAL  DEFAULT   11 .comment
        // 6:   2990  0    NOTYPE  LOCAL  HIDDEN     8 _DYNAMIC
        // 7:   1900  140  FUNC    GLOBAL PROTECTED  7 foo
        // 8:   0780  64   OBJECT  GLOBAL PROTECTED  6 foo.kd
        // If we are not able to find this equivalent symbol, then we just
        // use the original symbol reported by the relocation entry
        uint8_t RelocSymType = RelocSymbolIterator->getELFType();
        if (RelocSymType != llvm::ELF::STT_FUNC &&
            RelocSymType != llvm::ELF::STT_OBJECT &&
            RelocSymType != llvm::ELF::STT_GNU_IFUNC) {
          SymbolIntervalMapT::const_iterator DataFuncSymbolEquivalent =
              SymbolIntervalMap.find(*RelocSymAddrOrErr);
          if (DataFuncSymbolEquivalent != SymbolIntervalMap.end()) {
            RelocSymbol = *DataFuncSymbolEquivalent;
          }
        }
        // Calculate the relocation location's offset
        // (from the start of the symbol), and then insert it and
        // its symbol into the result's map
        uint64_t RelocationOffset = Reloc.getOffset() - *RelocSymAddrOrErr;
        Out.Relocations.insert({{RelocSymbol, RelocationOffset}, Reloc});

        LLVM_DEBUG(

            llvm::Expected<llvm::StringRef> SymNameOrErr =
                RelocSymbol.getName();
            if (llvm::Error Err = SymNameOrErr.takeError()) {
              M.getContext().emitError(toString(std::move(Err)));
            };
            llvm::dbgs() << llvm::formatv(
                "Found relocation for symbol {0} at address {1:x}, with offset "
                "{2:x} from the start of the symbol.\n",
                *SymNameOrErr, Reloc.getOffset(), RelocationOffset)

        );
      }
    }
  }

  return Out;
}

} // namespace luthier