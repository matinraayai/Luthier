#include "dwarf_debug.hpp"
#include "error.hpp"
#include "llvm/IR/DIBuilder.h"
#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/DebugInfo/DWARF/DWARFDie.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/AMDGPUMetadata.h>
#include <llvm/Support/Error.h>

namespace luthier {

DWARFDebugInfo::DWARFDebugInfo(const luthier::AMDGCNObjectFile &elfFile) {
  this->loadDebugInfo(elfFile);
}

DWARFDebugInfo::~DWARFDebugInfo() {}

/**
 * @brief Parses and loads the DWARF 'debug_info' section into the map.
 *
 * We get the ELF from the LCO
 * Then, we create the DWARFContext (for later, pass down the cached
 * dwarfcontext) and iterate over the DIEs and, for each die, if the die
 * has a DW_AT_Name attribute, we store an entry in our map.
 *
 * A compile unit represents the DIE for a file, which itself contains child
 * DIES, where each child may represent a subroutine, a type, or a variable. A
 * subroutine (or function) also has child DIES, which could be variables for
 * example.
 *
 * A type and a variable are typically the "leafs" of the DIE tree
 * structure of the DWARF debug_info section of an ELF.
 *
 * NOTE FOR FUTURE TESTS: MAKE SURE ALL THE DIES ARE PARSED!
 */
void DWARFDebugInfo::loadDebugInfo(const luthier::AMDGCNObjectFile &elfFile) {
  auto dwarfContext = llvm::DWARFContext::create(elfFile));
  this->dwarfCtx = *dwarfContext;
  for (auto cu : this->dwarfCtx.compile_units()) {
    llvm::DWARFDie cuDie =
        cu->getUnitDIE(); // get the DIE that represents the CU (root DIE)
    for (auto childDie : cuDie.children()) {
      auto dieSymbolName = childDie.getShortName();
      if (dieSymbolName) {
        this->symbolNameToDie.insert({dieSymbolName, childDie});
      }
    }
  }
}

llvm::Expected<std::optional<llvm::DWARFDie>>
DWARFDebugInfo::getDIE(llvm::StringRef symbolName) {
  try {
    auto correspondingDie = this->symbolNameToDie.at(symbolName);
    return correspondingDie;
  } catch (std::out_of_range &err) {
    return llvm::make_error(
        "Die not found for symbol: " +
        symbolName); // PLACEHOLDER FOR NOW, @matin, need to know proper way of
                     // throwing the exception using error.hpp
  }
}

/**
 * NEEDS DOCS:
 * SCOPE STUFF RESOLVED!
 * TODO: NEED to add it to the module + need to update disassembler.cpp
 *
 */
llvm::Expected<std::optional<llvm::DebugLoc>>
DWARFDebugInfo::getDebugLoc(llvm::StringRef symbolName,
                            uint64_t instructionAddress, llvm::Module module,
                            llvm::Metadata &instructionScope) {
  auto correspondingDie = this->getDIE(symbolName);
  LUTHIER_RETURN_ON_ERROR(correspondingDie.takeError());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION((*correspondingDie) != std::nullopt));
  auto cuDie = (*correspondingDie)->getDwarfUnit();
  auto dwarfLineTable = this->dwarfCtx.getLineTableForUnit(cuDie);
  if (dwarfLineTable) {
    for (auto &row : dwarfLineTable->Rows) {
      if (instructionAddress == row.Address.Address) {
        // NEED TO CHECK IF THIS FUNCTION ACTUALLY WORKS (the static get)
        llvm::DILocation diLocation = llvm::DILocation::get(row.Line, row.Column, instructionScope);
        return llvm::DebugLoc(diLocation);
      }
    }
    return llvm::make_error("Instruction address is not found in line table");
  } else {
    return llvm::make_error("Line table not found"); // PLACEHOLDER
  }
}
} // namespace luthier
