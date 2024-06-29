#include "dwarf_debug.hpp"
#include "llvm/IR/DIBuilder.h"
#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/DebugInfo/DWARF/DWARFDie.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/AMDGPUMetadata.h>
#include <llvm/Support/Error.h>
#include "error.hpp"



namespace luthier {




/**
 * @brief Parses and loads the DWARF 'debug_info' section into the map.
 *
 * We get the ELF from the LCO
 * Then, we create the DWARFContext (for later, pass down the cached dwarfcontext)
 * and iterate over the DIEs and, for each die, if the die represents an ExecutableSymbol in the LCO, we store
 * an entry in our map.
 *
 * A compile unit represents the DIE for a file, which itself contains child DIES,
 * where each child may represent a subroutine, a type, or a variable.
 * A subroutine (or function) also has child DIES, which could be variables for example.
 *
 * A type and a variable are typically the "leafs" of the DIE tree
 * structure of the DWARF debug_info section of an ELF.
*/
void DWARFDebugInfo::loadDebugInfo(hsa::LoadedCodeObject &loadedCodeObject) {
    auto elfFile = loadedCodeObject.getStorageELF();
    LUTHIER_RETURN_ON_ERROR(elfFile.takeError());
    auto dwarfContext = llvm::DWARFContext::create(*elfFile));
    for (auto cu: dwarfContext->compile_units()) {
        llvm::DWARFDie cuDie = cu->getUnitDIE(); // get the DIE that represents the CU (root DIE)
        for (auto childDie: cuDie.children()) {
            llvm::StringRef dieSymbolName = childDie.getShortName();
            auto executableSymbol = loadedCodeObject.getExecutableSymbolByName(dieSymbolName);
            // G(executableSymbol.takeError()) {
            if (!executableSymbol.takeError()) {
                this->symbolNameToDie.insert({dieSymbolName, childDie});
            }
        }
    }
}
llvm::Expected<std::optional<llvm::DWARDFDie>> DWARFDebugInfo::getDIE(llvm::StringRef symbolName) {
    try {
        auto correspondingDie = this->symbolNameToDie.at(symbolName);
        return correspondingDie;
    } catch (std::out_of_range &err) {
        return llvm::make_error("Die not found for symbol: " + symbolName); // PLACEHOLDER FOR NOW, @matin, need to know proper way of throwing the exception using error.hpp
    }
}

llvm::Expected<std::optional<llvm::DebugLoc>> DWARFDebugInfo::getDebugLoc(llvm::StringRef symbolName, llvm::Module module) {
    auto correspondingDie = this->getDIE(symbolName);
    LUTHIER_RETURN_ON_ERROR(correspondingDie.takeError());
  auto line = correspondingDie.find(llvm::dwarf::DW_AT_decl_line)->getAsUnsignedConstant().value();
  auto col =
      correspondingDie.find(llvm::dwarf::DW_AT_decl_column)->getAsUnsignedConstant().value();
  llvm::DIBuilder dIBuilder(
      module, true,
      nullptr); // allowUnresolved = true (i don't really know what that means)
  llvm::Metadata *mdScope = nullptr;
  // // Determine the scope
  if (auto dwarfScopeAttr = correspondingDie.find(llvm::dwarf::DW_AT_start_scope)) {
    const uint64_t mdScopeOffset = dwarfScopeAttr->getAsReference().value();
    if (auto mdScopeDie = correspondingDie.getDwarfUnit()->getDIEForOffset(mdScopeOffset)) {
      if (mdScopeDie.isValid()) {
        // if its a subprogram (func) or lexical block /scope ({})
        // need to handle case where: ScopeDie.getTag() ==
        // dwarf::DW_TAG_lexical_block
        if (mdScopeDie.getTag() == llvm::dwarf::DW_TAG_subprogram) {
          mdScope = dIBuilder.createFunction(
              nullptr, mdScopeDie.getShortName(), StringRef(), nullptr,
              mdScopeDie.getDeclLine(), nullptr, 0, llvm::DINode::FlagZero,
              llvm::DISubprogram::SPFlagZero);
          // need to add it to the module, but having trouble: signature
          // mismatch (FunctionType vs DISubprogram*) -> FunctionCallee
          // getOrInsertFunction (StringRef Name, FunctionType *T, AttributeList
          // AttributeList)
        } else if (mdScopeDie.getTag() ==
                   llvm::dwarf::DW_TAG_file_type) { // else, if it's a file
          mdScope = dIBuilder.createFile(
              mdScopeDie.getShortName(),
              mdScopeDie.getDeclFile(llvm::DILineInfoSpecifier::
                                         FileLineInfoKind::RelativeFilePath));
        }
      }
    }
  }
  if (!mdScope) {
    // Fallback to using the CU's file as the scope
    auto compileUnit = correspondingDie.getDwarfUnit()->getUnitDIE();
    if (auto dwarfFileAttr = compileUnit.find(llvm::dwarf::DW_AT_name)) {
      const char *fileShortName = compileUnit.getShortName();
      if (!fileShortName) {
        return llvm::make_error("File not found; scope of symbol name couldn't be determined."); // PLACEHOLDER FOR NOW
      }
      mdScope = dIBuilder.createFile(
          fileShortName,
          compileUnit.getDeclFile(
              llvm::DILineInfoSpecifier::FileLineInfoKind::
                  RelativeFilePath)); // declFile, NEED TO get the directory and
                                      // pass it to createFile
    }
  }
  return llvm::DebugLoc(llvm::DILocation::get(line, col, mdScope)));
}
}
