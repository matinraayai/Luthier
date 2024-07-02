#include <unordered_map>
#include <string>
#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/DebugInfo/DWARF/DWARFDie.h>
#include <llvm/Object/ELFObjectFile.h>
#include <optional>
#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_isa.hpp"
#include "hsa_loaded_code_object.hpp"
#include "llvm/ADT/StringMap.h"



namespace llvm {
    class DWARDFDie;
    class ELFObjectFile;
    class DebugLoc;
} // namespace llvm


namespace luthier {

/**
 * @brief represents debugging information for a single LCO (Loaded Code Object)
 *
 * This class is responsible for parsing and caching the DWARF debugging info
 * from the 'debug_info' section of the ELF that the LCO belongs to.
 *
 *
 In CodeLifter: create one of this and map to the corresponding LCO (unordermap in the CodeLifter)
*/
class DWARFDebugInfo {

private:

    /**
     * Maps the name of the symbol to its corresponding DIE
     * Specifically, we parse the 'DW_AT_Name' attribute from the DIE, and store
     * the entry in the map. This way, debugging information is available for
     * each hsa::ExecutableSymbol in the loaded code object.
    */
    llvm::StringMap<std::optional<llvm::DWARFDie>> symbolNameToDie;

    /**
     * Parses and stores the debugging information
     * from the given ELFObjectFile<ELFT> in the symbolNameToDie
     *
     * We use the templated ELFObjectFile to support: (64LE, 32LE, 64BE, 32BE)
     * @Args:
     *      loadedCodeObject: represents a loaded code object
    */
    void loadDebugInfo(hsa::LoadedCodeObject &loadedCodeObject);
public:
    /**
     * Constructs a DWARFDebugInfo with the given LCO.
     * The symbolNameToDie map will be populated by invoking loadDebugInfo().
     *
    */
    DWARFDebugInfo(hsa::LoadedCodeObject loadedCodeObject);

    /**
     * Destruct this DWARFDebugInfo object
    */
    ~DWARFDebugInfo();

    /**
     * Gets the DWARFDie associated with the given symbol name from the map.
     * If no die is found, an llvm::Error is thrown, and the Expected object
     * reflects that through the takeError() method.
    */
    llvm::Expected<std::optional<llvm::DWARDFDie>> getDIE(llvm::StringRef symbolName);

    /**
     * Gets the DebugLoc associated with the given symbol name
     * Invokes the getDIE, and extracts the relevant information to build an
     * llvm::DebugLoc object, which will be cached in each Instr
    */
    llvm::Expected<std::optional<llvm::DebugLoc>> getDebugLoc(llvm::StringRef symbolName, llvm::Module module);

};


// Questions for matin:
// 1) should i pass the LCO by reference to the constructor?
// 2) throwing exception (make_error vs error.cpp in getDebugLoc and getDIE
// 3) show how I create the DWARFDebugInfo and add it to the map in disassemble function!
// 4) should the map in this class be storing a unique pointer to the DIE? Or is this fine? Should I worry about the scope stuff here?
        // (i think, as long as we a have a unique pointer to this object in the CodeLifter, then it won't be destructed prematurely, thus, the map won't be too)
// 5) should we still be caching the DebugLoc in the Instr?
// 6) VERY IMPORTANT:
//      initial goal: get some DebugLoc to be passed to the BuildMI(), which currently invokes the default constructor
//      now, we store DebugInfo for each ExecutableSymbol for each LCO, but not the actual Instr -> so, how do we use my work to extract accurate debug info for each Instr?
//      MY BAD FOR NOT THINKING THIS THROUGH PREVIOUSLY!
//
// 7) A few errors to be resolved!
// 8) TESTING! Need to see if any of this works.
//  llvm::outs() or dump on die

} // namespace luthier
