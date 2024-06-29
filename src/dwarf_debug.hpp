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



namespace llvm {
    class DWARDFDie;
    class ELFObjectFile;
    class DebugLoc;
} // namespace llvm


namespace luthier {

/**
 * This class is responsible for parsing and caching the DWARF debugging info
 * that corresponds to one LCO and one ELF (since an LCO belongs to one ELF)
*  MORE DOCS NEEDED
 In CodeLifter: create one of this and map to the corresponding LCO (unordermap in the CodeLifter)
*/
class DWARFDebugInfo {

private:
    /**
     * Maps the name of the DIE to the actual DIE
     * Specifically, we parse the 'DW_AT_Name' attribute from the DIE, and store
     * the entry in the map. This way, debugging information is available for
     * each hsa::ExecutableSymbol in the loaded code object.
    */
    std::unordered_map<llvm::StringRef, std::optional<llvm::DWARFDie>> symbolNameToDie;

    /**
     * Parses and stores the debugging information
     * from the given ELFObjectFile<ELFT> in the symbolNameToDie
     *
     * CHECK IF THE symbol is in the LCO (getExecutableSymbolByName) and store if it is
     *
     * We use the templated ELFObjectFile to support: (64LE, 32LE, 64BE, 32BE)
     * @Args:
     *      elfObjFile: represents the
     *
     *
     * TAKE THE LCO INSTEAD OF THE ELOBJECTFILE!
    */
    [[nodiscard]] void loadDebugInfo(hsa::LoadedCodeObject &loadedCodeObject);
public:
    DWARFDebugInfo(hsa::LoadedCodeObject loadedCodeObject);

    ~DWARFDebugInfo();

    /**
     * Returns the DWARFDie associated with the given symbol name
     * ADD MORE DOCS DOCS DOCS DOCS
    */
    [[nodiscard]] llvm::Expected<std::optional<llvm::DWARDFDie>> getDIE(llvm::StringRef symbolName);

    /**
     * Returns the DebugLoc associated with the given symbol name
     * ADD MORE DOCS DOCS DOCS DOCS
    */
    [[nodiscard]] llvm::Expected<std::optional<llvm::DebugLoc>> getDebugLoc(llvm::StringRef symbolName, llvm::Module module);

};
} // namespace luthier
