# DWARF Debug Information Docs

## **Introduction**

When a HIP application is compiled (hipcc -g app.cpp), an ELF (executable file format) file is generated ('a.out'). By specifying the '-g' flag, the compiler includes debugging information in the ELF. The '.debug_info' section of the ELF contains DWARF (debugging With Attributed Record Formats) debugging information. DWARF is a standardized debugging data format used by many compilers and debuggers. Our goal will be to parse and cache the debugging information from the ELF, when the 'includeDebugInfo' argument of the disassemble function is true. This will be done once for each code oject (.hsaco). We will use 'llvm/debugInfo/DWARF/*' for parsing the DWARF from the ELFObjectFile and extracting the relevant debugging information (functions and variables).  

## **DWARF Format Overview**
DWARF data is organized into a series of sections, with .debug_info being one of the primary sections. The .debug_info section contains compilation units (CUs), each representing a source file. Each CU contains a series of debugging information entries (DIEs), which describe variables, types, functions, and other entities in the source code.

Structure:
  * Compilation Unit Header:
    *  Length: The total length of the compilation unit data, excluding the length field itself.
    * Version: The DWARF version number.
    * Abbreviation Offset: The offset into the .debug_abbrev section.
    * Pointer Size: The size of address pointers on the target machine.
  * Debugging Information Entries (DIEs):
    * Each DIE describes a specific entity in the source code (e.g., a function, variable, type).
    * DIEs are organized hierarchically, with parent-child relationships indicating scopes.
  * Attributes:
    * Each DIE contains attributes that provide information about the entity it describes.
    * Common attributes include the name, type, location, and line number of the entity.

Example:

    Compilation Unit @ offset 0x0:
    Length:        0x3f (32-bit)
    Version:       4
    Abbrev Offset: 0x0
    Pointer Size:  8

    <0><0x0000000b>    DW_TAG_compile_unit
                        DW_AT_producer    : (string) "GCC 10.2.0"
                        DW_AT_language    : DW_LANG_C_plus_plus
                        DW_AT_name        : (string) "example.cpp"
                        DW_AT_stmt_list   : 0x00000000
                        DW_AT_comp_dir    : (string) "/home/user"
                        DW_AT_low_pc      : 0x0000000000401000
                        DW_AT_high_pc     : 0x0000000000401020
    <1><0x0000002c>    DW_TAG_subprogram
                        DW_AT_name        : (string) "main"
                        DW_AT_decl_file   : 0x01
                        DW_AT_decl_line   : 5
                        DW_AT_type        : 0x00000038
                        DW_AT_low_pc      : 0x0000000000401000
                        DW_AT_high_pc     : 0x0000000000401014
    <2><0x00000038>    DW_TAG_base_type
                        DW_AT_byte_size   : 4
                        DW_AT_encoding    : DW_ATE_signed
                        DW_AT_name        : (string) "int"

## **Solution** (refined design)
THIS WILL BE BROKEN DOWN INTO THE SECTIONS FOR EACH FUNCTION
- Create a new class (DWARFDebugInfo)
    - Store: private unordered_map<std::string, llvm::DWARFDie> symbolNameToDie;
    - functions:
        - loadDebugInfo(LCO or ELFObjectFile<T>) -> populates symbolNameToDie
        - std::optional<llvm::Expected<DWARFDie>> getDWARFDie(StringRef symbolName) -> returns the DWARF associated with the symbol name
        - std::optional<llvm::Expected<DebugLoc>> getDebugLoc(StringRef symbolName) -> converts the Die associated with the symbol name to a DebugLoc
    - More on loadDebugInfo:
        - All we need is the ELFObjectFile
        - We create the DWARFContext, get all the compile units:
        - For each CU: DFS and store all children
        - For each DIE -> 
            - get the symbol name (if its a variable, or subroutine FOR NOW! add functionality later for other types), DWARF attribute: DW_AT_Name
            - add it to the Map
- Store this object in the CodeLifter
- Caching warning: make sure the object does not go out of scop while we need it.
