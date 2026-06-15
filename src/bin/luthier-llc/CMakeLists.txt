set(LLVM_LINK_COMPONENTS
        AllTargetsAsmParsers
        AllTargetsCodeGens
        AllTargetsDescs
        AllTargetsDisassemblers
        AllTargetsMCAs
        AMDGPUUtils
        AllTargetsInfos
        Analysis
        BinaryFormat
        AsmPrinter
        BitReader
        CodeGen
        CodeGenTypes
        Core
        IRPrinter
        IRReader
        Linker
        MC
        MCA
        MCDisassembler
        MIRParser
        Object
        Symbolize
        DebugInfoDWARF
        Passes
        Plugins
        Remarks
        ScalarOpts
        SelectionDAG
        Support
        Target
        TargetParser
        TransformUtils
        Vectorize
)

add_llvm_tool(luthier-llc
        llc.cpp
        NewPMDriver.cpp
        DEPENDS
        intrinsics_gen
        SUPPORT_PLUGINS
)

target_include_directories(luthier-llc PRIVATE ${LLVM_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(luthier-llc PRIVATE
        LuthierToolCodeGenTesting
        LuthierObject
        LuthierLLVM
        LuthierCommon)

export_executable_symbols_for_plugins(luthier-llc)
