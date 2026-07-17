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
        BitWriter
        CodeGen
        CodeGenTypes
        Core
        Demangle
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
        DEPENDS
        intrinsics_gen
        SUPPORT_PLUGINS
)

target_include_directories(luthier-llc PRIVATE ${LLVM_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(luthier-llc PRIVATE
        LuthierToolCodeGen
        LuthierToolIRCompilation
        LuthierToolCodeGenTesting
        LuthierPassPlugin
        LuthierIntrinsic
        LuthierObject
        LuthierLLVM
        LuthierCommon)

export_executable_symbols_for_plugins(luthier-llc)
