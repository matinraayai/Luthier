set(LLVM_LINK_COMPONENTS
        AllTargetsAsmParsers
        AllTargetsCodeGens
        AllTargetsDescs
        AllTargetsInfos
        Analysis
        AsmPrinter
        CodeGen
        CodeGenTypes
        Core
        IRPrinter
        IRReader
        MC
        MIRParser
        Passes
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

target_include_directories(luthier-llc PRIVATE ${LLVM_INCLUDE_DIRS})

export_executable_symbols_for_plugins(luthier-llc)
