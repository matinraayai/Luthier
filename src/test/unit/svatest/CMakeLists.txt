set(LLVM_LINK_COMPONENTS
        AllTargetsAsmParsers
        AllTargetsCodeGens
        AllTargetsDescs
        AllTargetsDisassemblers
        AllTargetsInfos
        AMDGPUUtils
        Analysis
        BinaryFormat
        BitWriter
        AsmPrinter
        BitReader
        CodeGen
        CodeGenTypes
        Core
        IRPrinter
        IRReader
        Linker
        MC
        MCDisassembler
        MIRParser
        ObjCARCOpts
        Object
        Symbolize
        DebugInfoDWARF
        Passes
        Remarks
        ScalarOpts
        SelectionDAG
        Support
        Target
        TargetParser
        TransformUtils
        ipo
        Vectorize
)

add_llvm_executable(SVAFrameLanesTest
        SVAFrameLanesTest.cpp

        DEPENDS
        intrinsics_gen
)

target_compile_definitions(SVAFrameLanesTest PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(SVAFrameLanesTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_SOURCE_DIR}/include/
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(SVAFrameLanesTest PRIVATE
        gtest
        gtest_main
        LuthierAMDGPU
        LuthierIntrinsic
        LuthierCommon
        LuthierToolCodeGen
        LuthierToolIRCompilation
        LuthierObject
        LuthierHSA
        LuthierLLVM
        LuthierRocprofilerSDK
        hsa-runtime64::hsa-runtime64
        rocprofiler-sdk::rocprofiler-sdk
)

gtest_discover_tests(SVAFrameLanesTest)
