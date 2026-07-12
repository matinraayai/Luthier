include(LuthierCreateOffloadBundle)

# ISAs precompiled into the bundle, toggling xnack / sramecc / wavefront-size / cu-mode.
# Must be kept in lockstep with the PrecompiledISAs table in \c EmbeddedBitcodeLoaderTest.hip.
set(_TEST_ISAS
        amdgcn-amd-amdhsa--gfx900:xnack-
        amdgcn-amd-amdhsa--gfx908:xnack+:sramecc-
        amdgcn-amd-amdhsa--gfx90a:xnack-:sramecc+
        amdgcn-amd-amdhsa--gfx942:xnack+:sramecc+
        amdgcn-amd-amdhsa--gfx1030:wavefrontsize64-:cumode-
        amdgcn-amd-amdhsa--gfx1100:wavefrontsize64+:cumode+)

luthier_create_offload_bundle(EmbeddedBitcodeLoaderTestObject
        ToolDeviceCodeOffloadParserTest.hip
        TARGET_ISAS ${_TEST_ISAS}
        DEVICE_OBJECT_LIBRARIES _DEV_TARGETS)

target_link_libraries(EmbeddedBitcodeLoaderTestObject PUBLIC
        LuthierTooling hip::host gtest)
foreach (_DEVICE_TARGET IN LISTS _DEV_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

# Enable SPIR-V testing if it is available
target_compile_definitions(EmbeddedBitcodeLoaderTestObject PRIVATE
        $<$<BOOL:${LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND}>:LUTHIER_HAS_SPIRV_TRANSLATOR>)

set(LLVM_LINK_COMPONENTS
        AllTargetsInfos
        AllTargetsCodeGens
        AllTargetsDescs
        AllTargetsAsmParsers
        AMDGPUUtils
        BitReader
        Core
        Passes
        Support
        Target
        TargetParser)

add_llvm_executable(ToolDeviceCodeParserLoaderTest
        $<TARGET_OBJECTS:EmbeddedBitcodeLoaderTestObject>)

target_include_directories(ToolDeviceCodeParserLoaderTest PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${LLVM_INCLUDE_DIRS})

target_link_libraries(ToolDeviceCodeParserLoaderTest PRIVATE
        gtest gtest_main LuthierTooling hip::host ${CMAKE_DL_LIBS})

gtest_discover_tests(ToolDeviceCodeParserLoaderTest)
