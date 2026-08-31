set(LLVM_LINK_COMPONENTS Support Object Symbolize DebugInfoDWARF)

# The arithmetic under test is header-only, but MemoryAllocationAccessor.h includes
# AMDGCNObjectFile.h, which pulls in LLVM's Object library and Luthier's error
# types. The few sources those need are compiled in directly rather than linking
# LuthierLLVM whole, following the pattern in ../rocprofiler/CMakeLists.txt --
# linking the object library instead drags in the AMDGPU target and its LLVM
# components for a test that needs none of it.
#
# No GPU and no runtime is involved. Deliberately so: the two bugs these tests pin
# were invisible to the lit suite, which reaches the same code through a
# disassembler and a mock loader whose device and host views are the same pointer.
add_llvm_executable(ToolCodeGenUnitTests
        PARTIAL_SOURCES_INTENDED
        MemoryAllocationAccessorTest.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/MemoryAllocationAccessor.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/LLVMError.cpp
        ${CMAKE_SOURCE_DIR}/src/lib/LLVM/streams.cpp)

target_compile_definitions(ToolCodeGenUnitTests PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(ToolCodeGenUnitTests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${LLVM_INCLUDE_DIRS})

target_link_libraries(ToolCodeGenUnitTests PRIVATE
        gtest gtest_main LuthierCommon)

gtest_discover_tests(ToolCodeGenUnitTests)
