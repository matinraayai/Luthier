add_executable(
        luthier-link
        luthier-link.cpp
)

target_include_directories(luthier-link PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${LLVM_INCLUDE_DIRS}
        ${LLD_INCLUDE_DIRS})

target_link_libraries(
        luthier-link
        LuthierLinker
        LuthierCommon
        LuthierObject
        LLVMObject
        LLVMBinaryFormat
        LLVMTargetParser
        LLVMSupport
        lldELF
        lldCommon
)

add_dependencies(luthier-lit-tests luthier-link)
