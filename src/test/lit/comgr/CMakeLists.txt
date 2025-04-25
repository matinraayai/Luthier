add_executable(
        comgr-link
        comgr-link.cpp
)

target_include_directories(comgr-link PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${LLVM_INCLUDE_DIRS})

target_link_libraries(
        comgr-link
        LuthierComgr
        LuthierCommon
        LuthierObject
        LLVMObject
        LLVMBinaryFormat
        LLVMTargetParser
        LLVMSupport
        amd_comgr
)

add_dependencies(luthier-lit-tests comgr-link)