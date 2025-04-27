add_executable(
        object-test
        object-test.cpp
)

target_include_directories(object-test PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${LLVM_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
        object-test
        LuthierCommon
        LuthierObject
        LuthierLLVM
        LLVMCore
        LLVMCodeGen
        LLVMObject
        LLVMBinaryFormat
        LLVMTargetParser
        LLVMSupport
)

add_dependencies(luthier-lit-tests object-test)