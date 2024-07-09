add_library(luthier_common STATIC
        cloning.cpp
        error.cpp
        object_utils.cpp
)

target_include_directories(luthier_common PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})

target_include_directories(luthier_common PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})


target_compile_options(luthier_common PUBLIC "-fPIC")
add_llvm_compile_definitions(luthier_common)

target_link_libraries(luthier_common PUBLIC hip::host amd_comgr LLVMCore LLVMMC LLVMObject LLVMSupport LLVMTarget)

if (${LUTHIER_BUILD_UNIT_TESTS})
    add_subdirectory(unittest)
endif()