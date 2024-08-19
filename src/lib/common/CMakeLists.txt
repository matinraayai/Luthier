add_library(LuthierCommon STATIC
        Cloning.cpp
        Error.cpp
        ObjectUtils.cpp
)

target_include_directories(LuthierCommon PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})

target_include_directories(LuthierCommon PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})


target_compile_options(LuthierCommon PUBLIC "-fPIC")
add_llvm_compile_definitions(LuthierCommon)

target_link_libraries(LuthierCommon PUBLIC hip::host amd_comgr LLVMCore LLVMMC LLVMObject LLVMSupport LLVMTarget)

if (${LUTHIER_BUILD_UNIT_TESTS})
    add_subdirectory(unittest)
endif ()