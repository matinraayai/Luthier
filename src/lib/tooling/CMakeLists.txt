add_library(luthier_tooling SHARED
        Controller.cpp
)

target_include_directories(luthier_tooling PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})
target_include_directories(luthier_tooling PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})

target_compile_options(luthier_tooling PUBLIC "-fPIC")
add_hsa_compile_definitions(luthier_tooling)
add_llvm_compile_definitions(luthier_tooling)

target_link_libraries(luthier_tooling PUBLIC luthier_common luthier_tooling_common luthier_hip luthier_hsa
        hip::host rocprofiler-sdk::rocprofiler-sdk amd_comgr
        hsa-runtime64::hsa-runtime64)