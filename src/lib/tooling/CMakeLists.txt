add_library(LuthierTooling SHARED
        Controller.cpp
)

target_include_directories(LuthierTooling PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})
target_include_directories(LuthierTooling PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})

target_compile_options(LuthierTooling PUBLIC "-fPIC")
add_hsa_compile_definitions(LuthierTooling)
add_llvm_compile_definitions(LuthierTooling)

target_link_libraries(LuthierTooling PUBLIC LuthierCommon LuthierToolingCommon LuthierHip LuthierHSA
        hip::host rocprofiler-sdk::rocprofiler-sdk amd_comgr
        hsa-runtime64::hsa-runtime64)