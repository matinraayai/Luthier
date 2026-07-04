# Set the bin and lib dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(hip REQUIRED CONFIG)

find_package(rocprofiler-sdk REQUIRED CONFIG)

find_package(hsa-runtime64 REQUIRED CONFIG)

find_package(LLVM REQUIRED CONFIG)

find_package(Clang REQUIRED CONFIG)

if (${LUTHIER_ENABLE_SPIRV})
    include(FindLLVMSPIRVTranslator)
    find_package(LLVMSPIRVTranslator)

    # SPIR-V JIT requires AMD device libs to be linked before IR optimization
    find_package(AMDDeviceLibs)
    if (AMD_DEVICE_LIBS_PREFIX AND
            EXISTS "${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode")
        set(LUTHIER_DEVICE_LIBS_DIR "${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode")
    endif ()
    if (LUTHIER_DEVICE_LIBS_DIR)
        message(STATUS
                "Luthier SPIR-V JIT device-libs bitcode: ${LUTHIER_DEVICE_LIBS_DIR}")
    else ()
        message(FATAL_ERROR
                "SPIR-V translator enabled but the AMD device-libs bitcode "
                "directory was not found.")
    endif ()
endif ()

# SPIR-V JIT requires AMD device libs to be linked before IR optimization
if (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    find_package(AMDDeviceLibs CONFIG)
    if (AMD_DEVICE_LIBS_PREFIX AND
            EXISTS "${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode")
        set(LUTHIER_DEVICE_LIBS_DIR "${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode")
    endif ()
    if (LUTHIER_DEVICE_LIBS_DIR)
        message(STATUS
                "Luthier SPIR-V JIT device-libs bitcode: ${LUTHIER_DEVICE_LIBS_DIR}")
    else ()
        message(FATAL_ERROR
                "SPIR-V translator enabled but the ROCm device-libs bitcode "
                "directory was not found (AMDDeviceLibs).")
    endif ()
endif ()

include(LuthierFetchLLVMSrc)

if (NOT IS_DIRECTORY ${LUTHIER_LLVM_SRC_DIR})
    luthier_fetch_llvm_src(${LLVM_INCLUDE_DIRS} LUTHIER_LLVM_SRC_DIR)
endif ()

# Get the include directory of hsa-runtime64
get_target_property(hsa-runtime64_INCLUDE_DIRS hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)

add_subdirectory(bin)
add_subdirectory(lib)

if (${LUTHIER_BUILD_UNIT_TESTS})
    add_subdirectory(test)
endif ()


