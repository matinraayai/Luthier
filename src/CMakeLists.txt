# Set the bin and lib dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(hip REQUIRED CONFIG)

find_package(rocprofiler-sdk REQUIRED CONFIG)

find_package(hsa-runtime64 REQUIRED CONFIG)

find_package(LLVM REQUIRED CONFIG)

find_package(Clang REQUIRED CONFIG)

find_package(LLVMSPIRVTranslator)
if (LUTHIER_LLVM_SPIRV_TRANSLATOR_FOUND)
    message(STATUS "Found AMD SPIR-V translator: ${LUTHIER_LLVM_SPIRV_TRANSLATOR}")
else ()
    message(STATUS "AMD SPIR-V to LLVM translator not found; SPIR-V support disabled "
            "(set LUTHIER_LLVM_SPIRV_TRANSLATOR_PREFIX_PATH to enable).")
    add_compile_definitions(LUTHIER_DISABLE_LLVM_SPIRV_TRANSLATOR)
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


