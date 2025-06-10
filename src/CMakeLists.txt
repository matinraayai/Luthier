# Set the bin and lib dirs
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(hip REQUIRED CONFIG)

find_package(amd_comgr REQUIRED CONFIG)

find_package(rocprofiler-sdk REQUIRED CONFIG)

find_package(hsa-runtime64 REQUIRED CONFIG)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(LLVM REQUIRED CONFIG)

include(LuthierFetchLLVMSrc)

if (NOT IS_DIRECTORY ${LUTHIER_LLVM_SRC_DIR})
    luthier_fetch_llvm_src(${LLVM_INCLUDE_DIRS} LUTHIER_LLVM_SRC_DIR)
endif ()

# Get the include directory of amd_comgr
get_target_property(amd_comgr_INCLUDE_DIRS amd_comgr INTERFACE_INCLUDE_DIRECTORIES)

# Get the include directory of hsa-runtime64
get_target_property(hsa-runtime64_INCLUDE_DIRS hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)

# Get the include directory and defines of hip::host
get_target_property(hip_INCLUDE_DIRS hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES)

get_target_property(hip_DEFINES hip::host INTERFACE_COMPILE_DEFINITIONS)
add_subdirectory(bin)
add_subdirectory(lib)

if (${LUTHIER_BUILD_UNIT_TESTS})
    add_subdirectory(test)
endif ()


