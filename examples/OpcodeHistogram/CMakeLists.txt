cmake_minimum_required(VERSION 3.21)
project(OpcodeHistogram LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/)

add_library(OpcodeHistogram SHARED OpcodeHistogram.hip)

# Some HSA include files won't work unless this is defined
target_compile_definitions(OpcodeHistogram PUBLIC AMD_INTERNAL_BUILD)

set_property(TARGET OpcodeHistogram PROPERTY COMPILE_FLAGS "-fPIC")

find_package(LLVM REQUIRED CONFIG PATHS ${LUTHIER_LLVM_BIN_DIR} NO_DEFAULT_PATH)

target_include_directories(OpcodeHistogram PUBLIC ../../include/)

target_include_directories(OpcodeHistogram PUBLIC ${LLVM_INCLUDE_DIRS})

add_llvm_compile_definitions(OpcodeHistogram)

target_include_directories(OpcodeHistogram PUBLIC /llvm-project/)

target_link_directories(OpcodeHistogram PUBLIC ${LUTHIER_DIR})

set(LUTHIER_BC_EMBED_PLUGIN_PATH "" CACHE STRING "Path to the Luthier compiler plugin for embedding optimized bitcode")

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fpass-plugin=${LUTHIER_BC_EMBED_PLUGIN_PATH}")

target_link_libraries(OpcodeHistogram PUBLIC LuthierTooling LLVMDemangle hip::host hip::device)