cmake_minimum_required(VERSION 3.21)
project(instr_count LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/)

add_library(instr_count SHARED InstrCount.hip)

# Some HSA include files won't work unless this is defined
target_compile_definitions(instr_count PUBLIC AMD_INTERNAL_BUILD)

set_property(TARGET instr_count PROPERTY COMPILE_FLAGS "-fPIC")

find_package(LLVM REQUIRED CONFIG PATHS ${LUTHIER_LLVM_BIN_DIR} NO_DEFAULT_PATH)

target_include_directories(instr_count PUBLIC ../../include/)

target_include_directories(instr_count PUBLIC ${LLVM_INCLUDE_DIRS})

target_link_directories(instr_count PUBLIC ${LUTHIER_DIR})

set(LUTHIER_BC_EMBED_PLUGIN_PATH "" CACHE STRING "Path to the Luthier compiler plugin for embedding optimized bitcode")

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fpass-plugin=${LUTHIER_BC_EMBED_PLUGIN_PATH}")

target_link_libraries(instr_count PUBLIC luthier_tooling hip::host hip::device)