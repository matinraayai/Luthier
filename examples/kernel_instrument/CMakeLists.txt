cmake_minimum_required(VERSION 3.2)
project(kernel_instrument LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

find_package(hsa-runtime64 REQUIRED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/)


add_library(kernel_instrument SHARED kernel_instrument.cpp)

# Some HSA include files won't work unless this is defined
target_compile_definitions(kernel_instrument PUBLIC AMD_INTERNAL_BUILD)


set_source_files_properties(kernel_instrument.cpp PROPERTIES LANGUAGE HIP)

set_property(TARGET kernel_instrument PROPERTY COMPILE_FLAGS "-fPIC")

find_package(LLVM REQUIRED CONFIG PATHS ${LUTHIER_LLVM_BIN_DIR} NO_DEFAULT_PATH)

target_include_directories(kernel_instrument PUBLIC ../../include/)

target_include_directories(kernel_instrument PUBLIC ${LLVM_INCLUDE_DIRS})

target_link_directories(kernel_instrument PUBLIC ${LUTHIER_DIR})

set(LUTHIER_BC_EMBED_PLUGIN_PATH "" CACHE STRING "Path to the Luthier compiler plugin for embedding optimized bitcode")

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xoffload-linker --emit-relocs -fpass-plugin=${LUTHIER_BC_EMBED_PLUGIN_PATH}")

target_link_libraries(kernel_instrument PUBLIC luthier hip::host hip::device)