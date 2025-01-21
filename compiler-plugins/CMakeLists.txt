cmake_minimum_required(VERSION 3.21)

project(LuthierCompilerPlugins
        VERSION 0.1.0
        LANGUAGES CXX
        DESCRIPTION "Compiler plugins for the Luthier project"
)

set(CMAKE_CXX_STANDARD 20)

find_package(LLVM REQUIRED CONFIG)

include(AddLLVM)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Header-only library interface for the intrinsics library
add_library(Intrinsics INTERFACE)
target_include_directories(Intrinsics INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

add_subdirectory(lib/EmbedIModulePlugin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

add_subdirectory(cmake/modules)