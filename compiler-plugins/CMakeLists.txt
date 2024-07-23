cmake_minimum_required(VERSION 3.21)

project(luthier-compiler-plugins
        VERSION 0.0.1
        LANGUAGES CXX
        DESCRIPTION "Compiler plugins for the Luthier project"
)

set(CMAKE_CXX_STANDARD 20)

find_package(LLVM REQUIRED CONFIG)

if (LLVM_LUTHIER_EMBED_OPTIMIZED_BC_PLUGIN_LINK_INTO_TOOLS)
    message(WARNING "Setting LLVM_LUTHIER_EMBED_OPTIMIZED_BC_PLUGIN=ON is not supported")
endif ()

include(AddLLVM)
add_llvm_pass_plugin(luthier_embed_optimized_bc_plugin embed_optimized_bitcode_pass.hpp embed_optimized_bitcode_pass.cpp)
target_include_directories(luthier_embed_optimized_bc_plugin PUBLIC ${LLVM_INCLUDE_DIRS})

# TODO: Add an installation/config recipe