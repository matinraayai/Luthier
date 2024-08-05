cmake_minimum_required(VERSION 3.21)

project(luthier-compiler-plugins
        VERSION 0.0.1
        LANGUAGES CXX
        DESCRIPTION "Compiler plugins for the Luthier project"
)

set(CMAKE_CXX_STANDARD 20)

find_package(LLVM REQUIRED CONFIG)

if (${LLVM_LUTHIER_EMBED_INSTRUMENTATION_MODULE_BITCODE_PLUGIN_LINK_INTO_TOOLS})
    message(WARNING "Setting LLVM_LUTHIER_EMBED_INSTRUMENTATION_MODULE_BITCODE_PLUGIN_LINK_INTO_TOOLS=ON
    is not supported")
endif ()

include(AddLLVM)
add_llvm_pass_plugin(LuthierEmbedInstrumentationModuleBitcodePlugin
        EmbedInstrumentationModuleBitcodePass.hpp
        EmbedInstrumentationModuleBitcodePass.cpp)
target_include_directories(LuthierEmbedInstrumentationModuleBitcodePlugin PUBLIC ${LLVM_INCLUDE_DIRS})

# TODO: Add an installation/config recipe