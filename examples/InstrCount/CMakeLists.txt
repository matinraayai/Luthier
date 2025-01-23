cmake_minimum_required(VERSION 3.21)
project(LuthierInstrCount LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

find_package(LLVM REQUIRED CONFIG)

find_package(LuthierCompilerPlugins REQUIRED CONFIG)

add_library(LuthierInstrCount SHARED InstrCount.hip)

luthier_add_hsa_compile_definitions(LuthierInstrCount)

luthier_add_position_independence(LuthierInstrCount)

luthier_add_llvm_compile_definitions(LuthierInstrCount)

luthier_add_compiler_plugin(LuthierInstrCount luthier::IModuleEmbedPlugin)

target_link_libraries(LuthierInstrCount PUBLIC LuthierTooling LLVMDemangle hip::device hip::host)