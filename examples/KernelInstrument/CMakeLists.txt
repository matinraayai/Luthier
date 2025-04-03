cmake_minimum_required(VERSION 3.21)
project(KernelInstrument LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

find_package(LLVM REQUIRED)

add_library(KernelInstrument SHARED KernelInstrument.hip)

luthier_add_compiler_plugin(KernelInstrument LuthierIModuleEmbedPlugin)

target_link_libraries(KernelInstrument PUBLIC LuthierTooling LLVMDemangle hip::device hip::host)