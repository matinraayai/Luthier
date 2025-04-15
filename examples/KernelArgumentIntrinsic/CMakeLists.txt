cmake_minimum_required(VERSION 3.21)
project(LuthierKernelArgumentIntrinsic LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

find_package(LLVM REQUIRED)

add_library(LuthierKernelArgumentIntrinsic SHARED KernelArgumentIntrinsic.hip)

luthier_add_compiler_plugin(LuthierKernelArgumentIntrinsic LuthierIModuleEmbedPlugin)

target_link_libraries(LuthierKernelArgumentIntrinsic PUBLIC LuthierTooling LLVMDemangle hip::device hip::host)