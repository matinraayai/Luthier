cmake_minimum_required(VERSION 3.21)
project(LiftLaunchedKernels LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

find_package(LLVM REQUIRED CONFIG)

add_library(LiftLaunchedKernels SHARED LiftLaunchedKernels.hip)

luthier_add_hsa_compile_definitions(LiftLaunchedKernels)
luthier_add_position_independence(LiftLaunchedKernels)
luthier_add_llvm_compile_definitions(LiftLaunchedKernels)

target_link_libraries(LiftLaunchedKernels PUBLIC LuthierTooling hip::device hip::host)