cmake_minimum_required(VERSION 3.21)
project(LiftLaunchedKernels LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED CONFIG)

add_library(LiftLaunchedKernels SHARED LiftLaunchedKernels.hip)

set_property(TARGET LiftLaunchedKernels PROPERTY COMPILE_FLAGS "-fPIC")

target_link_libraries(LiftLaunchedKernels PUBLIC LuthierTooling)