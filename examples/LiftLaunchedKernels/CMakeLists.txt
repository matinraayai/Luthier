cmake_minimum_required(VERSION 3.21)
project(LiftLaunchedKernels LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

add_library(LiftLaunchedKernels SHARED LiftLaunchedKernels.hip)



set_property(TARGET LiftLaunchedKernels PROPERTY COMPILE_FLAGS "-fPIC")

#target_include_directories(LiftLaunchedKernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(LiftLaunchedKernels PUBLIC LuthierTooling)