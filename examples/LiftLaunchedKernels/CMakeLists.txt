cmake_minimum_required(VERSION 3.21)

project(LiftLaunchedKernels LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

luthier_create_offload_bundle(LiftLaunchedKernelsObject LiftLaunchedKernels.hip
        DEVICE_OBJECT_LIBRARIES LiftLaunchedKernels_DEVICE_TARGETS)

target_link_libraries(LiftLaunchedKernelsObject PUBLIC luthier::LuthierTooling)
foreach (_DEVICE_TARGET IN LISTS LiftLaunchedKernels_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE luthier::LuthierTooling)
endforeach ()

add_library(LiftLaunchedKernels SHARED $<TARGET_OBJECTS:LiftLaunchedKernelsObject>)
target_link_libraries(LiftLaunchedKernels PUBLIC luthier::LuthierTooling)
