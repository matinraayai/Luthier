cmake_minimum_required(VERSION 3.21)

project(LiftLaunchedKernels LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

luthier_create_offload_bundle(LiftLaunchedKernels LiftLaunchedKernels.hip
        DEVICE_OBJECT_LIBRARIES LiftLaunchedKernels_DEVICE_TARGETS)

target_link_libraries(LiftLaunchedKernels PUBLIC LuthierTooling hip::host)
foreach (_DEVICE_TARGET IN LISTS LiftLaunchedKernels_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::host)
endforeach ()
