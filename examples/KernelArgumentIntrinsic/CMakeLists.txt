cmake_minimum_required(VERSION 3.21)

project(LuthierKernelArgumentIntrinsic LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

luthier_create_offload_bundle(LuthierKernelArgumentIntrinsicObject KernelArgumentIntrinsic.hip
        DEVICE_OBJECT_LIBRARIES LuthierKernelArgumentIntrinsic_DEVICE_TARGETS)

target_link_libraries(LuthierKernelArgumentIntrinsicObject PUBLIC luthier::LuthierTooling LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS LuthierKernelArgumentIntrinsic_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE luthier::LuthierTooling)
endforeach ()

add_library(LuthierKernelArgumentIntrinsic SHARED $<TARGET_OBJECTS:LuthierKernelArgumentIntrinsicObject>)
target_link_libraries(LuthierKernelArgumentIntrinsic PUBLIC luthier::LuthierTooling LLVMDemangle)
