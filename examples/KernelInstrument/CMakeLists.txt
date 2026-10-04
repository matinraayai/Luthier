cmake_minimum_required(VERSION 3.21)

project(KernelInstrument LANGUAGES HIP CXX ASM C)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

luthier_create_offload_bundle(KernelInstrumentObject KernelInstrument.hip
        DEVICE_OBJECT_LIBRARIES KernelInstrument_DEVICE_TARGETS)

target_link_libraries(KernelInstrumentObject PUBLIC luthier::LuthierTooling LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS KernelInstrument_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE luthier::LuthierTooling)
endforeach ()

add_library(KernelInstrument SHARED $<TARGET_OBJECTS:KernelInstrumentObject>)
target_link_libraries(KernelInstrument PUBLIC luthier::LuthierTooling LLVMDemangle)
