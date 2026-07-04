cmake_minimum_required(VERSION 3.21)

project(KernelInstrument LANGUAGES HIP CXX ASM C)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED)

luthier_create_offload_bundle(KernelInstrumentObject KernelInstrument.hip
        DEVICE_OBJECT_LIBRARIES KernelInstrument_DEVICE_TARGETS)

target_link_libraries(KernelInstrumentObject PUBLIC LuthierTooling hip::host LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS KernelInstrument_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

add_library(KernelInstrument SHARED $<TARGET_OBJECTS:KernelInstrumentObject>)
target_link_libraries(KernelInstrument PUBLIC LuthierTooling hip::host LLVMDemangle)
