cmake_minimum_required(VERSION 3.21)

project(LuthierOpcodeHistogram LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

luthier_create_offload_bundle(LuthierOpcodeHistogramObject OpcodeHistogram.hip
        DEVICE_OBJECT_LIBRARIES LuthierOpcodeHistogram_DEVICE_TARGETS)

target_link_libraries(LuthierOpcodeHistogramObject PUBLIC luthier::LuthierTooling LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS LuthierOpcodeHistogram_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE luthier::LuthierTooling)
endforeach ()

add_library(LuthierOpcodeHistogram SHARED $<TARGET_OBJECTS:LuthierOpcodeHistogramObject>)
target_link_libraries(LuthierOpcodeHistogram PUBLIC luthier::LuthierTooling LLVMDemangle)
