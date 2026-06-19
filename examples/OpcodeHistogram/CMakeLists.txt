cmake_minimum_required(VERSION 3.21)

project(LuthierOpcodeHistogram LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED)

luthier_create_offload_bundle(LuthierOpcodeHistogram OpcodeHistogram.hip
        DEVICE_OBJECT_LIBRARIES LuthierOpcodeHistogram_DEVICE_TARGETS)

target_link_libraries(LuthierOpcodeHistogram PUBLIC LuthierTooling hip::host LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS LuthierOpcodeHistogram_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::host)
endforeach ()
