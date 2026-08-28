cmake_minimum_required(VERSION 3.21)

project(LuthierReadWriteV0Probe LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED CONFIG)

luthier_create_offload_bundle(LuthierReadWriteV0ProbeObject ReadWriteV0Probe.hip
        DEVICE_OBJECT_LIBRARIES LuthierReadWriteV0Probe_DEVICE_TARGETS)

target_link_libraries(LuthierReadWriteV0ProbeObject PUBLIC LuthierTooling hip::host)
foreach (_DEVICE_TARGET IN LISTS LuthierReadWriteV0Probe_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

add_library(LuthierReadWriteV0Probe SHARED $<TARGET_OBJECTS:LuthierReadWriteV0ProbeObject>)
target_link_libraries(LuthierReadWriteV0Probe PUBLIC LuthierTooling hip::host)
