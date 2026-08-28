cmake_minimum_required(VERSION 3.21)

project(LuthierArgProbe LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED CONFIG)

# The tool library (offload bundle: host + device halves).
luthier_create_offload_bundle(LuthierArgProbeObject ArgProbe.hip
        DEVICE_OBJECT_LIBRARIES LuthierArgProbe_DEVICE_TARGETS)

target_link_libraries(LuthierArgProbeObject PUBLIC LuthierTooling hip::host)
foreach (_DEVICE_TARGET IN LISTS LuthierArgProbe_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

add_library(LuthierArgProbe SHARED $<TARGET_OBJECTS:LuthierArgProbeObject>)
target_link_libraries(LuthierArgProbe PUBLIC LuthierTooling hip::host)

# The tiny HIP driver program that runs a single instrumented kernel.
add_executable(LuthierArgProbeApp ArgProbeApp.hip)
target_link_libraries(LuthierArgProbeApp PRIVATE hip::host hip::device)
