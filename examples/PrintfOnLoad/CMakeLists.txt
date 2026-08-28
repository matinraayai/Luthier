cmake_minimum_required(VERSION 3.21)

project(LuthierPrintfOnLoad LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED CONFIG)

luthier_create_offload_bundle(LuthierPrintfOnLoadObject PrintfOnLoad.hip
        DEVICE_OBJECT_LIBRARIES LuthierPrintfOnLoad_DEVICE_TARGETS)

target_link_libraries(LuthierPrintfOnLoadObject PUBLIC LuthierTooling hip::host)
foreach (_DEVICE_TARGET IN LISTS LuthierPrintfOnLoad_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

add_library(LuthierPrintfOnLoad SHARED $<TARGET_OBJECTS:LuthierPrintfOnLoadObject>)
target_link_libraries(LuthierPrintfOnLoad PUBLIC LuthierTooling hip::host)
