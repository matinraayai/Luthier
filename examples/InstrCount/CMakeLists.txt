cmake_minimum_required(VERSION 3.21)

project(LuthierInstrCount LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

luthier_create_offload_bundle(LuthierInstrCountObject InstrCount.hip
        DEVICE_OBJECT_LIBRARIES LuthierInstrCount_DEVICE_TARGETS)

target_link_libraries(LuthierInstrCountObject PUBLIC luthier::LuthierTooling)
foreach (_DEVICE_TARGET IN LISTS LuthierInstrCount_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE luthier::LuthierTooling)
endforeach ()

add_library(LuthierInstrCount SHARED $<TARGET_OBJECTS:LuthierInstrCountObject>)
target_link_libraries(LuthierInstrCount PUBLIC luthier::LuthierTooling)
