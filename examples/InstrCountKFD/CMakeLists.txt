cmake_minimum_required(VERSION 3.21)

project(LuthierInstrCountKFD LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED CONFIG)

# Built exactly as ../InstrCount is, and the HIP compilation is not optional: the
# hooks are device functions, and ToolDeviceCodeOffloadParserTrait's static
# fields come from a macro whose __attribute__((managed)) requires a HIP
# translation unit.
luthier_create_offload_bundle(LuthierInstrCountKFDObject InstrCountKFD.hip
        DEVICE_OBJECT_LIBRARIES LuthierInstrCountKFD_DEVICE_TARGETS)

target_link_libraries(LuthierInstrCountKFDObject PUBLIC LuthierTooling hip::host)
foreach (_DEVICE_TARGET IN LISTS LuthierInstrCountKFD_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

add_library(LuthierInstrCountKFD SHARED
        $<TARGET_OBJECTS:LuthierInstrCountKFDObject>)
target_link_libraries(LuthierInstrCountKFD PUBLIC LuthierTooling hip::host)
