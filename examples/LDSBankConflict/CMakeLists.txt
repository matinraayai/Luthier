cmake_minimum_required(VERSION 3.21)

project(LuthierLDSBankConflict LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED)

luthier_create_offload_bundle(LuthierLDSBankConflictObject LDSBankConflict.hip
        DEVICE_OBJECT_LIBRARIES LuthierLDSBankConflict_DEVICE_TARGETS)

target_link_libraries(LuthierLDSBankConflictObject PUBLIC LuthierTooling hip::host LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS LuthierLDSBankConflict_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

add_library(LuthierLDSBankConflict SHARED $<TARGET_OBJECTS:LuthierLDSBankConflictObject>)
target_link_libraries(LuthierLDSBankConflict PUBLIC LuthierTooling hip::host LLVMDemangle)
