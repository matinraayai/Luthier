cmake_minimum_required(VERSION 3.21)

project(LuthierLDSBankConflict LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

luthier_create_offload_bundle(LuthierLDSBankConflictObject LDSBankConflict.hip
        DEVICE_OBJECT_LIBRARIES LuthierLDSBankConflict_DEVICE_TARGETS)

target_link_libraries(LuthierLDSBankConflictObject PUBLIC luthier::LuthierTooling LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS LuthierLDSBankConflict_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE luthier::LuthierTooling)
endforeach ()

add_library(LuthierLDSBankConflict SHARED $<TARGET_OBJECTS:LuthierLDSBankConflictObject>)
target_link_libraries(LuthierLDSBankConflict PUBLIC luthier::LuthierTooling LLVMDemangle)
