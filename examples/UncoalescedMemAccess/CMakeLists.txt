cmake_minimum_required(VERSION 3.21)

project(LuthierUncoalescedMemAccess LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

luthier_create_offload_bundle(LuthierUncoalescedMemAccessObject
        UncoalescedMemAccess.hip
        DEVICE_OBJECT_LIBRARIES LuthierUncoalescedMemAccess_DEVICE_TARGETS)

target_link_libraries(LuthierUncoalescedMemAccessObject PUBLIC luthier::LuthierTooling)
target_include_directories(LuthierUncoalescedMemAccessObject PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
foreach (_DEVICE_TARGET IN LISTS LuthierUncoalescedMemAccess_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE luthier::LuthierTooling)
    target_include_directories(${_DEVICE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach ()

add_library(LuthierUncoalescedMemAccess SHARED
        $<TARGET_OBJECTS:LuthierUncoalescedMemAccessObject>)
target_link_libraries(LuthierUncoalescedMemAccess PUBLIC luthier::LuthierTooling)
