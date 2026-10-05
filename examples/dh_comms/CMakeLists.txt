cmake_minimum_required(VERSION 3.21)

project(LuthierDhComms LANGUAGES CXX HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED CONFIG)

set(DH_COMMS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# The tool: hooks + device code (GPU) and the Luthier tool class (CPU).
luthier_create_offload_bundle(LuthierDhCommsObject src/DhCommsTool.hip
        DEVICE_OBJECT_LIBRARIES LuthierDhComms_DEVICE_TARGETS)

target_include_directories(LuthierDhCommsObject PRIVATE ${DH_COMMS_INCLUDE_DIR})
target_link_libraries(LuthierDhCommsObject PUBLIC LuthierTooling hip::host)
foreach (_DEVICE_TARGET IN LISTS LuthierDhComms_DEVICE_TARGETS)
    target_include_directories(${_DEVICE_TARGET} PRIVATE ${DH_COMMS_INCLUDE_DIR})
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::device)
endforeach ()

# The device slices compile a copy of the .hip (see LuthierCreateOffloadBundle)
# and get no compiler depfile, so editing a header would leave stale hook code
# inside the tool. Make the copy depend on our headers explicitly.
file(GLOB DH_COMMS_HEADERS CONFIGURE_DEPENDS ${DH_COMMS_INCLUDE_DIR}/dh_comms/*.h)
set_property(SOURCE
        ${CMAKE_CURRENT_BINARY_DIR}/LuthierDhCommsObject.dev_tu/DhCommsTool.hip
        APPEND PROPERTY OBJECT_DEPENDS ${DH_COMMS_HEADERS})

# CPU-only side of dh_comms: shared buffers, draining, message handling.
add_library(LuthierDhComms SHARED
        $<TARGET_OBJECTS:LuthierDhCommsObject>
        src/SharedBuffers.cpp
        src/MessageProcessor.cpp
        src/AccessIndexHandler.cpp
        src/CoalescingHandler.cpp)
target_include_directories(LuthierDhComms PRIVATE ${DH_COMMS_INCLUDE_DIR})
target_link_libraries(LuthierDhComms PUBLIC LuthierTooling hip::host)
