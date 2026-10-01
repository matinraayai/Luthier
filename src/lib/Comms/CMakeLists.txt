
add_library(LuthierComms OBJECT
        dh_comms.cpp
        data_headers.cpp
        message.cpp
        message_handlers.cpp
        hip_runtime_loader.cpp
)

target_compile_definitions(LuthierComms PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierComms
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")