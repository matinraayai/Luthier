add_library(LuthierComms OBJECT
        SharedBuffers.cpp
)

target_compile_definitions(LuthierComms PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS} ${hip_COMPILE_DEFINITIONS})

target_include_directories(LuthierComms
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        ${hip_INCLUDE_DIRS}
)
