add_library(LuthierCommon OBJECT
        Cloning.cpp
        CodeGenHelpers.cpp
        Error.cpp
        ObjectUtils.cpp
        streams.cpp
)

add_dependencies(LuthierCommon LuthierAMDGPUTableGen)

target_compile_definitions(LuthierCommon PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierCommon
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${amd_comgr_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(LuthierCommon PRIVATE LuthierAMDGPU)

if (${LUTHIER_BUILD_UNIT_TESTS})
    add_subdirectory(unittest)
endif ()