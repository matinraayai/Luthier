add_library(LuthierCommon OBJECT
        Cloning.cpp
        CodeGenHelpers.cpp
        Error.cpp
        ObjectUtils.cpp
)

add_dependencies(LuthierCommon LuthierAMDGPUTableGen)

target_include_directories(LuthierCommon PUBLIC ${CMAKE_SOURCE_DIR}/src/include)

target_include_directories(LuthierCommon PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_include_directories(LuthierCommon PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/luthier/amdgpu>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/include/luthier/amdgpu>"
)

luthier_add_position_independence(LuthierCommon)
luthier_add_llvm_compile_definitions(LuthierCommon)

target_link_libraries(LuthierCommon PUBLIC hip::host amd_comgr)

if (${LUTHIER_BUILD_UNIT_TESTS})
    add_subdirectory(unittest)
endif ()