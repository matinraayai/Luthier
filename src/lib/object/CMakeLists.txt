add_library(LuthierObject OBJECT
        ObjectFileUtils.cpp
)

target_compile_definitions(LuthierObject PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierObject
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
)
