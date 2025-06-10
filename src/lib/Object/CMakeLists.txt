add_library(LuthierObject OBJECT
        ObjectUtils.cpp
)

target_compile_definitions(LuthierObject PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierObject
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
)
