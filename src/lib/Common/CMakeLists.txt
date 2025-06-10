add_library(LuthierCommon OBJECT
        GenericLuthierError.cpp
        LuthierError.cpp
        ROCmLibraryError.cpp
)

target_compile_definitions(LuthierCommon PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierCommon
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
)
