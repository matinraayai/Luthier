add_library(LuthierHIP OBJECT
        HipError.cpp
)

target_compile_definitions(LuthierHIP PRIVATE ${LLVM_DEFINITIONS} ${hip_DEFINES})

target_include_directories(LuthierHIP
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hip_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)