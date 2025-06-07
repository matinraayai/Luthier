add_library(LuthierHIP OBJECT
        HipError.cpp
)

target_compile_definitions(LuthierHIP PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierHIP
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierHIP PRIVATE hip::host)