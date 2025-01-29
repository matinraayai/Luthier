add_library(LuthierComgr OBJECT
        ComgrError.cpp
)

target_compile_definitions(LuthierComgr PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierComgr
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${amd_comgr_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
)
