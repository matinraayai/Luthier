add_library(LuthierComgr OBJECT
        Comgr.cpp
        ComgrError.cpp
)

target_compile_definitions(LuthierComgr PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierComgr
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        ${amd_comgr_INCLUDE_DIRS}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${LLVM_INCLUDE_DIRS}
)
