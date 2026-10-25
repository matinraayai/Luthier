add_library(LuthierPassPlugin OBJECT
        LuthierPassPlugin.cpp
)

target_compile_definitions(LuthierPassPlugin PRIVATE ${LLVM_DEFINITIONS})

target_compile_options(LuthierPassPlugin
        PUBLIC ${LUTHIER_RTTI_FLAG})

target_include_directories(LuthierPassPlugin
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)