add_library(LuthierLegacyPlugin STATIC
        LegacyPlugin.cpp
)

target_compile_definitions(LuthierLegacyPlugin PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierLegacyPlugin
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierLegacyPlugin PRIVATE LLVMSupport LuthierAMDGPU hip::host)