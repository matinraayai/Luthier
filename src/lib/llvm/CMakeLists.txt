add_library(LuthierLLVM OBJECT
        Cloning.cpp
        CodeGenHelpers.cpp
        LLVMError.cpp
        streams.cpp
)

add_dependencies(LuthierLLVM LuthierAMDGPUTableGen)

target_compile_definitions(LuthierLLVM PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierLLVM
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(LuthierLLVM PRIVATE LuthierAMDGPU)