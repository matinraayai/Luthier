add_library(LuthierLLVM OBJECT
        LLVMError.cpp
)

target_compile_definitions(LuthierLLVM PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierLLVM
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
)