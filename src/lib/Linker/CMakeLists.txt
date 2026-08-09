add_library(LuthierLinker OBJECT
        Linker.cpp
)

# Pass the LLVM tools directory for locating lld at runtime
target_compile_definitions(LuthierLinker PRIVATE ${LLVM_DEFINITIONS}
        LUTHIER_LLVM_TOOLS_BINARY_DIR="${LLVM_TOOLS_BINARY_DIR}")

target_include_directories(LuthierLinker
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
)
