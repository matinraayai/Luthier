add_library(LuthierLinker OBJECT
        Linker.cpp
)

target_compile_definitions(LuthierLinker PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierLinker
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${LLD_INCLUDE_DIRS}
)
