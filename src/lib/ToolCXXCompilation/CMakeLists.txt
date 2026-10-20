add_library(LuthierToolCXXCompilation OBJECT
        Attributes.cpp
        Consumers.cpp)

target_compile_definitions(LuthierToolCXXCompilation
        PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierToolCXXCompilation
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${CLANG_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_compile_options(LuthierToolCXXCompilation
        PUBLIC ${LUTHIER_RTTI_FLAG})
