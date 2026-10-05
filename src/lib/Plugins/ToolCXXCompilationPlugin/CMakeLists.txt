add_llvm_library(LuthierToolCXXCompilationPlugin MODULE
        ToolCXXCompilationPlugin.cpp
        $<TARGET_OBJECTS:LuthierToolCXXCompilation>)

target_compile_definitions(LuthierToolCXXCompilationPlugin
        PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierToolCXXCompilationPlugin PRIVATE
        "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CLANG_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")

target_link_libraries(LuthierToolCXXCompilationPlugin PRIVATE
        clangAST
        clangBasic
        clangFrontend
        clangLex
        clangSema)

set_target_properties(LuthierToolCXXCompilationPlugin
        PROPERTIES CXX_RTTI OFF)

add_library(luthier::LuthierToolCXXCompilationPlugin ALIAS LuthierToolCXXCompilationPlugin)
