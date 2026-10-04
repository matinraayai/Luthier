add_llvm_pass_plugin(LuthierToolIRCompilationPlugin
        ToolIRCompilationPlugin.cpp
        $<TARGET_OBJECTS:LuthierToolIRCompilation>
        $<TARGET_OBJECTS:LuthierCommon>)

target_include_directories(LuthierToolIRCompilationPlugin PRIVATE
        "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")

target_link_libraries(LuthierToolIRCompilationPlugin PRIVATE
        LLVMSymbolize LLVMDebugInfoDWARF LLVMObject)

add_library(luthier::LuthierToolIRCompilationPlugin ALIAS LuthierToolIRCompilationPlugin)
