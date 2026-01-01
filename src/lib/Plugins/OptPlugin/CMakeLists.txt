add_llvm_pass_plugin(LuthierOptPlugin
        OptPlugin.cpp
        $<TARGET_OBJECTS:LuthierToolingCommon>
)

target_include_directories(LuthierOptPlugin PRIVATE
        "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")


