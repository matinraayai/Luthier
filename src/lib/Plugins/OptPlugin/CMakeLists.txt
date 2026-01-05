add_llvm_pass_plugin(LuthierOptPlugin
        OptPlugin.cpp
        $<TARGET_OBJECTS:LuthierToolingCommon>
)

target_compile_definitions(LuthierOptPlugin PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierOptPlugin PRIVATE
        ${hsa-runtime64_INCLUDE_DIRS}
        "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")

target_link_libraries(LuthierOptPlugin PRIVATE LuthierAMDGPU)