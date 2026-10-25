add_library(LuthierToolIRCompilation OBJECT
        MarkAnnotationsPass.cpp
        FinalizeIntrinsicsPass.cpp
        ToolDeviceCodeOffloadParserPass.cpp
)

target_compile_definitions(LuthierToolIRCompilation
        PRIVATE ${LLVM_DEFINITIONS})

target_compile_options(LuthierToolIRCompilation PUBLIC ${LUTHIER_RTTI_FLAG})


target_include_directories(LuthierToolIRCompilation
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
