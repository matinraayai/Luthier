if (${LLVM_LUTHIERIMODULEEMBEDPLUGIN_LINK_INTO_TOOLS})
    message(WARNING "Setting LLVM_LUTHIERIMODULEEMBEDPLUGIN_LINK_INTO_TOOLS=ON is not supported")
endif ()

add_llvm_pass_plugin(IModuleEmbedPlugin
        EmbedInstrumentationModuleBitcodePass.hpp
        EmbedInstrumentationModuleBitcodePass.cpp)

target_include_directories(IModuleEmbedPlugin PRIVATE "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>")


