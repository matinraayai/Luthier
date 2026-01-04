include(AddLLVM)

add_subdirectory(EmbedIModulePlugin)
add_subdirectory(LuthierPassPlugin)
add_subdirectory(OptPlugin)

include(LuthierAddCompilerPlugin)