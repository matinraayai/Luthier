include(AddLLVM)

add_subdirectory(EmbedIModulePlugin)
add_subdirectory(LegacyPlugin)
add_subdirectory(OptPlugin)

include(LuthierAddCompilerPlugin)