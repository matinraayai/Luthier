find_package(LLVM REQUIRED CONFIG)

include(AddLLVM)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(lib/EmbedIModulePlugin)

include(LuthierAddCompilerPlugin)