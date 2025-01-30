find_package(LLVM REQUIRED CONFIG)

include(AddLLVM)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(lib/EmbedIModulePlugin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#add_subdirectory(cmake/modules)

include(LuthierAddCompilerPlugin)