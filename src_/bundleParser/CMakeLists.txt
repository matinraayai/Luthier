add_library(
	bundleParser
	parser.cpp
	elf.cc
)
target_include_directories(bundleParser PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE Debug)