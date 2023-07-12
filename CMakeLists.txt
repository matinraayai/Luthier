cmake_minimum_required(VERSION 3.2)

project(sibir-top)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-fPIC")
# Add warning related flags
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

# Put results in bin/lib folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Including from root as default
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_definitions(-D_GLIBCXX_DEBUG)

# Include sub directories
set(SUB_DIRS src/ examples/kernel_instrument)
foreach (DIR ${SUB_DIRS})
  add_subdirectory(${DIR})
endforeach (DIR)



