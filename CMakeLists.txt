cmake_minimum_required(VERSION 3.2)

project(src/sibir)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-fPIC")
# Add warning related flags
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

# Put results in bin/lib folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Inlcuding from root as default
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_definitions(-D_GLIBCXX_DEBUG)

# Include sub directories
set(SUB_DIRS src/sibir/ examples/kernel_instrument)
foreach (DIR ${SUB_DIRS})
  add_subdirectory(${DIR})
endforeach (DIR)

#add_custom_target(sibir-cli ALL
#  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/sibir.py ${CMAKE_BINARY_DIR}/sibir
#  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sibir.py
#  COMMENT "Move sibir command to output folder")



