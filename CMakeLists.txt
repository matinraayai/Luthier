cmake_minimum_required(VERSION 3.2)

project(sibir)

# Include other files
list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/cmake")
message(${CMAKE_MODULE_PATH})



# Add warning related flags
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

# Put results in bin/lib folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Inlcuding from root as default
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_definitions(-D__HIP_PLATFORM_HCC__)
add_definitions(-D_GLIBCXX_DEBUG)

# Include sub directories
set(SUB_DIRS src)
foreach (DIR ${SUB_DIRS})
  add_subdirectory(${DIR})
endforeach (DIR)

add_custom_target(sibir-cli ALL
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/sibir.py ${CMAKE_BINARY_DIR}/sibir
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sibir.py
  COMMENT "Move sibir command to output folder")



