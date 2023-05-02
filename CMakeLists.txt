cmake_minimum_required(VERSION 3.2)

set(CMAKE_C_COMPILER "/opt/rocm/llvm/bin/clang")     # Dudes on stack overflow HATE defining compiler in CMakeLists
set(CMAKE_CXX_COMPILER "/opt/rocm/llvm/bin/clang++")

project(sibir)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(sibir)
add_subdirectory(examples)

# Include other files
# list(APPEND CMAKE_MODULE_PATH
#   "${PROJECT_SOURCE_DIR}/cmake")
# message(${CMAKE_MODULE_PATH})
# include(hip)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS "-fPIC")
# # Add warning related flags
# SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")

# # Put results in bin/lib folder
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# # Inlcuding from root as default
# include_directories(${CMAKE_BINARY_DIR})
# include_directories(${CMAKE_SOURCE_DIR})
# include_directories(${CMAKE_SOURCE_DIR}/include)

# add_definitions(-D__HIP_PLATFORM_HCC__)
# add_definitions(-D_GLIBCXX_DEBUG)

# # Include sub directories
# set(SUB_DIRS src src_/lib)
# foreach (DIR ${SUB_DIRS})
#   add_subdirectory(${DIR})
# endforeach (DIR)

# add_custom_target(sibir-cli ALL
#   COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/sibir.py ${CMAKE_BINARY_DIR}/sibir
#   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sibir.py
#   COMMENT "Move sibir command to output folder")

