cmake_minimum_required(VERSION 3.2)
project(sibir)

set(CMAKE_CXX_STANDARD 17)

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
# Find hip
find_package(hip)
# Create the library
add_library(sibir SHARED sibir_impl.hpp sibir_impl.cpp sibir.h sibir_meta.hpp sibir_meta.cpp)

#set_property(TARGET sibir
#        PROPERTY COMPILE_FLAGS "-fPIC -g")

#set_property(TARGET sibir PROPERTY LINK_FLAGS "-fPIC")
# Link with HIP
target_link_libraries(sibir hip::host hip::device)

target_include_directories(sibir PUBLIC .)