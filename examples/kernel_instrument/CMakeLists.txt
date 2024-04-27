cmake_minimum_required(VERSION 3.2)
project(kernel_instrument LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(hip)

#list(APPEND CMAKE_MODULE_PATH ../../src/luthier)

add_library(kernel_instrument SHARED kernel_instrument.cpp)

set_source_files_properties(kernel_instrument.cpp PROPERTIES LANGUAGE HIP)

set_property(TARGET kernel_instrument PROPERTY COMPILE_FLAGS "-fPIC")

target_include_directories(kernel_instrument PUBLIC ../../include/)
#target_link_directories(kernel_instrument PUBLIC ../)

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xoffload-linker --emit-relocs")

#set(CMAKE_HIP_LINK_LIBRARY_FLAG "${CMAKE_HIP_LINK_LIBRARY_FLAG} -Xoffload-linker --emit-relocs")
#target_compile_options(kernel_instrument PUBLIC "-Xoffload-linker --emit-relocs")

target_link_libraries(kernel_instrument PUBLIC luthier hip::host hip::device)