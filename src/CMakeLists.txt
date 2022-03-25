cmake_minimum_required(VERSION 3.2)

# Subdirectories
set(SUB_DIRS)
foreach (DIR ${SUB_DIRS})
  add_subdirectory(${DIR})
endforeach (DIR)

# The library that is shared by the driver and runtime implementation
set(SRC
    internal.h
    internal.cc
)
add_library(sibir SHARED ${SRC})
set_target_properties(sibir
    PROPERTIES VERSION 0.0.1)
set_property(TARGET sibir
    PROPERTY COMPILE_FLAGS "-fPIC")
target_link_libraries(sibir hip::host)

# Enable C++ 17
target_compile_features(sibir PRIVATE cxx_std_17)
