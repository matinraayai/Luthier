set(SRC 
src/assembler.cpp
src/bitops.cpp
src/encodetable.cpp
src/initialize.cpp 
src/operand.cpp)
add_library(assembler ${SRC})
set(CMAKE_BUILD_TYPE Debug)
target_include_directories(assembler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")