set(SRC 
src/initialize.cpp 
src/operand.cpp 
src/disassembler.cpp 
src/elf.cc 
src/bitops.cpp 
src/decodetable.cpp)
add_library(disassembler ${SRC})
set(CMAKE_BUILD_TYPE Debug)
target_include_directories(disassembler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")