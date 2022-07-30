cmake_minimum_required(VERSION 3.2)
project(Disassembler VERSION 1.0)

set(SRC initialize.cpp operand.cpp disassembler.cpp main.cpp elf.cc bitops.cpp decodetable.cpp)

add_executable(disassembler ${SRC})
set(CMAKE_BUILD_TYPE Debug)