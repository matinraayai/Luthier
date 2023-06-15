cmake_minimum_required(VERSION 3.2)
project(disassembler)

set(CMAKE_CXX_STANDARD 20)
add_library(disassembler
            bitops.h
            bitops.cpp
            decodetable.cpp
            assembler.h
            assembler.cpp
            disassembler.cpp
            inst.h
            inst.cpp
            instprinter.h
            initialize.cpp
            operand.cpp
            elf.cc
            bitops.cpp
        reg.cpp)
set(CMAKE_BUILD_TYPE Debug)