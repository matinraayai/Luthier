set(ASM_SRC
src/assembler.cpp
src/bitops.cpp
src/encodetable.cpp
src/initialize.cpp 
src/operand.cpp)

set(DISASM_SRC 
src/initialize.cpp 
src/operand.cpp 
src/disassembler.cpp 
src/elf.cc 
src/bitops.cpp 
src/decodetable.cpp)

set(BITOPS_SRC
src/bitops.cpp)

set(TRMP_SRC
src/trampoline.cpp)

add_library(assembler ${ASM_SRC})
add_library(disassembler ${DISASM_SRC})
add_library(bitops ${BITOPS_SRC})
add_library(trampoline ${TRMP_SRC})

set(CMAKE_BUILD_TYPE Debug)

target_include_directories(assembler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(disassembler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(bitops PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(trampoline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(trampoline PUBLIC assembler)
target_link_libraries(trampoline PUBLIC disassembler)
target_link_libraries(trampoline PUBLIC bitops)
