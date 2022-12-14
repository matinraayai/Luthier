add_executable(cpytex main.cpp)

add_library(bitops ../../disassembler/lib/src/bitops.cpp)

target_link_libraries(cpytex PRIVATE assembler)
target_link_libraries(cpytex PRIVATE bitops)
target_link_libraries(cpytex PRIVATE disassembler)