add_executable(cpytex main.cpp)
set(CMAKE_BUILD_TYPE Debug)
add_library(bitops ../../lib/src/bitops.cpp)

target_link_libraries(cpytex PRIVATE assembler)
target_link_libraries(cpytex PRIVATE bitops)
target_link_libraries(cpytex PRIVATE disassembler)
