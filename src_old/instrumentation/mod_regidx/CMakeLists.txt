add_executable(modidx main.cpp)
set(CMAKE_BUILD_TYPE Debug)


target_link_libraries(modidx PRIVATE assembler)

target_link_libraries(modidx PRIVATE disassembler)