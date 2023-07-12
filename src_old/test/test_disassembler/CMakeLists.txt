add_executable(disasm main.cpp)
set(CMAKE_BUILD_TYPE Debug)
target_link_libraries(disasm PRIVATE disassembler)