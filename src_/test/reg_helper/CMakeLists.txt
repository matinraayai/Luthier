add_executable(regNum main.cpp)
set(CMAKE_BUILD_TYPE Debug)
target_link_libraries(regNum PRIVATE disassembler)