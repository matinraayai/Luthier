add_executable(asm_test main.cpp)
set(CMAKE_BUILD_TYPE Debug)
target_link_libraries(asm_test PRIVATE assembler)