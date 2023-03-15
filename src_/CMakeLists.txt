cmake_minimum_required(VERSION 3.2)

project(sibira)

add_subdirectory(bundleParser)
add_subdirectory(instrumentation)
add_subdirectory(lib)
add_subdirectory(test/reg_helper)
add_subdirectory(test/bundleParser)
add_subdirectory(test/test_disassembler)
add_subdirectory(test/test_assembler)
