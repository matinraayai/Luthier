add_executable(instru main.cpp)
add_executable(kernel_copy copytextsec.cpp)

target_link_libraries(instru PRIVATE bundleParser)
target_link_libraries(instru PRIVATE disassembler)

target_link_libraries(kernel_copy PRIVATE assembler)
target_link_libraries(kernel_copy PRIVATE disassembler)
