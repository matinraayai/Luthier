add_subdirectory(copy_text)
add_subdirectory(mod_regidx)
add_subdirectory(mod_elf)

add_executable(instru main.cpp)

target_link_libraries(instru PRIVATE bundleParser)
target_link_libraries(instru PRIVATE disassembler)