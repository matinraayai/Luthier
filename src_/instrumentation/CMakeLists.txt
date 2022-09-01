add_executable(instru main.cpp)

target_link_libraries(instru PRIVATE bundleParser)
target_link_libraries(instru PRIVATE disassembler)