add_executable(cpytex main.cpp)

set(TRMP_SRC
trampoline.cpp)

add_library(trampoline ${TRMP_SRC})

set(CMAKE_BUILD_TYPE Debug)

target_link_libraries(trampoline PUBLIC assembler)
target_link_libraries(trampoline PUBLIC bitops)
target_link_libraries(trampoline PUBLIC disassembler)
target_link_libraries(cpytex PRIVATE trampoline)
