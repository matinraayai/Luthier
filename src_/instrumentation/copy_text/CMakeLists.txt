add_executable(cpytex main.cpp)

set(CMAKE_BUILD_TYPE Debug)

target_link_libraries(cpytex PRIVATE trampoline)
