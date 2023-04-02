add_executable(editelf main.cpp)
set(CMAKE_BUILD_TYPE Debug)


target_link_libraries(editelf PRIVATE trampoline)
target_link_libraries(editelf PRIVATE sectiongenerator)