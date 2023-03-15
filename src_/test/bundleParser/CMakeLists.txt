add_executable(parsebundle main.cpp)
set(CMAKE_BUILD_TYPE Debug)
target_link_libraries(parsebundle PRIVATE bundleParser)
