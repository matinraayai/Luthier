add_executable(cloning_test cloning_test.cpp)

target_link_libraries(
        cloning_test
        PUBLIC
        doctest::doctest
        luthier_common
)

