include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/main.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
# add_subdirectory(irtranslator)
add_subdirectory(svatest)
add_subdirectory(singleton)
add_subdirectory(stacktrace)
add_subdirectory(error)
add_subdirectory(tooldevicecodeparser)
add_subdirectory(densemapinfo)
add_subdirectory(rocprofiler)
add_subdirectory(debug)
add_subdirectory(kfd)