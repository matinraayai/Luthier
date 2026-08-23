# KFD-level AQL queue interception.
#
# Two targets on purpose:
#
#   LuthierKFD                  the logic, as an OBJECT library. Can be linked
#                               into tests or into Luthier without replacing
#                               ioctl for the whole process.
#   luthier-kfd-queue-wrapper   the preloadable shared library. Adds the ioctl
#                               interposer, which is the only symbol that has to
#                               be interposed.
#
# Keeping them apart means the interposer is opt-in: you get it by preloading
# this library, not merely by linking the logic.

add_library(LuthierKFD OBJECT
        QueueWrapper.cpp
)

target_compile_definitions(LuthierKFD PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierKFD
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${hsa-runtime64_INCLUDE_DIRS}
)

# Required: the objects are linked into a shared library below.
set_property(TARGET LuthierKFD PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(luthier-kfd-queue-wrapper SHARED
        Interposer.cpp
        $<TARGET_OBJECTS:LuthierKFD>
)

target_compile_definitions(luthier-kfd-queue-wrapper PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(luthier-kfd-queue-wrapper
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${hsa-runtime64_INCLUDE_DIRS}
)

# dlsym for finding the real ioctl; pthreads for the polling thread.
target_link_libraries(luthier-kfd-queue-wrapper PRIVATE ${CMAKE_DL_LIBS} pthread)
