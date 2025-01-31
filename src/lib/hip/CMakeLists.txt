if (${LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG})
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hip_intercept_gen.py
            --hip-api-trace-path ${hip_INCLUDE_DIR}/hip/amd_detail/hip_api_trace.hpp
            --hpp-structs-save-path ${CMAKE_BINARY_DIR}/include/luthier/hip/TraceApi.h
            --cpp-compiler-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HipCompilerApiInterceptor.cpp
            --cpp-runtime-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HipRuntimeApiInterceptor.cpp
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/HipCompilerApiInterceptor.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/HipRuntimeApiInterceptor.cpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hip_intercept_gen.py
        --hip-api-trace-path ${hip_INCLUDE_DIR}/hip/amd_detail/hip_api_trace.hpp
        --hpp-structs-save-path ${CMAKE_BINARY_DIR}/include/luthier/hip/TraceApi.h
        --cpp-compiler-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HipCompilerApiInterceptor.cpp
        --cpp-runtime-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HipRuntimeApiInterceptor.cpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
        COMMENT "Generating HIP API interceptor implementation")

add_library(LuthierHIP OBJECT
        ${CMAKE_CURRENT_BINARY_DIR}/HipCompilerApiInterceptor.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/HipRuntimeApiInterceptor.cpp
        HipError.cpp
)

target_compile_definitions(LuthierHIP PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierHIP
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(LuthierHIP PRIVATE hip::host)