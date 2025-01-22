if (${LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG})
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hip_intercept_gen.py
            --hip-api-trace-path ${hip_INCLUDE_DIR}/hip/amd_detail/hip_api_trace.hpp
            --hpp-structs-save-path ${CMAKE_BINARY_DIR}/include/luthier/hip/TraceApi.h
            --cpp-compiler-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HipCompilerApiInterceptor.cpp
            --cpp-runtime-implementation-save-path ${CMAKE_CURRENT_BINARY_DIR}/HipRuntimeApiInterceptor.cpp
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/HipCompilerApiInterceptor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/HipRuntimeApiInterceptor.cpp
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
)

target_include_directories(LuthierHIP PUBLIC "${CMAKE_SOURCE_DIR}/src/include")

target_include_directories(LuthierHIP PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

luthier_add_position_independence(LuthierHIP)
luthier_add_llvm_compile_definitions(LuthierHIP)

target_link_libraries(LuthierHIP PUBLIC hip::host)