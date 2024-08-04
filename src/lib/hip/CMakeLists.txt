if (${LUTHIER_GEN_INTERCEPTOR_IMPL_DURING_CONFIG})
    execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hip_intercept_gen.py
            --hip-api-trace-path ${hip_INCLUDE_DIR}/hip/amd_detail/hip_api_trace.hpp
            --hpp-structs-save-path ${CMAKE_SOURCE_DIR}/include/luthier/hip_trace_api.h
            --cpp-compiler-callback-save-path ${CMAKE_CURRENT_SOURCE_DIR}/hip_compiler_intercept.cpp
            --cpp-runtime-callback-save-path ${CMAKE_CURRENT_SOURCE_DIR}/hip_runtime_intercept.cpp
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
endif ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/hip_compiler_intercept.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_runtime_intercept.cpp
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/hip_intercept_gen.py
        --hip-api-trace-path ${hip_INCLUDE_DIR}/hip/amd_detail/hip_api_trace.hpp
        --hpp-structs-save-path ${CMAKE_SOURCE_DIR}/include/luthier/hip_trace_api.h
        --cpp-compiler-callback-save-path ${CMAKE_CURRENT_SOURCE_DIR}/hip_compiler_intercept.cpp
        --cpp-runtime-callback-save-path ${CMAKE_CURRENT_SOURCE_DIR}/hip_runtime_intercept.cpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
        COMMENT "Generating HIP API interceptor implementation")


add_library(luthier_hip STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_compiler_intercept.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_runtime_intercept.cpp
)

target_include_directories(luthier_hip PUBLIC ${LUTHIER_PRIVATE_INCLUDE_DIR})

target_include_directories(luthier_hip PUBLIC ${LUTHIER_PUBLIC_INCLUDE_DIR})


target_compile_options(luthier_hip PUBLIC "-fPIC")
add_llvm_compile_definitions(luthier_hip)

target_link_libraries(luthier_hip PUBLIC hip::host LLVMCore LLVMSupport)