macro(luthier_add_llvm_compile_definitions target_name)
    target_include_directories(${target_name} PUBLIC ${LLVM_INCLUDE_DIRS})

    target_compile_definitions(${target_name} PUBLIC ${LLVM_DEFINITIONS})
endmacro()