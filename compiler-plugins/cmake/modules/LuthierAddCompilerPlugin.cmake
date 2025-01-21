macro(luthier_add_compiler_plugin target compiler_plugin)
    add_dependencies(${target} ${compiler_plugin})
    target_compile_options(${target}
            PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fpass-plugin=$<TARGET_FILE:${compiler_plugin}>>
    )
endmacro()