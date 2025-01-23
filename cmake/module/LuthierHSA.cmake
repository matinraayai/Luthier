include_guard(GLOBAL)

# Macro to add HSA as a dependency to a target
macro(luthier_add_hsa_compile_definitions target_name)
    # Some HSA include files won't work unless AMD internal build is defined
    add_compile_definitions(${target_name} PUBLIC AMD_INTERNAL_BUILD)
endmacro()