macro(luthier_add_position_independence target_name)
    get_target_property(TARGET_TYPE ${target_name} TYPE)
    if (TARGET_TYPE STREQUAL "SHARED" OR TARGET_TYPE STREQUAL "OBJECT")
        set_target_properties(${target_name} PROPERTIES
                POSITION_INDEPENDENT_CODE ON
        )
        # Adding the -fPIC flag just for good measure
        set_property(TARGET ${target_name} PROPERTY COMPILE_FLAGS "-fPIC")
    endif ()
endmacro()