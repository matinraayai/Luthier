macro(luthier_add_include_dirs target_name)
    target_include_directories(${target_name} PRIVATE
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/include>"
    )
    target_include_directories(${target_name} INTERFACE
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )
endmacro()