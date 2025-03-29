install(DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

include(CMakePackageConfigHelpers)

# Write & Install Version Cmake File
write_basic_package_version_file(
        ${luthier_BINARY_DIR}/lib/cmake/luthier/luthierVersion.cmake
        VERSION "${luthier_VERSION}"
        COMPATIBILITY AnyNewerVersion
)
install(FILES ${luthier_BINARY_DIR}/lib/cmake/luthier/luthierVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

# Set luthier SRC DIR variable
set(luthier_SOURCE_DIR ${CMAKE_SOURCE_DIR})

# Write & Install Config File
configure_file(
        ${luthier_SOURCE_DIR}/cmake/luthierConfig.cmake.in
        ${luthier_BINARY_DIR}/lib/cmake/luthier/luthierConfig.cmake @ONLY)

# Change destination to /cmake/luthier
install(FILES ${luthier_BINARY_DIR}/lib/cmake/luthier/luthierConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

# Install luthier binaries
install(DIRECTORY ${CMAKE_BINARY_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

# Install & Export Targets
install(TARGETS LuthierAMDGPU LuthierTooling
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT luthierTargets)

install(TARGETS luthier-tblgen
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

# Install & Export luthierTargets to proper location
install(EXPORT luthierTargets 
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

export(EXPORT luthierTargets
        FILE "${CMAKE_BINARY_DIR}/lib/cmake/luthier/luthierTargets.cmake")