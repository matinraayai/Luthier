# Install the headers
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
# Create a basic version file and its installation rule
write_basic_package_version_file(
        ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsVersion.cmake
        VERSION "${LuthierCompilerPlugins_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

install(FILES ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierCompilerPlugins/)

# Create the package configuration file and its installation rule
configure_file(
        ${LuthierCompilerPlugins_SOURCE_DIR}/cmake/LuthierCompilerPluginsConfig.cmake.in
        ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsConfig.cmake @ONLY)

install(FILES ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierCompilerPlugins/)

# Install IModule Embed targets and export them
install(TARGETS IModuleEmbedPlugin Intrinsics DESTINATION ${CMAKE_INSTALL_LIBDIR} EXPORT
        LuthierCompilerPluginsTargets)

install(EXPORT LuthierCompilerPluginsTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierCompilerPlugins/
        NAMESPACE luthier::)

export(EXPORT LuthierCompilerPluginsTargets
        FILE "${CMAKE_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsTargets.cmake"
        NAMESPACE luthier::
)

# Install and export the modules under the cmake directory
configure_file(${CMAKE_SOURCE_DIR}/compiler-plugins/cmake/modules/LuthierAddCompilerPlugin.cmake
        ${CMAKE_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierAddCompilerPlugin.cmake COPYONLY)

install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierAddCompilerPlugin.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierCompilerPlugins/)