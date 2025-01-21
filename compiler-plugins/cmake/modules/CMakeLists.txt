include(CMakePackageConfigHelpers)
# Create a basic version file and its installation rule
write_basic_package_version_file(
        ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsVersion.cmake
        VERSION "${LuthierCompilerPlugins_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

install(FILES ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsVersion.cmake
        DESTINATION lib/cmake/LuthierCompilerPlugins/)

# Create the package configuration file and its installation rule
configure_file(
        ${LuthierCompilerPlugins_SOURCE_DIR}/cmake/LuthierCompilerPluginsConfig.cmake.in
        ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsConfig.cmake @ONLY)

install(FILES ${LuthierCompilerPlugins_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsConfig.cmake
        DESTINATION lib/cmake/LuthierCompilerPlugins/)

# Install IModule Embed targets and export them
install(TARGETS LuthierIModuleEmbedPlugin DESTINATION lib/ EXPORT
        LuthierCompilerPluginsTargets)

install(EXPORT LuthierCompilerPluginsTargets DESTINATION lib/cmake/LuthierCompilerPlugins/
        NAMESPACE LuthierCompilerPlugins::)

export(EXPORT LuthierCompilerPluginsTargets
        FILE "${CMAKE_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsTargets.cmake"
        NAMESPACE LuthierCompilerPlugins::
)

# Install and export the modules under the cmake directory
configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/LuthierAddCompilerPlugin.cmake
               ${CMAKE_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierAddCompilerPlugin.cmake COPYONLY)

install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierAddCompilerPlugin.cmake
        DESTINATION lib/cmake/LuthierCompilerPlugins/)