include(CMakePackageConfigHelpers)

# Write & Install Version File
write_basic_package_version_file(
        ${CMAKE_BINARY_DIR}/lib/cmake/luthier/luthierVersion.cmake
        VERSION "${luthier_VERSION}"
        COMPATIBILITY AnyNewerVersion
)
install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/luthier/luthierVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

# Write & Install Config Files
configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/luthierConfig.cmake.in
        ${CMAKE_BINARY_DIR}/lib/cmake/luthier/luthierConfig.cmake @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/luthier/luthierConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/modules/LuthierAddCompilerPlugin.cmake
        ${CMAKE_BINARY_DIR}/lib/cmake/luthier/LuthierAddCompilerPlugin.cmake COPYONLY)

install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/luthier/LuthierAddCompilerPlugin.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/modules/LuthierAddClangPlugin.cmake
        ${CMAKE_BINARY_DIR}/lib/cmake/luthier/LuthierAddClangPlugin.cmake COPYONLY)

install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/luthier/LuthierAddClangPlugin.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/modules/LuthierCreateOffloadBundle.cmake
        ${CMAKE_BINARY_DIR}/lib/cmake/luthier/LuthierCreateOffloadBundle.cmake COPYONLY)

install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/luthier/LuthierCreateOffloadBundle.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

configure_file(
        ${CMAKE_SOURCE_DIR}/cmake/modules/FindLLVMSPIRVTranslator.cmake
        ${CMAKE_BINARY_DIR}/lib/cmake/luthier/FindLLVMSPIRVTranslator.cmake COPYONLY)

install(FILES ${CMAKE_BINARY_DIR}/lib/cmake/luthier/FindLLVMSPIRVTranslator.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier)

# Install Include directories
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY ${CMAKE_BINARY_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install & Export Targets

install(TARGETS LuthierAMDGPU LuthierTooling LuthierToolIRCompilationPlugin
        LuthierToolCXXCompilationPlugin
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT luthierTargets
        FILE_SET amdgpu_tablegen_incs DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/luthier/AMDGPU/)

install(TARGETS luthier-tblgen
        RUNTIME DESTINATION ${CMAKE_INSTALL_BIN_DIR})

install(TARGETS luthier-llc
        RUNTIME DESTINATION ${CMAKE_INSTALL_BIN_DIR})

install(EXPORT luthierTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/luthier/)

export(EXPORT luthierTargets
        FILE "${CMAKE_BINARY_DIR}/lib/cmake/luthier/luthierTargets.cmake")