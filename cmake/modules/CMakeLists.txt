install(DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

include(CMakePackageConfigHelpers)

#Write & Install Version File
write_basic_package_version_file(
        ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectVersion.cmake
        VERSION "${LuthierProject_VERSION}"
        COMPATIBILITY AnyNewerVersion
)
install(FILES ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierProject)

#message(STATUS "Source Directory: ${LuthierProject_SOURCE_DIR}")
set(LuthierProject_SOURCE_DIR ${CMAKE_SOURCE_DIR})
#Write & Install Config File
configure_file(${LuthierProject_SOURCE_DIR}/cmake/LuthierProjectConfig.cmake.in
        ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectConfig.cmake @ONLY)

install(FILES ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierProject)

#ALSO NEED TO INSTALL LUTHIER BINARIES

#Install & Export Targets
#Use a cmake file 
install(TARGETS LuthierAMDGPU LuthierTooling
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT LuthierProjectTargets)

#Adding RUNTIME ahead of DESTINATION ruins it with EXPORT LuthierProjectTargets
#Adding RUNTIME without EXPORT LuthierProjectTargets works fine
install(TARGETS luthier-tblgen
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})

export(EXPORT LuthierProjectTargets
        FILE "${CMAKE_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectTargets.cmake")