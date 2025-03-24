
#Find package for hip & rocprofiler-sdk
#Look at hip compiler to see how this is done
# Look for ROCm components
#find_package(hip REQUIRED)
#find_package(rocprofiler-sdk REQUIRED)

# Check if ROCm components were found
#if (hip_FOUND AND rocprofiler-sdk_FOUND)
#    message(STATUS "ROCm components found: hip and rocprofiler-sdk")
#else()
#    message(WARNING "ROCm components (hip and/or rocprofiler-sdk) not found. Some features may be disabled.")
#endif()


install(DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})

include(CMakePackageConfigHelpers)

#Write & Install Version Cmake File
write_basic_package_version_file(
        ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectVersion.cmake
        VERSION "${LuthierProject_VERSION}"
        COMPATIBILITY AnyNewerVersion
)
install(FILES ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierProject)

#Set LuthierProject SRC DIR variable
set(LuthierProject_SOURCE_DIR ${CMAKE_SOURCE_DIR})

#Write & Install Config File
configure_file(
        ${LuthierProject_SOURCE_DIR}/cmake/LuthierProjectConfig.cmake.in
        ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectConfig.cmake @ONLY)

#Change destination to /cmake/LuthierProject/
install(FILES ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierProject)

#ALSO NEED TO INSTALL LUTHIER BINARIES
install(DIRECTORY ${CMAKE_BINARY_DIR}/include
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

#Install & Export Targets
#Use a cmake file 
install(TARGETS LuthierAMDGPU LuthierTooling
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT LuthierProjectTargets)

#Adding RUNTIME ahead of DESTINATION ruins it with EXPORT LuthierProjectTargets
#Adding RUNTIME without EXPORT LuthierProjectTargets works fine
install(TARGETS luthier-tblgen
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

#Install & Export LuthierProjectTargets to proper location
install(EXPORT LuthierProjectTargets 
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierProject)

#LuthierProject_BINARY_DIR or CMAKE_BINARY_DIR?
export(EXPORT LuthierProjectTargets
        FILE "${CMAKE_BINARY_DIR}/lib/cmake/LuthierProject/LuthierProjectTargets.cmake")