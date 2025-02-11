#Maintain logic for installing headers, modules, cmake config files

# Install header files from each folder recursively with /include/
# Install all headers from the include directory, recursively
# CMAKE_INSTALL_PREFIX is set to /opt/luthier/llvm during build commands
# CMAKE_SOURCE_DIR is relative, it's luthier

set(CMAKE_INSTALL_PREFIX "/")
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX})

#Helps with pkg config files (.cmake)
include(CMakePackageConfigHelpers)

#Create Version File
# Informs other projects about pkg version and compatability
# Compiler-plugins uses the built binary_dir env variable to 
write_basic_version_file(
    #Q WHERE DOES THIS ENV VARIABLE COME FROM / GET ESTABLISHED?
    #Need to change these variables
    ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProjectConfigVersion.cmake
    VERSION "${LuthierProject_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

#Install Version File
# can use the binary_dir of just luthier
# install(FILES $(

#Need to change environment variable below
install(FILES ${LuthierProject_BINARY_DIR}/lib/cmake/LuthierProjectConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierProject/)

#Create the Pkg Configuration File 
# configure_file() -> this uses the config .cmake.in file 

configure_file(
        ${LuthierConfig_SOURCE_DIR}/cmake/LuthierConfig.cmake.in
        ${LuthierConfig_BINARY_DIR} /lib/cmake/LuthierConfig/LuthierConfig.cmake @ONLY)

#Install Pkg Config File
# install()
install(FILES ${LuthierConfig_BINARY_DIR}/lib/cmake/LuthierCompilerPlugins/LuthierCompilerPluginsConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuthierConfig/)

#All object library installations w/ destination CMAKE_INSTALL_LIBDIR
#CMAKE_INSTALL_LIBDIR specifies directory where all libraries will be installed (lib/)
install(TARGETS LuthierAMDGPU DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierAMDGPUTableGen DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierComgr DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierCommon DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierHIP DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierHSA DESTINATION ${CMAKE_INSTALL_LIBDIR}) 
install(TARGETS LuthierIntrinsic DESTINATION ${CMAKE_INSTALL_LIBDIR}) #depends on AMDGPUTableGen
install(TARGETS LuthierLLVM DESTINATION ${CMAKE_INSTALL_LIBDIR}) #depends on AMDGPUTableGen
install(TARGETS LuthierRocprofilerSDK DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierTooling DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierToolingCommon DESTINATION ${CMAKE_INSTALL_LIBDIR}) #depends on AMDGPUTableGen

#Installing the Examples
#Can add EXPORT _____ to be used for other projects with find_package()
install(TARGETS LuthierInstrCount DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS intercept_hsa DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS intercept_hip DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierKernelArgumentIntrinsic DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS KernelInstrument DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierLDSBankConflict DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LuthierOpcodeHistogram DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS LiftLaunchedKernels DESTINATION ${CMAKE_INSTALL_LIBDIR})


#Install & Export Modules in cmake dir
#No modules created yet, Q:What can I make a module?

