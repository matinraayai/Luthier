#Maintain logic for installing headers, modules, cmake config files

# Install header files from each folder recursively with /include/
# Install all headers from the include directory, recursively
# CMAKE_INSTALL_PREFIX is set to /opt/luthier/llvm during build commands
# CMAKE_SOURCE_DIR is relative, it's luthier
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_PREFIX})

#Helps with pkg config files (.cmake)
include(CMakePackageConfigHelpers)

#Create Version File
# Informs other projects about pkg version and compatability
# 
# write_basic_version_file()

#Install Version File
# can use the binary_dir of just luthier
# install(FILES $(
# 

#Create the Pkg Configuration File 
# configure_file()

#Install Pkg Config File
# install()

#Install Luthier Targets & Export- ?
# install(TARGETS  )

# install(EXPORT )

# export(EXPORT )

#Install & Export Modules in cmake dir


