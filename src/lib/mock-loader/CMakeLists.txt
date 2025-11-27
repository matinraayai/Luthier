add_library(LuthierMockLoader OBJECT
        amd_hsa_loader.cpp
        amd_loader_context.cpp
        amd_hsa_code.cpp
        amd_elf_image.cpp
        HsaMockLoader.cpp
)

target_compile_definitions(LuthierMockLoader PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierMockLoader
        PRIVATE
        "${CMAKE_SOURCE_DIR}/src/include"
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")