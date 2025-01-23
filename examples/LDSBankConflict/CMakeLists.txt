cmake_minimum_required(VERSION 3.21)
project(LuthierLDSBankConflict LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

find_package(LLVM REQUIRED CONFIG)

find_package(LuthierCompilerPlugins REQUIRED CONFIG)

add_library(LuthierLDSBankConflict SHARED LDSBankConflict.hip)

luthier_add_hsa_compile_definitions(LuthierLDSBankConflict)

luthier_add_position_independence(LuthierLDSBankConflict)

luthier_add_llvm_compile_definitions(LuthierLDSBankConflict)

luthier_add_compiler_plugin(LuthierLDSBankConflict luthier::IModuleEmbedPlugin)

target_link_libraries(LuthierLDSBankConflict PUBLIC LuthierTooling LLVMDemangle hip::device hip::host)