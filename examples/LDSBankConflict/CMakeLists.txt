cmake_minimum_required(VERSION 3.21)
project(LuthierLDSBankConflict LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(luthier REQUIRED)

find_package(LLVM REQUIRED)

add_library(LuthierLDSBankConflict SHARED LDSBankConflict.hip)

luthier_add_compiler_plugin(LuthierLDSBankConflict LuthierIModuleEmbedPlugin)

target_link_libraries(LuthierLDSBankConflict PUBLIC LuthierTooling LLVMDemangle hip::device hip::host)