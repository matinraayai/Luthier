cmake_minimum_required(VERSION 3.21)

project(LuthierInstrCount LANGUAGES HIP CXX)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)

add_library(LuthierInstrCount SHARED InstrCount.hip)

set_property(TARGET LuthierInstrCount PROPERTY COMPILE_FLAGS "-fPIC")

luthier_add_compiler_plugin(LuthierInstrCount LuthierIModuleEmbedPlugin)

target_link_libraries(LuthierInstrCount PRIVATE LuthierTooling hip::device)