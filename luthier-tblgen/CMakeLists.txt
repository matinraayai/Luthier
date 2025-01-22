find_package(LLVM REQUIRED CONFIG)

include(AddLLVM)
include(TableGen)

add_tablegen(luthier-tblgen luthier
        RealToPseudoOpcodeMapBackend.hpp
        RealToPseudoOpcodeMapBackend.cpp
        RealToPseudoRegisterMapBackend.hpp
        RealToPseudoRegisterMapBackend.cpp
        Main.cpp
)

target_include_directories(luthier-tblgen PRIVATE ${LLVM_INCLUDE_DIRS})
target_include_directories(luthier-tblgen PRIVATE ${LUTHIER_LLVM_SRC_DIR}/llvm/utils/TableGen)

target_compile_definitions(luthier-tblgen PRIVATE ${LLVM_DEFINITIONS})

target_link_libraries(luthier-tblgen PRIVATE LLVMSupport LLVMTableGen LLVMTableGenCommon)