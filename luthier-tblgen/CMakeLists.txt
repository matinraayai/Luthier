include(AddLLVM)
include(TableGen)

add_tablegen(luthier-tblgen luthier
        RealToPseudoOpcodeMapBackend.hpp
        RealToPseudoOpcodeMapBackend.cpp
        RealToPseudoRegisterMapBackend.hpp
        RealToPseudoRegisterMapBackend.cpp
        Main.cpp
)

add_llvm_compile_definitions(luthier-tblgen)
target_include_directories(luthier-tblgen PUBLIC ${LUTHIER_LLVM_SRC_DIR}/llvm/utils/TableGen)
target_link_libraries(luthier-tblgen PUBLIC LLVMSupport LLVMTableGen LLVMTableGenCommon)