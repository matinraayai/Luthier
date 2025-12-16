add_library(LuthierIntrinsic OBJECT
        ReadReg.cpp
        WriteReg.cpp
        WriteExec.cpp
        IntrinsicProcessor.cpp
        ImplicitArgPtr.cpp
        SAtomicAdd.cpp
)

add_dependencies(LuthierIntrinsic LuthierAMDGPUTableGen)

target_compile_definitions(LuthierIntrinsic PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(LuthierIntrinsic
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(LuthierIntrinsic PRIVATE LuthierAMDGPU)