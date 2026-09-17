add_executable(InstrSemanticFuzzer
  InstrFuzzerMain.cpp
  HSADispatcher.cpp
  InstrDescriptor.cpp
  KernelCompiler.cpp
  StandaloneMIBuilder.cpp
  IRKernelGen.cpp
  MachineKernelBuilder.cpp
  FuzzerDriver.cpp
  $<TARGET_OBJECTS:LuthierToolCodeGen>
  $<TARGET_OBJECTS:LuthierHSA>
  # hsa::agentForGpuId bridges an agent to the driver's gpu_id through
  # sysfs. Not linked via LuthierKFDTooling, which would drag in the rest
  # of the KFD layer; this one file has no dependencies of its own.
  ${CMAKE_SOURCE_DIR}/src/lib/KFD/Topology.cpp
)

target_compile_definitions(InstrSemanticFuzzer PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(InstrSemanticFuzzer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_SOURCE_DIR}/include/
        ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(InstrSemanticFuzzer PRIVATE
  gtest
  gtest_main
  LuthierAMDGPU
  hsa-runtime64::hsa-runtime64
  LLVMCore
  LLVMSupport
  LLVMTarget
  LLVMCodeGen
  LLVMMCParser
  LLVMMIRParser
  LLVMAMDGPUCodeGen
  LLVMAMDGPUDesc
  LLVMAMDGPUInfo
  LLVMAMDGPUAsmParser
)

gtest_discover_tests(InstrSemanticFuzzer)
