add_executable(InstrSemanticFuzzer
  InstrFuzzerMain.cpp
  HSADispatcher.cpp
  InstrDescriptor.cpp
  KernelCompiler.cpp
  StandaloneMIBuilder.cpp
  IRKernelGen.cpp
  MachineKernelBuilder.cpp
  FuzzerDriver.cpp
  # Object libraries — include their .o files directly in the executable.
  $<TARGET_OBJECTS:LuthierIRTranslator>
  $<TARGET_OBJECTS:LuthierHSA>
)

# LLVM definitions (e.g. -D_GNU_SOURCE, -DNDEBUG, etc.)
target_compile_definitions(InstrSemanticFuzzer PRIVATE ${LLVM_DEFINITIONS})

target_include_directories(InstrSemanticFuzzer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  # LLVM public headers
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(InstrSemanticFuzzer PRIVATE
  gtest
  gtest_main
  # Luthier libraries (OBJECT libs added via $<TARGET_OBJECTS:> above)
  LuthierAMDGPU
  # HSA runtime (CMake imported target from find_package(hsa-runtime64))
  hsa-runtime64::hsa-runtime64
  # LLVM libraries
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
