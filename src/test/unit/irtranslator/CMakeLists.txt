# Reference path of the AMDGPU instruction semantic fuzzer.
#
# Layout:
#   include/            all headers
#   src/fuzzer/         orchestration (driver, dispatcher, kernel front end)
#   src/instructions/   per-instruction-class reference-kernel builders
#   src/translation/    not-yet-ported LLVM IR path (excluded from the build)
#
# The translation path (src/translation/*) is written against an older,
# never-completed Luthier API generation (luthier::translateSingleInstr /
# PhysRegValueMap) that does not exist in this tree; it is kept for reference
# but not compiled.

add_executable(InstrSemanticFuzzer
  # Orchestration.
  src/fuzzer/InstrFuzzerMain.cpp
  src/fuzzer/HSADispatcher.cpp
  src/fuzzer/InstrDescriptor.cpp
  src/fuzzer/FuzzerDriver.cpp
  src/fuzzer/MachineKernelBuilder.cpp
  src/fuzzer/RefKernelSupport.cpp
  src/fuzzer/DiagnosticDump.cpp
  # Per-instruction-class builders.
  src/instructions/VOP.cpp
  src/instructions/SOP.cpp
  src/instructions/FLAT.cpp
  src/instructions/Scratch.cpp
  src/instructions/SMEM.cpp
  src/instructions/MUBUF.cpp
  src/instructions/DS.cpp
  $<TARGET_OBJECTS:LuthierHSA>
  # Translation path: raise the reference-kernel MachineFunction back to LLVM IR
  # via TraceFunctionTranslator, recompile, and compare against the reference
  # path. Only the translator's own sources are compiled in (the full
  # LuthierToolCodeGen object drags in the instrumentation pipeline).
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/TraceFunctionTranslator.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/MIInlineAsmEmitter.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/MIRConvenience.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/Metadata.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/RegValueMetadata.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/TargetMachineInstrMDNode.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/PseudoOpcodeAndRegMapper.cpp
  ${CMAKE_SOURCE_DIR}/src/lib/ToolCodeGen/FunctionAnnotations.cpp
)
add_dependencies(InstrSemanticFuzzer
  LuthierSIInstrSemanticsGen
  LuthierRealToPseudoOpcodeMap
  LuthierRealToPseudoRegEnumMap)

target_compile_definitions(InstrSemanticFuzzer PRIVATE
  AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(InstrSemanticFuzzer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_SOURCE_DIR}/include
  # Generated translator headers (SIInstrSemantics.inc, pseudo-map .hpp).
  ${CMAKE_BINARY_DIR}/src/lib/ToolCodeGen
  ${CMAKE_BINARY_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${hsa-runtime64_INCLUDE_DIRS}
)

target_link_libraries(InstrSemanticFuzzer PRIVATE
  gtest
  LuthierAMDGPU
  LuthierCommon
  # Provides luthier::dbgs()/errs(), referenced by LuthierHSA's debug logging.
  LuthierLLVM
  # Links the emitted relocatable object into a loadable code object.
  LuthierLinker
  LLVMBinaryFormat
  hip::host
  hsa-runtime64::hsa-runtime64
  ${CMAKE_DL_LIBS}
  LLVMCore
  LLVMSupport
  LLVMTarget
  LLVMTargetParser
  LLVMCodeGen
  LLVMMC
  LLVMMCParser
  # Pulled in by LuthierCommon's stack tracing.
  LLVMSymbolize
  LLVMDebugInfoDWARF
  LLVMObject
  LLVMAMDGPUCodeGen
  LLVMAMDGPUDesc
  LLVMAMDGPUInfo
  LLVMAMDGPUAsmParser
  LLVMAMDGPUUtils
  # Pulled in by LuthierToolCodeGen (translation path).
  LLVMMIRParser
  LLVMIRReader
  LLVMAsmParser
  LLVMPasses
  LLVMipo
  LLVMScalarOpts
  LLVMInstCombine
  LLVMTransformUtils
  LLVMAnalysis
  LLVMBitWriter
  LLVMBitReader
  LLVMLinker
  LLVMProfileData
  LLVMDemangle
)

# luthier::linker::linkRelocatableToExecutable shells out to `ld.lld`, so the
# test needs it on PATH. ROCm ships it next to its own LLVM tools, which is not
# normally on PATH, so locate it and prepend its directory for the test run.
find_program(LUTHIER_LD_LLD
        NAMES ld.lld
        HINTS ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES bin llvm/bin lib/llvm/bin)

if (LUTHIER_LD_LLD)
    get_filename_component(LUTHIER_LD_LLD_DIR "${LUTHIER_LD_LLD}" DIRECTORY)
    message(STATUS "InstrSemanticFuzzer: using ld.lld at ${LUTHIER_LD_LLD}")
    gtest_discover_tests(InstrSemanticFuzzer
            PROPERTIES ENVIRONMENT "PATH=${LUTHIER_LD_LLD_DIR}:$ENV{PATH}")
else ()
    message(WARNING "InstrSemanticFuzzer: ld.lld not found; the reference path "
            "cannot link its code objects and its tests will fail")
    gtest_discover_tests(InstrSemanticFuzzer)
endif ()
