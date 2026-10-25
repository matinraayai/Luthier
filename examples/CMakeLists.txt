add_subdirectory(LiftLaunchedKernels)
add_subdirectory(InstrCount)
add_subdirectory(KernelArgumentIntrinsic)
# add_subdirectory(KernelInstrument)
add_subdirectory(LDSBankConflict)
add_subdirectory(OpcodeHistogram)

# Hide unnecessary symbols in the tool (including LLVM-related things)
set(LUTHIER_TOOL_VERSION_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/LuthierTool.map)
foreach (_TOOL LiftLaunchedKernels LuthierInstrCount LuthierKernelArgumentIntrinsic
        LuthierLDSBankConflict LuthierOpcodeHistogram)
    target_link_options(${_TOOL} PRIVATE "LINKER:--version-script=${LUTHIER_TOOL_VERSION_SCRIPT}")
    set_property(TARGET ${_TOOL} APPEND PROPERTY LINK_DEPENDS ${LUTHIER_TOOL_VERSION_SCRIPT})
endforeach ()