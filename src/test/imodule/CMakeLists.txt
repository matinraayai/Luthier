## Build-time IModule fixtures for payload-injection tests.
##
## Each .hip imodule is compiled with the LuthierToolCXXCompilation plugin in
## --cuda-device-only mode, then run through the LuthierToolIRCompilation
## worker passes (mark-annotations, finalize-intrinsics,
## substitute-amdgcn-intrinsics) to produce the canonical IModule IR that
## tests consume directly.
##
## Produces ${CMAKE_BINARY_DIR}/test/imodule/<Name>-<gfx>.ll for each
## (imodule, arch) combination listed below.

set(LUTHIER_TEST_IMODULE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/test/imodule)
file(MAKE_DIRECTORY ${LUTHIER_TEST_IMODULE_OUTPUT_DIR})

set(_clang ${LLVM_TOOLS_BINARY_DIR}/clang++)
set(_opt ${LLVM_TOOLS_BINARY_DIR}/opt)

set(LUTHIER_TEST_GFX_ARCHS
        gfx906
        gfx908
        gfx90a
        gfx942
        gfx1030
        gfx1100
        gfx1200)

set(LUTHIER_TEST_IMODULE_NAMES
        TrivialCounter
        InstrCount
        OpcodeHistogram
        BranchDivergence
        ExecMaskTracker
        WriteExecGuarded
        RegLogger)

set(_ir_worker_pipeline
        "luthier-mark-annotations,luthier-finalize-intrinsics,luthier-substitute-amdgcn-intrinsics")

set(_imodule_fixture_outputs "")

foreach (_name ${LUTHIER_TEST_IMODULE_NAMES})
    set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${_name}/${_name}.hip)
    foreach (_arch ${LUTHIER_TEST_GFX_ARCHS})
        set(_raw ${LUTHIER_TEST_IMODULE_OUTPUT_DIR}/${_name}-${_arch}-raw.ll)
        set(_out ${LUTHIER_TEST_IMODULE_OUTPUT_DIR}/${_name}-${_arch}.ll)

        add_custom_command(
                OUTPUT ${_raw}
                COMMAND ${_clang}
                -x hip --offload-arch=${_arch} --cuda-device-only
                -fplugin=$<TARGET_FILE:LuthierToolCXXCompilationPlugin>
                -include hip/hip_runtime.h
                -isystem ${CMAKE_SOURCE_DIR}/include
                -isystem ${LLVM_INCLUDE_DIRS}
                -O2 -emit-llvm -S
                -o ${_raw} ${_src}
                DEPENDS ${_src} LuthierToolCXXCompilationPlugin
                COMMENT "Compiling test imodule ${_name} for ${_arch}")

        add_custom_command(
                OUTPUT ${_out}
                COMMAND ${_opt}
                -load-pass-plugin=$<TARGET_FILE:LuthierToolIRCompilationPlugin>
                -passes=${_ir_worker_pipeline}
                -S ${_raw} -o ${_out}
                DEPENDS ${_raw} LuthierToolIRCompilationPlugin
                COMMENT "Finalizing test imodule ${_name}-${_arch}")

        list(APPEND _imodule_fixture_outputs ${_out})
    endforeach ()
endforeach ()

add_custom_target(luthier-test-imodules ALL DEPENDS ${_imodule_fixture_outputs})
