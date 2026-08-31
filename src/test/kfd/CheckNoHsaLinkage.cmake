# Fails if the non-HSA test binary links the HSA runtime.
#
# This is the assumption the whole suite rests on: the binary stands in for an
# application that never uses HSA. If HSA arrives through a transitive
# dependency, the tests still pass and still look meaningful while no longer
# testing the case issue #85 exists for. Nothing else would notice, so it is
# checked here.
#
#   cmake -DBINARY=<path> -P CheckNoHsaLinkage.cmake

if (NOT DEFINED BINARY)
    message(FATAL_ERROR "CheckNoHsaLinkage.cmake needs -DBINARY=<path>")
endif ()

find_program(LDD_EXECUTABLE ldd)
if (NOT LDD_EXECUTABLE)
    message(FATAL_ERROR "ldd not found, cannot verify linkage")
endif ()

execute_process(COMMAND ${LDD_EXECUTABLE} ${BINARY}
        OUTPUT_VARIABLE DEPS
        ERROR_VARIABLE DEPS_ERR
        RESULT_VARIABLE RC)
if (NOT RC EQUAL 0)
    message(FATAL_ERROR "ldd failed on ${BINARY}: ${DEPS_ERR}")
endif ()

# hsakmt is expected and fine -- it is the thin driver-call library, not the HSA
# runtime. What must be absent is the runtime itself and HIP.
string(REGEX MATCHALL "[^\n]*(hsa-runtime|amdhip)[^\n]*" FORBIDDEN "${DEPS}")

if (FORBIDDEN)
    string(REPLACE ";" "\n  " FORBIDDEN_TEXT "${FORBIDDEN}")
    message(FATAL_ERROR
            "${BINARY} links the HSA runtime:\n  ${FORBIDDEN_TEXT}\n"
            "This binary must stand in for an application that never uses HSA. "
            "With the runtime linked, the tests no longer measure what they "
            "claim to. Find the dependency that pulled it in and remove it.")
endif ()

message(STATUS "linkage check passed: no HSA runtime in ${BINARY}")
