# Fails if the preloadable KFD wrapper links LLVM.
#
# The wrapper is injected into arbitrary applications with LD_PRELOAD, so every
# library it needs must be present in those processes. LLVM is not, and a process
# that cannot satisfy the dependency fails to start at all -- taking the
# application with it, for a reason that points at the loader rather than at us.
#
# The property is structural: KFD's LLVM-facing code lives in a separate target
# (LuthierKFDTooling) that is deliberately not part of this library. But
# luthier-kfd-queue-wrapper is built from $<TARGET_OBJECTS:LuthierKFD>, which takes
# *all* of that target's objects, so adding one LLVM-using source to the wrong
# target silently breaks it. Nothing else would notice until a preload failed on a
# machine without LLVM, so it is checked here.
#
#   cmake -DBINARY=<path> -P CheckNoLlvmLinkage.cmake

if (NOT DEFINED BINARY)
    message(FATAL_ERROR "CheckNoLlvmLinkage.cmake needs -DBINARY=<path>")
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

string(REGEX MATCHALL "[^\n]*(LLVM|libclang)[^\n]*" FORBIDDEN "${DEPS}")

if (FORBIDDEN)
    string(REPLACE ";" "\n  " FORBIDDEN_TEXT "${FORBIDDEN}")
    message(FATAL_ERROR
            "${BINARY} links LLVM:\n  ${FORBIDDEN_TEXT}\n"
            "This library is preloaded into arbitrary applications, which cannot "
            "be assumed to have LLVM available. Something that uses LLVM was "
            "almost certainly added to the LuthierKFD target; it belongs in "
            "LuthierKFDTooling instead.")
endif ()

message(STATUS "linkage check passed: no LLVM in ${BINARY}")
