enable_testing()
add_subdirectory(bin)
add_subdirectory(imodule)
add_subdirectory(lib)
add_subdirectory(lit)
add_subdirectory(unit)

#===----------------------------------------------------------------------===#
# The KFD hardware suite is not built at present, and re-enabling it is a port
# rather than a fix.
#
# Every scenario in it launches the workload under
# LD_PRELOAD=luthier-kfd-queue-wrapper, and that library no longer exists: the
# queue wrapper's logic became luthier::PacketMonitorTrait's fields and methods,
# and interception became LD_AUDIT. So the suite has to change in three places
# that are not mechanical:
#
#  * the launch mechanism. LD_PRELOAD of one library becomes
#    LD_AUDIT=$<TARGET_FILE:LuthierAudit> plus AH_PLUGINS naming the tool, since
#    the tool is now the audit_hook plugin;
#  * the observation points. NonHsaMain.cpp and HsaOracleMain.cpp reach the
#    wrapper through dlsym of luthierKfdAddPacketCallback,
#    luthierKfdWrappedQueueCount and friends. Those C exports are gone -- the
#    counts are instance methods on the trait now -- so the harnesses need a tool
#    to ask rather than a symbol to look up;
#  * the -badsym mutation, which proved --require-wrapper measures something. It
#    worked by rebuilding the GOTCHA interceptor against a symbol that does not
#    exist. The equivalent under LD_AUDIT is a plugin that registers a hook for a
#    nonexistent symbol, which is a different artifact to build.
#
# Left switched off rather than deleted or half-converted: a suite that is
# visibly absent is honest, whereas one that builds and silently exercises the
# wrong mechanism is the failure mode this directory exists to catch.
#===----------------------------------------------------------------------===#
option(LUTHIER_BUILD_KFD_HARDWARE_TESTS
        "Build the KFD hardware test suite. Needs porting from LD_PRELOAD to \
LD_AUDIT first -- see the comment in src/test/CMakeLists.txt." OFF)

if (LUTHIER_BUILD_KFD_HARDWARE_TESTS)
    add_subdirectory(kfd)
endif ()
