# The KFD-level half of Luthier.
#
# Two targets:
#
#   LuthierKFD          the LLVM-free driver-boundary logic -- the allocation map
#                       and the DRM descriptor sharing. Linked into a tool, and
#                       also into KFDUnitTests on its own.
#   LuthierKFDTooling   the parts that use LLVM: the accessor that bridges the
#                       allocation map to Luthier's code generation pipeline, and
#                       the sysfs-derived target machine.
#
# There used to be a third, luthier-kfd-queue-wrapper: a preloadable shared
# library that installed a GOTCHA interception layer and carried the queue
# wrapper's logic. Both are gone. The logic moved into
# luthier::PacketMonitorTrait, whose fields it now is, and the interception is
# LD_AUDIT via audit_hook -- which needs no preloadable library of ours at all,
# because the tool itself is the audit_hook plugin and ld.so is what installs the
# hooks.
#
# The split into two targets survives that change for its original reason: a tool
# is injected into arbitrary applications, so the driver-boundary logic must not
# require LLVM to be present, and anything that links LLVM therefore needs its own
# target.

add_library(LuthierKFD OBJECT
        AllocationTracker.cpp
        FdSharing.cpp
)

target_compile_definitions(LuthierKFD PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

# LLVM's includes are needed now that the allocation tracker reports failures as
# llvm::Error. This target used to be deliberately LLVM-free so it could go into
# a library preloaded into arbitrary processes; there is no such library any more
# -- the tool is an audit_hook plugin and already links LLVM.
target_include_directories(LuthierKFD
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
)

# Required: the objects are linked into a shared library.
set_property(TARGET LuthierKFD PROPERTY POSITION_INDEPENDENT_CODE ON)

#===----------------------------------------------------------------------===#
# The LLVM-facing half: bridges the allocation map to Luthier's code generation
# pipeline.
#===----------------------------------------------------------------------===#
add_library(LuthierKFDTooling OBJECT
        KfdAllocationResolver.cpp
        IsaInfo.cpp
        KfdTargetMachine.cpp
        Topology.cpp
)

target_compile_definitions(LuthierKFDTooling PRIVATE AMD_INTERNAL_BUILD ${LLVM_DEFINITIONS})

target_include_directories(LuthierKFDTooling
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        ${LLVM_INCLUDE_DIRS}
        ${hsa-runtime64_INCLUDE_DIRS}
)

set_property(TARGET LuthierKFDTooling PROPERTY POSITION_INDEPENDENT_CODE ON)
