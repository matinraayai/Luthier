# The LD_AUDIT interception engine Luthier's PacketMonitorTrait runs on.
#
# Vendored from https://github.com/woodard/audit_hook, branch main (upstream has
# no tagged releases), commit e0c7c25a1e598bf057efa183354a86dcc03c4934 of
# 2026-08-16:
#
#   include/audit_hook.hpp        -> include/luthier/Audit/audit_hook.hpp
#   include/audit_hook_dynamic.h  -> include/luthier/Audit/audit_hook_dynamic.h
#   src/audit_core.cpp            -> src/lib/Audit/audit_core.cpp
#   LICENSE                       -> src/lib/Audit/LICENSE.audit_hook
#
# All three sources are byte-identical to upstream -- nothing was patched, so a
# diff against that commit is the whole review. Everything Luthier-specific is in
# this file.
#
# +------------------------------------------------------------------------+
# | LICENSING: audit_hook is GPL v3, while the rest of Luthier is Apache     |
# | 2.0. Linking LuthierAudit into LuthierTooling makes the resulting binary |
# | a GPLv3 work. Apache 2.0 is one-way compatible with GPLv3, so that       |
# | combination is permitted, but it is not the licence the rest of this     |
# | repository is distributed under. The full text is appended to Luthier's  |
# | LICENSE and kept verbatim in LICENSE.audit_hook here.                    |
# +------------------------------------------------------------------------+
#
#===----------------------------------------------------------------------===#
# Replicating upstream's build
#
# Upstream is autotools. Its src/Makefile.am says:
#
#   AM_CPPFLAGS                 = -I$(top_srcdir)/include
#   libaudit_core_la_SOURCES    = audit_core.cpp
#   libaudit_core_la_CXXFLAGS   = -fPIC -shared
#   libaudit_core_la_LDFLAGS    = -module -avoid-version
#
# and configure.ac adds -std=c++20. Each maps onto exactly one property below:
# the include directory onto target_include_directories (which is also what lets
# audit_core.cpp keep its verbatim #include "audit_hook.hpp"), -fPIC onto
# POSITION_INDEPENDENT_CODE, -avoid-version onto simply not setting VERSION or
# SOVERSION, and -module onto nothing -- it is libtool's way of saying "this is a
# plugin, not a versioned library", which is the default for a CMake SHARED
# target with no version set.
#
# Upstream's second library, libaudit_hook_dynamic, is deliberately not vendored.
# It is a stub an *application* links so that its ah_set_target calls can be
# bridged into the auditor's namespace. Luthier needs no bridge: the tool is
# itself the plugin, so it registers its hooks inside the auditor's namespace
# directly. Copying a library nothing calls would be copying dead GPL code.
#===----------------------------------------------------------------------===#

add_library(LuthierAudit SHARED
        audit_core.cpp
)

# -std=c++20 from upstream's configure.ac. Pinned on this target rather than
# inherited from Luthier's CMAKE_CXX_STANDARD of 23, so that the vendored sources
# are compiled the way upstream tests them.
set_target_properties(LuthierAudit PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)

# PUBLIC on the parent include directory, so a Luthier header can say
# #include "luthier/Audit/audit_hook.hpp". PRIVATE on the Audit directory itself,
# which is what makes audit_core.cpp's own #include "audit_hook.hpp" resolve --
# upstream's AM_CPPFLAGS, expressed for a tree where the headers moved.
target_include_directories(LuthierAudit
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        "${CMAKE_SOURCE_DIR}/include/luthier/Audit"
)

# dl for dladdr, which the dynamic dispatcher uses to identify a caller's library.
target_link_libraries(LuthierAudit PRIVATE ${CMAKE_DL_LIBS})

# Installed next to Luthier's own libraries, because LD_AUDIT names the engine by
# path at process start: an instrumented application is launched with
# LD_AUDIT=<prefix>/lib/libLuthierAudit.so, so this has to be a file in the
# install tree rather than an artefact left in a build directory.
install(TARGETS LuthierAudit
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        EXPORT luthierTargets)
