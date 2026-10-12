# Set of knobs overriding TheRock's pre_hook_amd-llvm.cmake variables for
# the dev build of LLVM
# Force LLVM to build every component as a shared object instead of a static
# library. This way, including debug info won't baloon the LLVM installation
# to 100GB+.
set(BUILD_SHARED_LIBS ON)
# Disable dylib building from TheRock because we already forced LLVM to
# build shared libraries
set(LLVM_BUILD_LLVM_DYLIB OFF)
set(LLVM_LINK_LLVM_DYLIB OFF)
# Set assertions to ON but leave expensive checks OFF
set(LLVM_ENABLE_ASSERTIONS ON)
set(LLVM_ENABLE_EXPENSIVE_CHECKS OFF)
