# The AMD GPU Mock Loader

## Introduction

The mock loader is a very useful tool for Luthier developers that streamlines the instrumentation pass development
process, allowing them to easily experiment with different possible instrumentation scenarios without the constraints
of having physical access to an AMD GPU device to load their target code. In this section we go over Luthier's
`MockAMDGPULoader` design and its usage.

## Background

By design, Luthier's lifting and inspection passes only inspect the contents of executable memory regions on a
target device, and use code objects as an optional secondary source of information or "hint" if they are available.
Exclusively running the inspection passes directly on GPU memory has the following issues:

1. It makes testing the inspection passes cumbersome and hard to scale, as having physical access to
   all supported devices is almost always not feasible.
2. It provides no way to instrument code objects offline without first loading them onto a target device.

## Design

A loader implementation called the "Mock AMDGPU Loader" is introduced to dynamically load and link AMD GPU code
objects on host memory instead of device memory, thereby "mocking" or emulating the normal loading process and
eliminating the need for a physical AMD GPU to utilize Luthier's instrumentation passes.

The `MockAMDGPULoader` class works somewhat similarly to the AMD HSA loader's `hsa_executable_t` (and it indeed started
off as a modified version of it). It allows for loading multiple code objects, and even manually defining variables
externally to be used to resolve any undefined variables in the loaded code objects (equivalent to
`hsa_executable_agent_global_variable_define`). Once the user is done loading code objects, they can call
`finalize()` to finalize the loading process (equivalent to `hsa_executable_freeze` in HSA).

There are, however some important differences between the mock loader and the HSA executable loader. Unlike the
HSA loader, the mock loader:

1. only supports HSA code object versions 3 and up. Code object version 1 does not follow proper ELF conventions to
   begin with and its relocation enumerations are incompatible with newer HSA code object versions. Even though code
   object version 2 does support the current AMD GPU relocations and is better formed, the loader chooses not to support
   it due to having to inspect the note section of the code object in order to distinguish between code objects version
   1 and 2.
2. does not apply static and dynamic relocations the moment the code objects are loaded. Instead, it defers the
   relocation resolving to the finalization stage. This way, the loader can support loading circularly dependent code
   objects (e.g., code object A defines variable A used by code object B, and code object B defines variable B used by
   code object A).
3. links externally defined variable and the external variables in loaded code objects regardless of their types.
4. supports all relocations as documented in
   the [AMDGPU LLVM backend](https://llvm.org/docs/AMDGPUUsage.html#relocation-records).
5. allows for loading of code objects targeting other OSes including PAL or Mesa for OS-agnostic testing.
6. allows loading of code objects targeting different architectures and capabilities on the host memory space to emulate
   linking of code objects on multiple devices.

It is worth mentioning that the loader will not work with object files (i.e., ELFs that don't have program headers).

The mock loader is not thread-safe as it is intended for testing.
