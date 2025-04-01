# Luthier: A Dynamic Binary Instrumentation Framework Targeting AMD GPUs

## What is Luthier?
Luthier is a dynamic binary instrumentation framework for writing tools targeting AMD GPUs.
It supports (or aims to support):

- Analyzing the content of the code objects loaded on the GPU at runtime; This includes kernels, device functions,
  static variables, etc.
- Inserting calls to (multiple) device instrumentation functions.
- Removal/modification/addition of instructions.
- Querying/modification of the GPU's ISA visible state.

## What AMD GPU Applications Luthier Supports?

Luthier supports any [ROCm](https://www.amd.com/en/products/software/rocm.html)-backed application on Linux.
This includes any applications that uses HIP, OpenMP, OpenCL, or even the 
[ROCM runtime](https://github.com/RadeonOpenCompute/ROCR-Runtime/) directly to load and launch kernels on the GPU,
regardless of how their module (code object) was created and loaded.

Luthier does not support the [Platform Abstraction Layer (PAL)](https://github.com/GPUOpen-Drivers/pal)-based 
applications or Windows.

## How Can I get started?

For more information on how to build, get started, and contribute, see the [documentation](docs) folder.
To report bugs, feel free to file an issue on GitHub. 

## Found a Bug? Having Issues? Experiencing Crashes? Have a Suggestion or a feature request?
Feel free to file an issue. If you're reporting a crash, please include the stack trace of the
error, and preferably, compile Luthier with `CMAKE_BUILD_TYPE=Debug` and include the debug info alongside
the stack trace of the crash.
