# Building Sibir

## Build requirements
Sibir should work on any Linux-based distribution with ROCm support. Sibir does not have Microsoft Windows support. 
There are no plans for supporting Sibir on Windows at this time.

The following software components are required as external dependencies to Sibir:
1. **[CMake](https://cmake.org/)** is used by Sibir as its build system along with 
**[GNU Make](https://www.gnu.org/software/make/)**. The [CMakeLists.txt](../CMakeLists.txt) file for Sibir is located 
at the top-level directory of the project. **CMake v3.21 and above** is required, since it is the earliest version that 
supports HIP as a first-class citizen with built-in HIP/ROCm-specific CMake variables.
2. **[AMD Code Object Manager Library (COMGR)](https://github.com/RadeonOpenCompute/ROCm-CompilerSupport/)** is used for 
assembling/disassembling AMDGPU instructions, parsing AMDGPU code object notes, and querying information about GPU ISA
to use for generating instrumented code.
3. **[AMD HSA and ROCm Runtime Library](https://github.com/RadeonOpenCompute/ROCR-Runtime)** is used for intercepting
calls to AMD's Heterogeneous Systems Architecture (HSA) implementation (including kernel launches), loading/unloading
instrumentation code, querying the HSA agents (GPUs) present on the system, and other core GPU functionality needed by
AMDGPUs for instrumentation.
4. **[AMD Compute Language Runtimes](https://github.com/ROCm-Developer-Tools/clr)** provides AMD's
Heterogeneous-Compute Interface for Portability (HIP) compiler and its runtime. Sibir requires HIP to intercept its 
calls, compile device-side code, and load user-written device code.
5. **A C/C++ compiler**, like [GNU GCC](https://gcc.gnu.org/) or [Clang](https://clang.llvm.org/).

## Build Options

### Sibir-specific Options
- **```-DSIBIR_BUILD_EXAMPLES```**: Builds the example tools under the [examples](../examples) folder if set to 
```ON```. It is enabled by default.
- **```-DSIBIR_LOG_LEVEL```**: Sets the log level for Sibir. Valid options include ```ERROR``` (always enabled), 
```INFO``` (prints which internal functions are called), and ```DEBUG``` (prints useful information for debugging).

### Useful CMake Options
- **```-DCMAKE_CXX_COMPILER```** and **```-DCMAKE_C_COMPILER```** set paths to the desired C/C++ compiler.
- **```-DCMAKE_BUILD_TYPE```** can enable/disable build with source debug information if set to ```Debug``` or 
```RelWithDebugInfo```.

## Example Build Command

```shell
mkdir build/
cd build/
cmake -DCMAKE_CXX_COMPILER=gcc -DSIBIR_LOG_LEVEL=DEBUG -DCMAKE_BUILD_TYPE=Debug .. 
make -j
```