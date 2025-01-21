# Luthier Compiler Plugins
This sub-folder contains LLVM compiler plugins for the Luthier tooling project. For now, it only contains the 
`EmbedIModulePlugin`, used by Luthier dynamic instrumentation tools during compilation. In the future, 
we will introduce other instrumentation-related compiler plugins here (e.g. static instrumentation of LLVM IR/MIR). 

This project is not built directly by the master [CmakeLists.txt](../CMakeLists.txt); It should be compiled and 
installed separately first before being used by the Luthier tooling project. This is because unlike the tooling project
(which depends on the upstream LLVM), the compiler plugins links against the LLVM installed under ROCm so that it can
be used by the `hipcc` compiler. As there is no good way of handling multiple versions of the LLVM same library (even
in different scopes and folders), there is no way around this issue.

As the ROCm LLVM is older than LLVM master, this project must be kept backward-compatible with older LLVM versions, 
down to the LLVM version in ROCm 6.2 (18).

Refer to the [Build Instructions](../docs/3.build.md) on how to build this along with the Luthier tooling project.
