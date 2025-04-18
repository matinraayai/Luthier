# Luthier Compiler Plugins
This sub-folder contains LLVM compiler plugins for the Luthier tooling project. For now, it only contains the 
`EmbedIModulePlugin`, used by Luthier dynamic instrumentation tools during compilation. In the future, 
we will introduce other instrumentation-related compiler plugins here (e.g. static instrumentation of LLVM IR/MIR).

Besides compiler plugins, this project also houses any code shared between the compiler plugins and the main Luthier
project. This code is header-only to accommodate using different LLVM versions in both the compiler plugins and the
tooling projects.