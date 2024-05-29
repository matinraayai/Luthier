# Inserting Device Code Instrumentation Hooks in Luthier

## Introduction
Luthier lets tool writers insert device code instrumentation "hooks" at any point in the original application. 
Hooks give read/write access to the ISA visible state of the insertion point, and allow tools to express
instrumentation logic in plain HIP. Luthier then compiles the hook into machine code while guaranteeing no conflict
with the application state occurs. This makes hooks much more flexible and easier to work with compared to 
using LLVM's MIR builder API, and allows easier portability between different GPU architectures.

## Implementation Details
Under the hood, hooks are written as noinline device functions in HIP. During tool compilation, the clang compiler is
asked to embedd the LLVM bitcode of the tool compilation module in the generated code object.

During instrumentation, the HIP runtime loads the instrumentation code object as usual, which gives Luthier access to 
its storage ELF in host memory, and hence the LLVM bitcode of the tool through ROCr. Luthier then asks the 
`CodeObjectManager` to load the bitcode into an `llvm::Module`, which serves as a starting point for generating 
instrumented code. The LLVM Module is cached by the `CodeObjectManager` for future usage.

It's important to emphasize that Luthier **does not** use the instrumentation functions already compiled by HIP, nor 
tries to lift it into LLVM MIR. Starting from the LLVM IR of the instrumentation code allows more flexibility, 
gives access to passes required by hooks including function inlining, generates stack frame objects 
which does not collide with the original kernel in case of spilling, and as we will demonstrate later, is easier to 
express register usage requirements once we get past the instruction selection stage.

This, however, does not undermine the importance of the tool code object loaded by the HIP/HSA runtime, since the code
object:
1. Serves as a convenient means for accessing the LLVM IR of the instrumentation code. 
2. Makes the HIP runtime initialize the static device variables defined by the tool writer to aggregate instrumentation 
results.



1. Unlike device functions, hooks guarantee that the ISA visible state of the insertion point stays intact when they 
take over execution. Device functions cannot guarantee this as the passed arguments might require some registers to 
be spilled to the stack.
2. Hooks are essentially force inlined functions; Luthier ensures that the arguments to hooks are passed in registers 
that are both not in use by the app or queried by the hook.  

This is a challenging task to do efficiently with the normal HIP compiler pipeline, since Instrumentation functions
written in HIP adhere to a version of the C-calling convention, as explained in
[AMDGPU backend's calling conventions](https://llvm.org/docs/AMDGPUUsage.html#calling-conventions) and
[LLVM's list of supported calling conventions](https://llvm.org/docs/LangRef.html#calling-conventions).

1. The compiler, based on the calling convention used, assumes a set of caller-saved registers are free for usage.
   If Luthier was to patch the generated device function as-is, it has to:
    1. First enable the private segment buffer and flat scratch in the kernel (if not already enabled by the original
       app)
    2. Spill the caller-saved registers to the stack, as well as the return address registers, stack/frame registers,
       and private segment registers (if they are used by the app for a different purpose)
    3. Finally, set up a call to the instrumentation function, based on its calling convention.

   This approach has the following shortcomings:

    1. It spills more registers than necessary; It does not take advantage of any dead registers at the instrumentation
       point that can be used for instrumentation.
    2. It uses more registers than necessary. For example, setting up a call frame in GFX908 might require the setup of
       an additional 10 SGPRs (4 SGPRs for the private segment buffer, 1 SGPR for the stack pointer, 1 SGPR for the
       frame pointer, 2 SGPRs for setting up the target jump address, 2 SGPRs for saving the return address). 
       This is unacceptable for instrumentation, especially given that registers are a very scarce resource in 
       high-performance kernels.
    3. As the instrumentation function's frame information (i.e. its stack operands in MIR) is lost once it is
       emitted into plain assembly, it becomes more tedious for usage "as is" for instrumentation:
        1. As from the HIP compiler's point of view, if the instrumentation function itself doesn't call other
           functions,
           then it doesn't require a frame setup. This makes it harder to instrument device functions in that **DO**
           have
           a frame.
        2. If the HIP compiler is forced to emit a frame, then it becomes more tedious to instrument code that doesn't
           have a frame already setup as we have to do it ourselves, not to mention clobbering of the extra registers.
    4. Enabling the flat scratch, private segment buffer, and the private
       segment offset registers potentially changes the kernel's initial execution state, requiring extra diligence
       to correct and set up on top of the epilogue inserted by the LLVM Codegen pipeline.

2. AMDGPU device functions are not designed to access their callers' parents all the way up to their LLVM IR
   representation.  
   Besides efficiency concerns
   of using scratch memory, this scheme makes it hard to keep track of where each register is, and generate code to
   access/modify the state.

Hence instrumentation functions in HIP should be regarded as
Insertion of device functions in Luthier
Before we delve into how Luthier "inserts callbacks" into the application code we first need to define what
At first glance inserting calls to instrumentation functions might seem; The tool writer expects 