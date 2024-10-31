# Inserting Device Code Instrumentation Hooks in Luthier

## Introduction

Luthier lets tool writers insert device code instrumentation "hooks" at any point in the original application.
Hooks give read/write access to the ISA visible state of the insertion point, and allow tools to express
instrumentation logic in plain HIP. Luthier then compiles the hook into machine code while guaranteeing the compiled
code
does not conflict with the application state. This makes hooks much more flexible and easier to work with compared to
using LLVM's MIR builder API, and allows portability between different GPU architectures.

## Implementation Details and Lifetime of a Hook

Under the hood, hooks in a Luthier tool are written as device functions in HIP with the `used` attribute set to prevent
the compiler from removing it. During compilation, the clang compiler is asked to also generate the LLVM bitcode of 
the tool device code and embedd it in all the HSA code objects in the tool's FAT binary using the option 
`-Xclang -fembed-bitcode=all`.

During execution, the HIP runtime parses the HIP FAT binary of the tool and loads the necessary tool code object(s) via
ROCr as usual. Luthier can then easily access the embedded LLVM bitcode of the tool code object by inspecting its
storage ELF without requiring a dedicated loader. The `ToolExecutableLoader` in Luthier is tasked with loading the bitcode
into an `llvm::Module`. As the embedded bitcode is un-optimized, it needs to first go through IR-level optimizations 
. The optimized IR will be the starting point for generating instrumented code. The LLVM Module is cached by
the `ToolExecutableLoader` for future usage and is destroyed once the tool loaded code object's HSA executable is
destroyed.

It's important to emphasize that Luthier **does not** use the instrumentation functions already compiled by HIP, nor
tries to lift them into LLVM MIR. Starting from the LLVM IR of the instrumentation code compared to its lifted MIR
version has the following benefits:

1. LLVM IR is in SSA form and will be transformed to an SSA form of MIR in the code generation pipeline. The lifted MIR
   does no such thing. The SSA form of MIR, combined with the list of live physical registers at each instrumentation
   point,
   guides the register allocation pass to optimize its register usage and generate spill code when needed.
2. LLVM IR can use the function inlining passes, which is not available as a Machine Function Passes in LLVM.
   This is important for hooks, as they are the inlined version of the device functions generated in the tool writing/
   compilation stage. Inlining also eliminates any unnecessary stack access from the device function early on.
3. LLVM IR contains the stack frame information of each function, which otherwise gets lost after the assembly
   is printed and is not trivial to recover by lifting into MIR. LLVM IR makes it easy to create stack objects in the
   instrumentation logic that avoid collision with the original kernel's stack.

Note that although the text section of the tool loaded code object is not used by Luthier, it is still required for
its device code to be fully compiled so that the HIP/HSA runtimes is forced to load it and initialize its static device
variables for aggregating instrumentation results.

An `InstrumentationTask` passed to Luthier specifies a list of instrumentation points as well as device hooks to
be inserted at those points. The `CodeGenerator` will locate the device hooks inside the Module obtained from the
tool's loaded code object, and performs a "deep copy" of them into a brand-new LLVM Module,
which we call "instrumentation module". The "deep copy" operation first analyzes the IR instructions of all the hooks,
and finds all Module components used by those instructions. Then any LLVM Functions used by the hooks will be deeply
copied into the instrumentation module, and any used Global Variable will only be declared as external and not deeply
copied over. This allows the ROCr runtime to correctly map the instrumented code to tool's static variables. If present,
any debug and metadata information will also be copied over to the instrumentation module.

For now all hooks in an instrumentation task must come from the same LLVM Module i.e. the same tool loaded code object.
This is to ensure no collision of debug metadata from two separate file occurs.
This requirement can be revisited in the future if needed.

After copying over the hooks, Luthier then copies the definition of all Functions present in the lifted app module to
the instrumentation module, without copying the MIR portion. For each kernel It iterates over instrumentation points in
the lifted
app Modules, and generates an `llvm::BasicBlock` for each of them.

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
           then it doesn't require a frame setup. This makes it harder to instrumentAndLoad device functions in that **DO**
           have
           a frame.
        2. If the HIP compiler is forced to emit a frame, then it becomes more tedious to instrumentAndLoad code that doesn't
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