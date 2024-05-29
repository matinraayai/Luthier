# Inserting Device Instrumentation Function Calls in Luthier

Luthier lets tool writers insert calls to instrumentation device functions at any point in the original application.
Unlike writing sub-target specific instrumentation code with LLVM's MIR, device functions allow tool writers to write
portable instrumentation code at the (potential) cost of using slightly more registers.

Although both written in HIP, instrumentation functions differ from "normal" device functions, in one key aspect:
**Instrumentation functions accesses its caller's register state if the tool writer requires it**.
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