# Lifting AMD HSA Kernel Descriptors During Code Discovery
This section describes the current state of how the kernel descriptor is represented inside LLVM MIR, 
and how each bit or entry in the kernel descriptor is obtained by LLVM when printing the MIR. This
section has to be kept uptodate with LLVM upstream.

Luthier's code discovery pass uses this information to lift a kernel descriptor into its correct
representation at the `MachineFunction` level.

## `group_segment_fixed_size`
The `LDSSize` entry inside the `llvm::AMDGPUMachineFunction` represents this entry. 
It can only be set by defining the `"amdgpu-lds-size"` attribute for the corresponding IR 
`Function` prior to creating the kernel's `MachineFunction`. The value is read into 
`SIProgramInfo`'s `LDSSize` before being used by AMDGPU's `AsmPrinter` to emit the KD.
## `private_segment_fixed_size`
The value is held by `llvm::SIFunctionResourceInfo::PrivateSegmentSize`, first 
calculated by the `llvm::AMDGPUResourceUsageAnalysis` using the `MachineFrameInfo` of 
the kernel. It is calculated as the sum of:
1. The stack size from the `getStackSize()` method of the `MachineFrameInfo`; plus
2. A fixed value called `AssumedStackSizeForDynamicSizeObjects` set by the 
   `amdgpu-assume-dynamic-stack-object-size` command line option, only if the kernel has a 
   dynamically sized stack (i.e., `hasVarSizedObjects` of `MachineFrameInfo` returns `true`) 
3. Maximum frame alignment obtained via `getMaxAlign` of the frame info, only if
   `isStackRealigned` returns `true`.

The `AsmPrinter` first emits this value from the resource info into an `llvm::MCSymbol` before
using the symbol value to populate the `group_segment_fixed_size` field.

## kernarg_size
The size of the kernel argument is deduced by the number of explicit arguments and implicit
arguments of the kernel function:
- Explicit argument sizes are calculated directly from the kernel's IR function prototype.
- Implicit arguments are mainly controlled by IR attributes. Emission of the implicit args
  differ between code object V4 and V5. In COV5 since the implicit arg entries are at fixed 
  offsets, the compiler might end up asking for more implicit argument range than in needs to. 

On targets that support preloading kernel arguments into user SGPRs, some implicit arguments
might end up being added to the kernel's function prototype and marked with the attributes 
`"amdgpu-hidden-argument"` or `"inreg"`. Some alignments from the target's data layout is
also considered for aligning the arguments. The total kernel argument size is aligned to
8 bytes for AMDHSA and 4 bytes for other targets.

### 

