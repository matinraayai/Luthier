# Lifting AMD HSA Kernel Descriptors During Code Discovery
This section describes the current state of how the kernel descriptor is represented inside LLVM MIR, 
and how each bit or entry in the kernel descriptor is obtained by LLVM when printing the MIR. This
section has to be kept uptodate with LLVM upstream.

Luthier's code discovery pass uses this information to lift a kernel descriptor into its correct
representation at the `MachineFunction` level.

### `group_segment_fixed_size`
The `LDSSize` entry inside the `llvm::AMDGPUMachineFunction` represents this entry. 
It can only be set by defining the `"amdgpu-lds-size"` attribute for the corresponding IR 
`Function` prior to creating the kernel's `MachineFunction`.
### `private_segment_fixed_size`

