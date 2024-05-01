# What is Luthier?
Luthier is a dynamic instrumentation tool for AMD GPUs at runtime, similar to [NVBit](https://github.com/NVlabs/NVBit)
for NVIDIA GPUs similar functionalities. It gives tool writers the ability to:
- View and analyze the instructions in a kernel.
- Insert calls to (multiple) instrumentation functions.
- Remove/Modify instructions.
- Query the ISA visible state on the device side.

# What AMD GPU Applications Luthier Supports?
Luthier should support any AMD GPU programming API in Linux that is backed by the ROCm implementation of the 
[HSA](https://hsafoundation.com) library, named [ROCr](https://github.com/RadeonOpenCompute/ROCR-Runtime/). 
This should include HIP, OpenMP, OpenCL, and AMDGPU.jl. 

On the HIP side, it supports both statically (lazily)-loaded modules,
dynamically-loaded modules, and Just-in-Time (JIT)-compiled HIP modules. 

Luthier does not support the [Platform Abstraction Layer (PAL)](https://github.com/GPUOpen-Drivers/pal)-based backends
and/or Windows.

# How to write a Tool with Luthier:
The following is a high-level example of how a tool for Luthier is written:
```c++
#include <luthier.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>

MARK_LUTHIER_DEVICE_MODULE

__managed__ int globalCounter = 0;

 __device__ __noinline__ extern "C" void instrumentation_kernel() {
    *globalCounter = *globalCounter + 1;
}

LUTHIER_EXPORT_FUNC(instrumentation_kernel)

void instrumentKernelLaunchCallback(hsa_kernel_dispatch_packet_t packet) {
     if (not luthier_is_instrumented(packet->kernel_object)) {
         std::vector<Instr> instrVec = luthier_disassemble_kernel_object(packet->kernel_object);
         for (const auto& instr: instrVec) {
             luthier_insert_call(instr, LUTHIER_GET_EXPORTED_FUNC(instrumentation_kernel), LUTHIER_IPOINT_AFTER);
         }
        luthier_override_with_instrumented(dispatchPacket);
     }
}

void luthier_at_init() {
    fprintf(stdout, "Instruction Counter Tool is launching.\n");
}


void luthier_at_term() {
     fprintf(stdout, "Number of instructions executed: %d\n", globalCounter);
     fprintf(stdout, "Instruction Counter Tool is terminating!\n");
}


void luthier_at_hsa_event(hsa_api_args_t* args, luthier_api_phase_t phase, hsa_api_id_t api_id) {
    if (phase == LUTHIER_API_PHASE_ENTER) {
        if (api_id == HSA_EVT_ID_KERNEL_LAUNCH) {
            fprintf(stdout, "<Intercepted kernel launch.\n");
            instrumentKernelLaunchCallback(args->hsa_evt_kernel_launch.dispatch_packet);
            }
    }
}

void atHipEvt(void* args, luthier_api_phase_t phase, int hip_api_id) {}
```
A few notable points:
1. When making device-side instrumentation tools with Luthier, the macro ```MARK_LUTHIER_DEVICE_MODULE``` must be called
once. This will help Luthier identify the Hip Module compiled with the tool and keep track of it.
2. A device function that will be injected between instructions in Luthier needs to be "exported" using 
```LUTHIER_EXPORT_FUNC```. This will prevent the HIP compiler from eliminating the device function that "seemingly" is
not used inside a kernel.
    ```LUTHIER_EXPORT_FUNC``` is just a kernel that wraps around the instrumentation function, and calls it without 
    giving it any arguments. The kernel itself is not used to launch workloads on the GPU. However, Luthier uses the 
    "Shadow Host Pointer" associated with the wrapper kernel to keep track of its instrumentation function. 
    the compiler from eliminating the device function. See [Code Object Manager](#code-object-manager).
3. The tool writer needs to define two sets of callbacks, one for HSA and one for HIP. In the example tool, the HIP
   callback doesn't do anything. The HSA callback, on kernel launch, provides the ```hsa_kernel_dispatch_packet_t```, 
   which is used to launch the kernel, and uses it to instrument the "kernel object" (address to where the kernel is
   located on the device).
4. The tool writer can get lift the instruction in the kernel object with the ```luthier_disassemble_kernel_object```
   function.
5. Finally, to insert a call to the instrumentation function, one can use the ```luthier_insert_call``` function.
   ```LUTHIER_GET_EXPORTED_FUNC``` will get a handle to the wrapper kernel of the instrumented function. See 
   [Code Object Manager](#code-object-manager) for why.
6. Optionally, one can add arguments to the instrumentation function, but here is not required. Unlike NVBit, Luthier
   instrumentation functions can use both statically and dynamically allocated memory on the GPU directly. To see
   how this is done, see [Code Object Manager](#code-object-manager).

As of right now, writing tools in HIP is only supported; But we plan to add more ways of writing tools for Luthier in 
the future.

# How does Luthier work?
Luthier works similarly to NVBit, but with a few key differences in design/abstractions required to work 
with AMD GPUs. Before diving into how different Luthier components work, we define the following terminology:
- __Target Application__: An AMD GPU applications to be instrumented. At the very least it contains API-calls to HIP/
HSA. It most likely loads AMD GPU code objects and launches kernels on an AMD GPU, but it is not strictly
required to do so.
- __Instrumentation Function__: A HIP ```__device__``` function written by the tool writer. It can be injected before or
after instructions of a kernel.
- __Target Kernel__: The un-instrumented version of the kernel to be instrumented by Luthier.
- __Instrumented Kernel__: The instrumented version of a target kernel.

## The HIP/HSA API Interception Layer
Similar to NVBit, Luthier uses the ```LD_PRELOAD``` trick to load before any of the ROCm libraries. When loaded,
Luthier will initialize the HIP/HSA interception layers to capture their API calls, similar to the Driver Interposer in 
NVBit.

Initializing the HSA API Interceptor consists of the following steps:
1. __The original HSA API tables are first captured and saved__. HSA API tables are a set of structures containing
function pointers to all the APIs defined in the AMD HSA library. Internally, instead of calling API functions directly,
HSA first uses an API table to look up the pointer of the function it should use, and then calls it.
    
    During startup, a pointer to that internal HSA API table is provided to any AMD ROCm-based tool (including Luthier) by 
the ROCTracer library, via the ```OnLoad``` and ```OnUnLoad``` methods. The HSA Interceptor in Luthier saves a copy of
this table internally to be used later.

2. __The HSA API tables are overwritten to point to the table provided by Luthier__. The Luthier HSA API table entries
include logic to provide callbacks to the tool writer, and also calling the original function meant to be called in the
first place.

Since as of right now, the ROCTracer library does not provide an API table for HIP to capture inside a 3rd party tool,
Luthier captures the HIP API by ```dlsym```-ing the ```libamdhip64.so``` library and duplicating the HIP C API, to 
override the ones provided by HIP. The duplicated functions notify the tool writer by using a callback.

The indirection provided by API tables is superior to the ```dlsym``` approach, as it can be adjusted dynamically at
runtime. For example, with the HSA API tables, one can only choose to capture the ```hsa_queue_create``` function, and 
let every other API to point to its intended target, hence reducing some overhead. This is not feasible with ```dlsym```
for HIP. It is worth noting that 
the API table approach is not novel, and is most likely using it internally
by NVBit, via the undocumented 
[```cuGetExportTable```](https://forums.developer.nvidia.com/t/cudagetexporttable-a-total-hack/20226)
function.

There are key differences between the Driver Interposer in NVBit and the Interception layer in Luthier, which includes
__Proxy HSA Queues__ certain "events" that do not necessarily correlate with an API call, which will be discussed later.


### Why Not Use the RocTracer and RocProfiler APIs for Capturing HIP/HSA APIs?
There are three reasons Luthier re-implemented what was available in RocTracer and RocProfiler:
1. The tracing and profiling libraries enforce ```const```-ness of their callback arguments, preventing an 
instrumentation tool like Luthier to modify them.
2. The Tracing API does not capture a set of necessary functions required for Luthier to function, including the family 
of ```__hipRegister``` functions called during startup of a HIP application.
3. Some internal functionality of Luthier (e.g. loading instrumentation code) requires HIP APIs to be prevented from 
executing. The RocTracer API does not allow for such freedom.

__At the time of writing this document, the ROCm profiler team is working on overhauling the tracing/profiling APIs 
in the ROCm stack.__ The "Rocprofiler V2" is going to have significant improvements including HIP API table support, 
and improved support for HSA proxy Queues. We are in contact with them to make sure their design accommodates Luthier
and other potential tools in the future.

## Code Object Manager
The code object manager keeps track of the following and their respective code objects:
- The HSA executables of the instrumented kernels, plus their HSA executable symbols (and their kernel descriptors)
- The HSA executables of instrumentation functions loaded from the Luthier tool, where their instrumentation functions
are located (as of right now HSA doesn't support indirect function symbols), as well as the HSA executable symbol
associated with the wrapper kernels of the instrumentation function.


### Tool Code Object Loading
Unlike NVBit, Luthier doesn't have a dedicated "Tool Functions Loader". Luthier tools are compiled with HIP, therefore
during startup, they use the ```__hipRegister``` function family (```__hipRegisterFatBinary```, 
```__hipRegisterFunction```, ```__hipRegisterManagedVar```, ```__hipRegisterSurface```, ```__hipRegisterTexture```,
and ```__hipRegisterVar```) to load their FAT binary to the HIP runtime.

Internally, Luthier modifies the normal loading procedure of FAT Binaries. It first identifies if the FAT Binary being
registered is part of a Luthier tool. The macro ```MARK_LUTHIER_DEVICE_MODULE``` creates a dummy static managed variable
in the FAT Binary associated with a Luthier tool, which can be easily identified when the arguments to 
```__hipRegisterManagedVar``` are captured.

If a Luthier FAT binary is identified, its FAT binary and all its static variable types are allowed to register normally. 
The kernels, however, (arguments of ```__hipRegisterFunction```) are not registered with the HIP runtime, since the 
kernels in Luthier modules are not meant for launching in the first place. Instead of  
registering the kernels with the HIP runtime, the kernels are registered with the Code Object Manager to keep track of them.

Other FAT binaries that don't belong to Luthier are loaded normally.

Using the HIP runtime to load instrumentation functions associated with Luthier has the following benefits:
1. Not using the HIP runtime to load Luthier FAT Binaries would mean replicating the same loading logic on the Luthier
   side (which in and out of itself is very complicated task), 
   diverting precious development and maintenance time away from more important parts of Luthier.
2. Registering with the HIP runtime means that (unlike NVBit), 
   the tool writers can freely use HIP language constructs like static and
   dynamic, managed and device variables in their instrumentation functions. It also means that the Code Object Manager
   can use the same "Shadow Host Pointer" of the constructs to keep track of them. For example, when Adding arguments to 
   their instrumentation functions, the tool writer doesn't need to use the "string" name of the arguments they want to 
   add, they can simply use the variable directly. It also pushes issues with tool variables 
   not being defined to compile time. A similar approach is used with referring to device function, using the 
   ```LUTHIER_GET_EXPORTED_FUNCTION``` macro.
   

The ```__hipRegister``` function family is meant to load static code objects to the HIP runtime. The HIP runtime has a 
policy of loading the FAT Binaries in a "lazy" fashion, meaning that it will only load a FAT binary into HSA ONLY IF 
it requires to use a variable or a kernel associated with it. The only exception to this rule is if a static managed
variable is included in the device module, which will force the module to be loaded as soon as any kernel is launched 
from HIP. This is a problem for Luthier, as it means the instrumentation functions will not be loaded onto the device.

To remedy this issue, the ```MARK_LUTHIER_DEVICE``` module defines a managed dummy variable in Luthier FAT Binaries.
Besides being used to identify FAT Binaries of Luthier, it is used to force the HIP runtime to load Luthier modules
eagerly. To have control over when Luthier modules are loaded, a dummy ```nullptr_t``` kernel is called by Luthier in
HIP, triggering the loading of Luthier HIP FAT binaries into HSA.


## Code Generator



## CodeLifter
CodeLifter is in charge of lifting the instructions in a kernel object, and provide a high-level view of them to the tool writer.


## Context Manager
Context Manager is in charge 


## How Does Luthier Instrument AMD GPU Device Code?



![](Luthier%20Instrumentation%20Diagram.png)

## Why Not Use HIP for Instrumentation?


## If Luthier Doesn't Require the Target Application to Use HIP, Why Capture the HIP APIs in The First Place?



# How 


