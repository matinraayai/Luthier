# Tool Device Code in Luthier

Luthier allows two ways of loading code to carry out instrumentation:

1. The static way: If the tool's shared object has any HIP code and is annotated with `MARK_LUTHIER_DEVICE_MODULE`,
   then HIP is going to load the FAT binary associated with the tool, which will contain the optimized bitcode. The
   compiled device code by itself is not used, it only serves as a means to load static global variables (e.g. managed
   variables) to accumulate profiling data.
2. The dynamic way: In this scenario, the user can load Luthier tool bitcode directly from disk or memory. In this scenario,
if any static variables are used, they need to be manually defined

How to get a handle to instrumentation functions:
Unfortunately, HIP does not allow using pointers to device functions in host code. One way to get around this is to define
a dummy kernel with its name pointing to the actual device function, but that ends up adding more things to the bitcode
emitted by LLVM. Since there's no way to sell an illusion of having a compile time check over the pointers passed to Luthier.

Instead, I think we need to do the following:
1. If the code was loaded statically, then each device attached to the host will get an executable containing the tool code.
The LLVM bitcode stays the same because it is sub-target-agnostic, but any static variables need to be loaded for each device.
So we have:
   1. A bitcode shared among everyone
   2. A list of static variables per device, that correspond to the ones found over at the bitcode.

This means we need to create a Luthier's instrumentation Module encapsulation to contain these items. We don't need 
to keep track of other things like the device functions, since we don't use them in any shape or form. 

For dynamic bitcode, since there are no static variables to keep track of, we need to be like HIP; Meaning each static
variable will get an extern reference to a dynamically allocated portion of the memory. 

For static instrumentation modules, we need to provide a way for the tool to retrieve it. There might be multiple 
static code objects that belong to Luthier, but we use the `hip_cuid` symbol to identify if multiple executables are the
same but for different devices.


If lazy loading is enabled, then tool writers can only perform instrumentation during the packet submit event. This is 
because at any other point (e.g. right after executable freeze) the HIP runtime (or any other runtime for that matter) might
need to write the address of device variables into the static variables, including managed variables. Managed variables
are not implemented over at HSA, they are implemented in the HIP layer. Even though the HSA standard does have a notion of program allocation
variables or code objects, it does not implement it. Instead, when the code object is loaded to HIP, the runtime will 
allocate the variable dynamically (using HSA, ironically), writes the initial value of that variable into said chunk, and
then writes a pointer of that allocated variable into the static symbol which is used by the code object. Hence managed variables
are essentially a pointer to a dynamically allocated managed pointer.

Because of this initialization, there's no guarantee that when the user gets their hands on the HSA executable that the 
managed variable pointer is initialized. Therefore, Luthier cannot know where the managed variable's pointer is, hence
it cannot load the instrumented executable correctly.

Each kernel launch operation in HIP first ensures that all static managed variables on the launched kernel's device gets
initialized. Which means any code object that has managed variables will have to be loaded, frozen, and populated manually
by HIP. Luthier will get multiple executable freeze callbacks which only correspond to a single kernel launch. 

If eager loading is enabled, same issue persists; The executable freeze happens earlier, but by that time, no managed variable
is initialized.

Tool code must have CUID enabled.