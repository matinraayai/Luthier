# High-level Structure of Luthier
Luthier consists of the following components:

1. **The Controller**: Under ```luthier``` namespace, in [```src/luthier.cpp```](../src/luthier.cpp). It implements 
the high-level interface for the tool writer, and initializes and finalizes all essential components as needed.
2. **HSA Intercept Layer**: implemented in [```src/hsa_intecept.hpp```](../src/hsa_intercept.hpp) and 
[```src/hsa_intercept.cpp```](../src/hsa_intercept.cpp). It intercepts all the HSA functions called by the target 
application, and (if enabled), performs a callback to both the user and the tool.
3. **HIP Intercept Layer**: implemented in [```src/hip_intercept.hpp```](../src/hip_intercept.hpp) and
[```src/hip_intercept.cpp```](../src/hip_intercept.cpp). It intercepts all HIP functions called by the HIP runtime,
and (if enabled) performs a callback to both the user and the tool.
4. **Target Manager**: implemented in [```src/target_manager.hpp```](../src/target_manager.hpp) and
   [```src/target_manager.cpp```](../src/target_manager.cpp). It records the HSA ISA of the agents attached to the system
and creates LLVM-related data structures for each unique ISA available to the runtime, and caches them. 
Any other component can query this information.
5. **HSA Abstraction Layer** implemented under ```luthier::hsa``` namespace. Provides a useful, 
object-oriented abstraction over the C-API of the HSA library, to provide a less-verbose interface to Luthier, 
and implement any required features not currently implemented in HSA (e.g. indirect function support). 
APIs called by this layer are not intercepted. Other components should not use the HSA library directly.
6. **CodeLifter**: can be found in [```src/disassembler.hpp```](../src/disassembler.hpp) and
[```src/disassembler.hpp```](../src/disassembler.cpp). It disassembles every ```hsa::ExecutableSymbol``` using LLVM, 
and caches the results. It is the only component allowed to create ```hsa::Instr``` objects. 
7. **Code Generator**: Under [```src/code_generator.hpp```](../src/code_generator.hpp) and 
[```src/code_generator.cpp```](../src/code_generator.cpp). After the user describes the instrumentation task, 
the code generator creates the instrumented code objects via analysing the disassembled instructions of the target code
object in LLVM.


[//]: # (   - HIP/HSA API tables are automatically generated via the python ```hip_intercept_gen.py``` and)

[//]: # (     ```hsa_intecept_gen.py```.)

[//]: # (   - As of right now, HIP uses ```dlsym``` to capture necessary HIP APIs. We are in the process of migrating to)

[//]: # (     Gotcha, to provide a dynamic way of turning unnecessary API captures on/off. HIP API tables are promised)

[//]: # (     in ROCm 6.0+.)

[//]: # (   - HSA uses the ```libroctool.so``` to capture the HSA API table. Dynamically turning capturing on/off is)

[//]: # (     currently being worked on.)