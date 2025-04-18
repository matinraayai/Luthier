# High-level Folder Structure of Luthier
The Luthier project consists of the following folders:
- [`.devcontainer/`](../.devcontainer/): Contains the [`devcontainer.json`](../.devcontainer/devcontainer.json) file.
It can be used with IDEs like CLion or VSCode to run and develop Luthier inside a container. By default, it uses
the dev container shipped by us, but you can build the container yourself in the [`dockerfiles`](../dockerfiles) folder.
- [`.vscode`](../.vscode): contains compile commands which can be used with VSCode to enable Intellisense code 
completion.
- [`compiler-plugins`](../src/lib/compiler-plugins): contains the LLVM compiler plugins for Luthier. It is currently used
by Luthier tools; In the future, it will contain compiler plugins used in static instrumentation as well.
- [`dockerfiles`](../dockerfiles): contains the Dockerfiles of the Luthier project to run and develop Luthier-based
tools.
- [`docs`](../docs): contains the Luthier documentation.
- [`examples`](../examples): contains example Luthier tools.
- [`include`](../include): contains the public-facing API of Luthier.
- [`scripts`](../src/scripts): contains scripts for generating HIP and HSA callbacks for Luthier.
- [`src`](../src): contains the Luthier source code.
- [`tests`](../tests): contains integration tests.

[//]: # (Luthier consists of the following components:)

[//]: # ()
[//]: # (1. **The Controller**: Under ```luthier``` namespace, in [```src/luthier.cpp```]&#40;../lib/luthier.cpp&#41;. It implements )

[//]: # (the high-level interface for the tool writer, and initializes and finalizes all essential components as needed.)

[//]: # (2. **HSA Intercept Layer**: implemented in [```src/hsa_intecept.hpp```]&#40;../lib/hsa_intercept.hpp&#41; and )

[//]: # ([```src/hsa_intercept.cpp```]&#40;../lib/hsa_intercept.cpp&#41;. It intercepts all the HSA functions called by the target )

[//]: # (application, and &#40;if enabled&#41;, performs a callback to both the user and the tool.)

[//]: # (3. **HIP Intercept Layer**: implemented in [```src/hip_intercept.hpp```]&#40;../lib/hip_intercept.hpp&#41; and)

[//]: # ([```src/hip_intercept.cpp```]&#40;../lib/hip_intercept.cpp&#41;. It intercepts all HIP functions called by the HIP runtime,)

[//]: # (and &#40;if enabled&#41; performs a callback to both the user and the tool.)

[//]: # (4. **Target Manager**: implemented in [```src/target_manager.hpp```]&#40;../lib/target_manager.hpp&#41; and)

[//]: # (   [```src/target_manager.cpp```]&#40;../lib/target_manager.cpp&#41;. It records the HSA ISA of the agents attached to the system)

[//]: # (and creates LLVM-related data structures for each unique ISA available to the runtime, and caches them. )

[//]: # (Any other component can query this information.)

[//]: # (5. **HSA Abstraction Layer** implemented under ```luthier::hsa``` namespace. Provides a useful, )

[//]: # (object-oriented abstraction over the C-API of the HSA library, to provide a less-verbose interface to Luthier, )

[//]: # (and implement any required features not currently implemented in HSA &#40;e.g. indirect function support&#41;. )

[//]: # (APIs called by this layer are not intercepted. Other components should not use the HSA library directly.)

[//]: # (6. **CodeLifter**: can be found in [```src/disassembler.hpp```]&#40;../lib/code_lifter.hpp&#41; and)

[//]: # ([```src/disassembler.hpp```]&#40;../lib/code_lifter.cpp&#41;. It disassembles every ```hsa::ExecutableSymbol``` using LLVM, )

[//]: # (and caches the results. It is the only component allowed to create ```hsa::Instr``` objects. )

[//]: # (7. **Code Generator**: Under [```src/code_generator.hpp```]&#40;../lib/code_generator.hpp&#41; and )

[//]: # ([```src/code_generator.cpp```]&#40;../lib/code_generator.cpp&#41;. After the user describes the instrumentation task, )

[//]: # (the code generator creates the instrumented code objects via analysing the disassembled instructions of the target code)

[//]: # (object in LLVM.)


[//]: # (   - HIP/HSA API tables are automatically generated via the python ```hip_intercept_gen.py``` and)

[//]: # (     ```hsa_intecept_gen.py```.)

[//]: # (   - As of right now, HIP uses ```dlsym``` to capture necessary HIP APIs. We are in the process of migrating to)

[//]: # (     Gotcha, to provide a dynamic way of turning unnecessary API captures on/off. HIP API tables are promised)

[//]: # (     in ROCm 6.0+.)

[//]: # (   - HSA uses the ```libroctool.so``` to capture the HSA API table. Dynamically turning capturing on/off is)

[//]: # (     currently being worked on.)