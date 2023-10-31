# Code Structure of Luthier
Luthier consists of the following components:

1. The Controller: Under ```luthier::impl``` namespace. Implements the high-level interface for the tool writer,
and initializes and finalizes all essential components as needed.
2. HIP/HSA Intercept Layers: Under ```hip_intecept.*pp``` and ```hsa_intercept.*pp```, respectively. In charge of
capturing HIP and HSA API calls of the target applications.
   - HIP/HSA API tables are automatically generated via the python ```hip_intercept_gen.py``` and 
   ```hsa_intecept_gen.py```. 
   - As of right now, HIP uses ```dlsym``` to capture necessary HIP APIs. We are in the process of migrating to
   Gotcha, to provide a dynamic way of turning unnecessary API captures on/off. HIP API tables are promised
   in ROCm 6.0+
   - HSA uses the ```libroctool.so``` to capture the HSA API table. Dynamically turning capturing on/off is 
   currently being worked on.

3. Context Manager: Under ```context_manager.*pp```. Is in charge of recording the information regarding the
GPU HSA Agents available on the system, and retrieving their information either using HSA or AMD Comgr.
4. Disassembler: Under ```disassmbler.*pp```. Is in charge of disassembling the code object, whether they 
are on the host only, or part of a loaded code object on the device. Creates ```Instr``` objects for the tool
writer to use. The disassembler can also identify the ```hsa_agent_t```, ```hsa_executable_t``` and the 
```hsa_executable_symbol_t``` associated with the instruction.
5. Code Generator: Under ```code_generator.*pp```. Is in charge of queueing instrumentation tasks, and carrying 
them out after a user's HIP/HSA callback. Also provides instruction assembling APIs.
6. Code Object Manipulator: Under ```code_object_manipulation.*pp```. Provides a range of APIs for reading/writing
AMDGPU code objects.