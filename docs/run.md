# Running Luthier-based Tools
To run tools developed using Luthier, run the following command:
```shell
HIP_ENABLE_DEFERRED_LOADING=0 LD_PRELOAD=${LUTHIER_PATH}:${TOOL_PATH} ${APPLICATION_CMD}
```
- ```HIP_ENABLE_DEFERRED_LOADING=0``` forces the HIP runtime to eagerly load any FAT binaries statically loaded into
the runtime, and not defer the loading to when the code object is used.
- ```LD_PRELOAD``` loads the specified shared objects before loading any other library.
- ```${LUTHIER_PATH}``` is the path to the ```libluthier.so```
- ```${TOOL_PATH}``` is the path to the tool built as a shared object using Luthier
- ```${APPLICATION_CMD}``` is the command one uses to launch the target application